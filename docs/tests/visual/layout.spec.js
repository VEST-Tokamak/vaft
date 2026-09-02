const { test, expect } = require('@playwright/test');

const isDesktop = (project) => project.name.startsWith('desktop-');
const isMobile = (project) => project.name.startsWith('mobile-');

test('global navigation is flat and uses unique stable levels', async ({ page }) => {
  await page.goto('workflows/start-here/');

  const chapters = page.locator('.book-summary li.chapter');
  const levels = await chapters.evaluateAll((items) =>
    items.map((item) => item.getAttribute('data-level')),
  );

  expect(levels.length).toBeGreaterThan(1);
  expect(new Set(levels).size).toBe(levels.length);
  await expect(page.locator('.book-summary li.chapter ul')).toHaveCount(0);

  await page.evaluate(() => {
    localStorage.setItem('expChapters', JSON.stringify({ '1.1': true }));
  });
  await page.reload();
  await expect(page.locator('.book-summary li.chapter ul')).toHaveCount(0);
});

test('page ToC contains h2-h3 anchors and omits pages without them', async ({ page, request }, testInfo) => {
  await page.goto('workflows/experimental-interpretation/');

  await expect(page.locator('.book')).toHaveClass(/has-page-toc/);
  await expect(page.locator('#page-toc')).toHaveCount(1);
  const hrefs = await page.locator('.page-toc-link').evaluateAll((links) =>
    links.map((link) => link.getAttribute('href')),
  );
  expect(hrefs.length).toBeGreaterThan(3);

  const validHeadingIds = await page.locator('.markdown-section h2, .markdown-section h3')
    .evaluateAll((headings) => headings.map((heading) => `#${heading.id}`));
  expect(hrefs.every((href) => validHeadingIds.includes(href))).toBe(true);

  const firstLink = page.locator('.page-toc-link').first();
  const firstHref = await firstLink.getAttribute('href');
  if (isMobile(testInfo.project)) {
    await page.locator('.page-toc-toggle').click();
  }
  await firstLink.click();
  await expect(page).toHaveURL(new RegExp(`${firstHref}$`));

  const target = page.locator(firstHref);
  await expect(target).toBeInViewport();
  const targetTop = await target.evaluate((element) => element.getBoundingClientRect().top);
  expect(targetTop).toBeGreaterThanOrEqual(-1);
  expect(targetTop).toBeLessThan(900);

  const redirectResponse = await request.get('guide/Quick_start_guide/');
  const redirectHtml = await redirectResponse.text();
  expect(redirectHtml).not.toContain('id="page-toc"');
  expect(redirectHtml).not.toContain('page-toc-toggle');
});

test('desktop renders three non-overlapping regions and a sticky ToC', async ({ page }, testInfo) => {
  test.skip(!isDesktop(testInfo.project));
  await page.goto('workflows/experimental-interpretation/');

  const boxes = await page.evaluate(() => {
    const rect = (selector) => {
      const box = document.querySelector(selector).getBoundingClientRect();
      return { left: box.left, right: box.right, top: box.top, bottom: box.bottom };
    };
    return {
      summary: rect('.book-summary'),
      body: rect('.book-body'),
      toc: rect('#page-toc'),
    };
  });

  expect(boxes.summary.right).toBeLessThanOrEqual(boxes.body.left);
  expect(boxes.body.right).toBeLessThanOrEqual(boxes.toc.left);
  expect(boxes.toc.top).toBe(0);
  expect(boxes.toc.bottom).toBe(900);

  await page.locator('.body-inner').evaluate((element) => { element.scrollTop = 1700; });
  await expect(page.locator('#page-toc')).toBeVisible();
  await expect(page.locator('.page-toc-item.active > .page-toc-link')).toHaveCount(1);
  await expect(page).toHaveScreenshot('desktop-long-page.png');

  await page.locator('.book-header .fa-align-justify').locator('..').click({ force: true });
  await expect(page.locator('.book')).not.toHaveClass(/with-summary/);
  await expect(page).toHaveScreenshot('desktop-left-nav-collapsed.png');
});

test('mobile navigation and page ToC drawers remain independent', async ({ page }, testInfo) => {
  test.skip(!isMobile(testInfo.project));
  await page.goto('workflows/experimental-interpretation/');

  const tocToggle = page.locator('.page-toc-toggle');
  const leftIsOpen = async () => (await page.locator('.book').getAttribute('class')).includes('with-summary');
  await expect(tocToggle).toBeVisible();
  if (await leftIsOpen()) {
    await page.locator('.book-header .fa-align-justify').locator('..').click();
    await expect.poll(leftIsOpen).toBe(false);
  }
  await tocToggle.click();
  await expect(page.locator('.book')).toHaveClass(/with-page-toc/);
  await expect(tocToggle).toHaveAttribute('aria-expanded', 'true');
  await expect(page).toHaveScreenshot('mobile-page-toc-open.png');

  await page.keyboard.press('Escape');
  await expect(page.locator('.book')).not.toHaveClass(/with-page-toc/);
  await expect(tocToggle).toHaveAttribute('aria-expanded', 'false');

  const initialLeftState = await leftIsOpen();
  await page.locator('.book-header .fa-align-justify').locator('..').click();
  await expect.poll(leftIsOpen).toBe(!initialLeftState);
  await expect(page.locator('.book')).not.toHaveClass(/with-page-toc/);

  await page.locator('.book-header .fa-align-justify').locator('..').click();
  await expect.poll(leftIsOpen).toBe(initialLeftState);

  await tocToggle.click();
  await expect.poll(leftIsOpen).toBe(initialLeftState);
  await expect(page.locator('.book')).toHaveClass(/with-page-toc/);

  await page.locator('.page-toc-link').first().click();
  await expect(page.locator('.book')).not.toHaveClass(/with-page-toc/);
});

test('code, Copy controls, Rouge tokens, and search marks stay legible', async ({ page }, testInfo) => {
  await page.goto('workflows/automated-pipelines/');

  const wrappers = page.locator('.code-block-wrapper');
  const codeBlocks = page.locator('.markdown-section pre:not(.mermaid-diagram)');
  await expect(wrappers).toHaveCount(await codeBlocks.count());
  await expect(wrappers.first().locator('.copy-code-button')).toHaveCount(1);

  const longPre = codeBlocks.filter({ has: page.locator('code') }).nth(4);
  const hasHorizontalOverflow = await codeBlocks.evaluateAll((blocks) =>
    blocks.some((block) => block.scrollWidth > block.clientWidth),
  );
  expect(hasHorizontalOverflow).toBe(true);

  const firstWrapper = wrappers.first();
  await expect(firstWrapper.locator('.copy-code-button')).toBeVisible();
  await firstWrapper.locator('.copy-code-button').click();
  await expect(firstWrapper.locator('.copy-code-button')).toHaveText('Copied');

  if (testInfo.project.name === 'desktop-chromium') {
    const copiedText = await page.evaluate(() => navigator.clipboard.readText());
    const codeText = await firstWrapper.locator('pre code').innerText();
    expect(copiedText).toBe(codeText);
  }

  await expect(firstWrapper).toHaveScreenshot('code-block-copy.png');
  await expect(longPre).toBeVisible();

  await page.goto('workflows/automated-pipelines/?h=pipeline');
  const markedCode = page.locator('.code-block-wrapper')
    .filter({ has: page.locator('mark[data-markjs="true"]') })
    .first();
  await expect(markedCode.locator('mark[data-markjs="true"]').first()).toBeVisible();
  await expect(markedCode).toHaveScreenshot('search-highlight.png');
});

test('Mermaid diagrams stay separate from fenced code controls', async ({ page }, testInfo) => {
  await page.goto('workflows/equilibrium-kinetic-profiles/');

  if (isMobile(testInfo.project) && (await page.locator('.book').getAttribute('class')).includes('with-summary')) {
    await page.locator('.book-header .fa-align-justify').locator('..').click();
    await expect(page.locator('.book')).not.toHaveClass(/with-summary/);
  }

  const diagram = page.locator('pre.mermaid-diagram');
  await expect(diagram).toHaveCount(1);
  await expect(diagram.locator('svg')).toBeVisible();
  await expect(diagram.locator('.copy-code-button')).toHaveCount(0);
  await expect(diagram.locator('xpath=ancestor::div[contains(@class, "code-block-wrapper")]'))
    .toHaveCount(0);

  const sizing = await diagram.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
    svgHeight: element.querySelector('svg').getBoundingClientRect().height,
  }));
  expect(sizing.svgHeight).toBeGreaterThan(80);
  expect(sizing.scrollWidth).toBeGreaterThanOrEqual(sizing.clientWidth);
  if (isMobile(testInfo.project)) {
    expect(sizing.scrollWidth).toBeGreaterThan(sizing.clientWidth);
  }

  await expect(diagram).toHaveScreenshot('mermaid-pipeline.png');

  if (isMobile(testInfo.project)) {
    await page.locator('.book-header .fa-align-justify').locator('..').click();
    await expect(page.locator('.book')).toHaveClass(/with-summary/);
  }
  await page.locator('.book-summary a[href$="/workflows/automated-pipelines/"]').click();
  await expect(page).toHaveURL(/\/workflows\/automated-pipelines\/$/);
  await expect(page.locator('pre.mermaid-diagram svg')).toBeVisible();
  await expect(page.locator('.code-block-wrapper pre.mermaid-diagram')).toHaveCount(0);
  await expect(page.locator('pre.mermaid-diagram .copy-code-button')).toHaveCount(0);
});
