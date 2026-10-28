const { test, expect } = require('@playwright/test');

const isDesktop = (project) => project.name.startsWith('desktop-');
const isMobile = (project) => project.name.startsWith('mobile-');

test('homepage exposes equal workflow and reference entry paths', async ({ page }, testInfo) => {
  await page.goto('');
  const cards = page.locator('.entry-path-card');
  await expect(cards).toHaveCount(2);
  await expect(cards.nth(0)).toContainText('Start here');
  await expect(cards.nth(1)).toContainText('VAFT API');
  await expect(cards.nth(0)).toHaveAttribute('href', /\/workflows\/start-here\/$/);
  await expect(cards.nth(1)).toHaveAttribute('href', /\/reference\/api\/$/);
  await page.locator('.entry-paths').scrollIntoViewIfNeeded();
  await expect(page.locator('.entry-paths')).toBeInViewport();
  // Mobile Chromium rasterizes some card text differently from the stored
  // reference on macOS; the same layout and full text are verified above.
  await expect(page.locator('.entry-paths')).toHaveScreenshot('homepage-entry-paths.png', {
    maxDiffPixelRatio: isMobile(testInfo.project) ? 0.025 : 0.01,
  });
});

test('homepage hero keeps the approved message and readable theme logo', async ({ page }) => {
  await page.goto('');
  await expect(page.locator('.vaft-hero-tagline')).toHaveText(
    'Connecting Nuclear Fusion Knowledge Across Domains for Integrated Tokamak Research',
  );
  await expect(page.locator('.vaft-hero-summary')).toHaveText(
    'Machine-agnostic workflows for interoperable IMAS data and reproducible tokamak analysis.',
  );
  await expect(page.locator('.vaft-hero-logo-light')).toBeVisible();
  await expect(page.locator('.vaft-hero-logo-dark')).toBeHidden();
  await page.locator('.book').evaluate((book) => book.classList.add('color-theme-2'));
  await expect(page.locator('.vaft-hero-logo-light')).toBeHidden();
  await expect(page.locator('.vaft-hero-logo-dark')).toBeVisible();
});

test('sidebar renders eight flat thematic sections', async ({ page }) => {
  await page.goto('workflows/start-here/');
  const labels = page.locator('.nav-section-label');
  await expect(labels).toHaveText([
    'Getting started',
    'Research workflows',
    'Framework and scientific architecture',
    'Plasma physics and models',
    'VEST and IMAS data',
    'APIs and visualization',
    'Software and integrations',
    'Resources and support',
  ]);
  const chapters = page.locator('.book-summary li.chapter[data-level^="nav.v1."]');
  await expect(chapters).toHaveCount(50);
  await expect(page.locator('.book-summary li.chapter ul')).toHaveCount(0);
  // Active-page highlighting follows the entry into its new section.
  const active = page.locator('.book-summary li.chapter.active[data-level^="nav.v1."]');
  await expect(active).toHaveCount(1);
  await expect(active).toHaveAttribute('data-level', 'nav.v1.getting-started.start-here');
  await expect(active).toContainText('Start here');
});

test('active highlighting works in a reassigned reference section', async ({ page }) => {
  await page.goto('reference/vest-diagnostics/');
  const active = page.locator('.book-summary li.chapter.active[data-level^="nav.v1."]');
  await expect(active).toHaveCount(1);
  await expect(active).toHaveAttribute('data-level', 'nav.v1.vest-imas-data.vest-diagnostics');
});

test('canonical guides expose overview and related-resource contracts', async ({ page }) => {
  await page.goto('reference/database-data-sources/');
  await expect(page.locator('.guide-overview')).toBeVisible();
  await expect(page.locator('.guide-overview dt')).toHaveCount(4);
  await expect(page.locator('.related-resource-group[data-resource-kind="notebooks"]')).toBeVisible();
  await expect(page.locator('.related-resource-group[data-resource-kind="api"]')).toBeVisible();
  await expect(page.locator('.related-resource-group[data-resource-kind="data-sources"]')).toBeVisible();
  await expect(page.locator('.related-resource-group[data-resource-kind="outputs"]')).toBeVisible();
  await expect(page.locator('[data-resource-error="unknown"]')).toHaveCount(0);
});

test('legacy routes carry a canonical redirect', async ({ request }) => {
  for (const [legacy, canonical] of [
    ['guide/Installation/', '/vaft/workflows/start-here/'],
    ['guide/Profiles/', '/vaft/workflows/equilibrium-kinetic-profiles/'],
    ['guide/API_reference/', '/vaft/reference/api/'],
    ['pages/contact/', '/vaft/reference/contacts/'],
  ]) {
    const response = await request.get(legacy);
    expect(response.ok()).toBe(true);
    const html = await response.text();
    expect(html).toContain(`content="0; url=${canonical}"`);
    expect(html).toContain(`${canonical}">`);
  }
});

test('notebook outputs show artifacts and expandable provenance', async ({ page }, testInfo) => {
  await page.goto('reference/notebooks/');
  const cards = page.locator('.notebook-output');
  await expect(cards).toHaveCount(9);
  await expect(page.locator('#output-pipeline-overview img')).toHaveAttribute(
    'alt',
    /Horizontal diagram from raw DAQ/,
  );
  await page.locator('#output-pipeline-overview summary').click();
  await expect(page.locator('#output-pipeline-overview .output-provenance')).toHaveAttribute('open', '');
  await expect(page.locator('#output-pipeline-overview')).toContainText('Notebook SHA-256');
  await expect(page.locator('#output-pipeline-overview')).toContainText('deterministic-offline-diagram');

  const card = page.locator('#output-pipeline-overview');
  await expect(card).toHaveScreenshot(
    isMobile(testInfo.project) ? 'mobile-pipeline-output.png' : 'desktop-pipeline-output.png',
  );
});

test('notebook gallery remains usable at the current viewport', async ({ page }, testInfo) => {
  await page.goto('reference/notebooks/#verified-notebook-outputs');
  const gridColumns = await page.locator('.notebook-output-grid').evaluate(
    (element) => getComputedStyle(element).gridTemplateColumns.split(' ').length,
  );
  expect(gridColumns).toBe(isMobile(testInfo.project) ? 1 : 2);
  if (isMobile(testInfo.project)) {
    const documentOverflows = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
    );
    expect(documentOverflows).toBe(false);
  }
  if (isDesktop(testInfo.project)) {
    await page.locator('.body-inner').evaluate((element) => { element.scrollTop = 600; });
  }
  await expect(page.locator('#output-first-result img')).toBeVisible();
  await expect(page.locator('#output-mirnov-spectrogram img')).toBeVisible();
});
