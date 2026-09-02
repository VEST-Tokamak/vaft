const { test, expect } = require('@playwright/test');

const isDesktop = (project) => project.name.startsWith('desktop-');
const isMobile = (project) => project.name.startsWith('mobile-');

test('homepage exposes equal workflow and reference entry paths', async ({ page }) => {
  await page.goto('');
  const cards = page.locator('.entry-path-card');
  await expect(cards).toHaveCount(2);
  await expect(cards.nth(0)).toContainText('Start here');
  await expect(cards.nth(1)).toContainText('VAFT API');
  await expect(cards.nth(0)).toHaveAttribute('href', /\/workflows\/start-here\/$/);
  await expect(cards.nth(1)).toHaveAttribute('href', /\/reference\/api\/$/);
  await expect(page.locator('.entry-paths')).toHaveScreenshot('homepage-entry-paths.png');
});

test('sidebar renders two flat first-class sections', async ({ page }) => {
  await page.goto('workflows/start-here/');
  const labels = page.locator('.nav-section-label');
  await expect(labels).toHaveCount(2);
  await expect(labels.nth(0)).toHaveText('Research workflows');
  await expect(labels.nth(1)).toHaveText('Library and data reference');
  await expect(page.locator('.book-summary li.chapter[data-level^="nav.v1."]')).toHaveCount(15);
  await expect(page.locator('.book-summary li.chapter ul')).toHaveCount(0);
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
