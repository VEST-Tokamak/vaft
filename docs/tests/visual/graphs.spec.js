const { test, expect } = require('@playwright/test');

// The generated interactive graphs (#1646). These drive the real snapshot the
// build generated, so they name modules that exist in every VAFT tree rather
// than counts that change with each commit.

const count = (page) => page.locator('[data-vg-count]');

async function openExplorer(page, hash = '') {
  await page.goto(`reference/dependency-graph/${hash}`);
  await expect(count(page)).toContainText(/\d+ nodes/);
}

async function search(page, query) {
  const box = page.locator('[data-vg-search]');
  await box.fill(query);
  await box.press('Enter');
}

test('the dependency explorer initializes on the package view', async ({ page }) => {
  await openExplorer(page);
  await expect(page.locator('[data-vg-status]')).toBeHidden();
  await expect(page.locator('input[name="vg-level"][value="packages"]')).toBeChecked();
  await expect(page.locator('.vg-details')).toContainText('Select a node');
  const shown = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.nodes().length);
  expect(shown).toBeGreaterThan(5);
});

test('search finds a module and opens its details', async ({ page }) => {
  await openExplorer(page);
  await search(page, 'vaft.formula.geometry');
  await expect(page.locator('.vg-title')).toHaveText('vaft.formula.geometry');
  await expect(page).toHaveURL(/focus=vaft\.formula\.geometry/);
  await expect(page.locator('input[name="vg-level"][value="modules"]')).toBeChecked();
  await expect(page.locator('.vg-details')).toContainText('Dependencies (imports)');
  await expect(page.locator('.vg-details')).toContainText('Dependents (imported by)');
});

test('the neighbourhood direction restricts the view', async ({ page }) => {
  await openExplorer(page, '#level=modules&focus=vaft.formula');
  const both = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.nodes().length);
  await page.locator('input[name="vg-direction"][value="out"]').check();
  await expect(page).toHaveURL(/direction=out/);
  const out = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.nodes().length);
  expect(out).toBeLessThan(both);
  const sources = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.edges()
    .map((edge) => edge.source().id()));
  expect(new Set(sources)).not.toContain('vaft.process');
});

test('a public object links to its API page and its pinned source', async ({ page }) => {
  await openExplorer(page);
  const id = await page.evaluate(() => {
    const data = document.querySelector('.vg-root').vaftGraph.data;
    return data.api.find((entry) => entry.api_url && entry.source.line > 0).id;
  });
  await search(page, id);
  const row = page.locator(`.vg-details [data-api="${id}"]`);
  await expect(row).toHaveClass(/vg-highlight/);
  await expect(row.locator('a', { hasText: 'API documentation' })).toHaveAttribute('href', /\/reference\/api\/[^/]+\/#/);
  await expect(row.locator('a', { hasText: 'Source' }))
    .toHaveAttribute('href', /github\.com\/VEST-Tokamak\/vaft\/blob\/[0-9a-f]{40}\/vaft\/.+#L\d+-L\d+/);
});

test('without JavaScript the page still explains itself', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(`${test.info().project.use.baseURL}reference/dependency-graph/`);
  await expect(page.locator('[data-vg-status]')).toContainText('needs JavaScript');
  await expect(page.locator('.page-inner')).toContainText('an arrow from');
  await context.close();
});
