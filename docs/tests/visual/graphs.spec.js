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

test('clicking an API object keeps its module selected and highlights the object', async ({ page }) => {
  await openExplorer(page, '#level=modules&focus=vaft.formula.geometry&api=show');
  const id = await page.evaluate(() => {
    const leaf = document.querySelector('.vg-root').vaftGraph.cy.nodes('.vg-api')[0];
    leaf.emit('tap');
    return leaf.id();
  });
  await expect(page.locator('.vg-title')).toHaveText('vaft.formula.geometry');
  await expect(page.locator(`.vg-details [data-api="${id}"]`)).toHaveClass(/vg-highlight/);
  const leaves = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.nodes('.vg-api').length);
  expect(leaves).toBeGreaterThan(0);
});

test('a module focus survives a round trip through the package level', async ({ page }) => {
  await openExplorer(page, '#level=modules&focus=vaft.formula.geometry');
  await page.locator('input[name="vg-level"][value="packages"]').check();
  await expect(page.locator('.vg-title')).toHaveText('vaft.formula');
  await page.locator('input[name="vg-level"][value="modules"]').check();
  await expect(page.locator('.vg-title')).toHaveText('vaft.formula.geometry');
});

test('an ordinary in-page anchor does not reset the explorer', async ({ page }) => {
  await openExplorer(page, '#level=modules&focus=vaft.formula.geometry');
  await page.evaluate(() => { window.location.hash = 'some-heading'; });
  await page.waitForTimeout(300);
  await expect(page.locator('.vg-title')).toHaveText('vaft.formula.geometry');
});

test('a real click selects the node under the pointer after the page has scrolled', async ({ page }) => {
  await openExplorer(page);
  // GitBook scrolls an inner element, not the window; Cytoscape must not use a stale canvas offset.
  await page.evaluate(() => document.querySelector('.vg-canvas').scrollIntoView({ block: 'center' }));
  await page.waitForTimeout(300);
  const before = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.pan());
  const point = await page.evaluate(() => {
    const viewer = document.querySelector('.vg-root').vaftGraph;
    const box = viewer.cy.container().getBoundingClientRect();
    const position = viewer.cy.getElementById('vaft.formula').renderedPosition();
    return { x: box.left + position.x, y: box.top + position.y };
  });
  await page.mouse.move(point.x, point.y);
  await page.mouse.down();
  await page.mouse.up();
  await expect(page.locator('.vg-title')).toHaveText('vaft.formula');
  // and moving the pointer afterwards no longer drags the view along
  const panAfterClick = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.pan());
  await page.mouse.move(point.x + 200, point.y + 120, { steps: 8 });
  await page.waitForTimeout(200);
  const panAfterMove = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.pan());
  expect(panAfterMove).toEqual(panAfterClick);
  expect(before).toBeTruthy();
});

test('without JavaScript the page still explains itself', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(`${test.info().project.use.baseURL}reference/dependency-graph/`);
  await expect(page.locator('[data-vg-status]')).toContainText('needs JavaScript');
  await expect(page.locator('.page-inner')).toContainText('an arrow from');
  await context.close();
});

// The pipeline lineage explorer (#1647).

async function openPipelines(page, hash = '') {
  await page.goto(`reference/pipeline-graph/${hash}`);
  await expect(count(page)).toContainText(/\d+ nodes/);
}

const shownIds = (page) => page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.nodes().map((n) => n.id()));

test('the pipeline explorer loads the rule graph of both pipelines', async ({ page }) => {
  await openPipelines(page);
  await expect(page.locator('input[name="vg-view"][value="rules"]')).toBeChecked();
  const ids = await shownIds(page);
  expect(ids).toContain('routine:generate_efit_ods');
  expect(ids).toContain('corrective:generate_core_profiles_ods');
});

test('pipeline switching restricts the graph', async ({ page }) => {
  await openPipelines(page);
  await page.locator('input[name="vg-pipelines"][value="routine"]').uncheck();
  const ids = await shownIds(page);
  expect(ids.some((id) => id.startsWith('routine:'))).toBe(false);
  expect(ids).toContain('corrective:generate_thomson_ods');
});

test('view switching shows jobs, artifacts and publication', async ({ page }) => {
  await openPipelines(page);
  for (const [view, kind] of [['dag', 'job'], ['artifacts', 'artifact'], ['publication', 'stage']]) {
    await page.locator(`input[name="vg-view"][value="${view}"]`).check();
    const kinds = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.nodes().map((n) => n.data('kind')));
    expect(kinds).toContain(kind);
  }
});

test('selecting a rule shows its artifacts and a pipeline-1 reference as a non-execution edge', async ({ page }) => {
  await openPipelines(page, '#focus=corrective:generate_core_profiles_ods');
  const details = page.locator('.vg-details');
  await expect(details).toContainText('Scientific references (consulted, not scheduled)');
  await expect(details).toContainText('generate_efit_ods');
  await expect(details).toContainText('Outputs');
  const kinds = await page.evaluate(() => document.querySelector('.vg-root').vaftGraph.cy.edges()
    .filter((e) => e.source().id() === 'routine:generate_efit_ods').map((e) => e.data('kind')));
  expect(kinds.length).toBeGreaterThan(0);
  expect(new Set(kinds)).toEqual(new Set(['scientific_reference']));
  await expect(details.locator('a', { hasText: 'Snakefile' })).toHaveAttribute('href', /blob\/[0-9a-f]{40}\/workflow\/.+Snakefile#L\d+/);
});

test('selecting a stage shows what it owns and where it is published, offline', async ({ page }) => {
  const remote = [];
  page.on('request', (request) => { if (!request.url().startsWith('http://127.0.0.1') && !request.url().startsWith('http://localhost')) remote.push(request.url()); });
  await openPipelines(page, '#view=publication&focus=stage:eddy');
  const details = page.locator('.vg-details');
  await expect(details).toContainText('Owns (publishes only these IDS)');
  await expect(details).toContainText('pf_passive');
  await expect(details).toContainText('replicate_eddy_to_hsds');
  expect(remote.filter((url) => /hsds|:5101/.test(url))).toEqual([]);
});

test('in-site navigation between the explorers leaks no handlers or dividers', async ({ page }) => {
  await page.goto('workflows/start-here/');
  for (const label of ['Dependency explorer', 'Pipeline lineage explorer', 'Dependency explorer']) {
    await page.locator('.book-summary a', { hasText: label }).first().click();
    await expect(page.locator('.vg-root[data-vg-mounted]')).toHaveCount(1);
    await expect(count(page)).toContainText(/\d+ nodes/);
  }
  const state = await page.evaluate(() => ({
    mouseup: (jQuery._data(document.body, 'events') || {}).mouseup.length,
    dividers: document.querySelectorAll('.divider-content-summary').length,
  }));
  expect(state).toEqual({ mouseup: 1, dividers: 1 });
});
