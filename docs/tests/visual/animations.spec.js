const {test,expect}=require('@playwright/test');
// A composed site may contain an older track that has not shipped these components.
test.beforeEach(async ({request}, testInfo) => {
  if (testInfo.title.startsWith('adaptive')) {
    const response=await request.get('');
    test.skip(!(await response.text()).includes('assets/gitbook/adaptive.js'), 'This track has not shipped the adaptive component');
  } else {
    const response=await request.get('reference/animation/');
    test.skip(response.status()===404, 'This track has no animation catalog/gallery');
  }
});

test('gallery is lazy, filters without transfer, and seeks authoritative SVG states',async({page})=>{
  const requests=[];page.on('request',r=>{if(/animations\/.*(?:\.json|player\.js)/.test(r.url()))requests.push(r.url());});
  await page.goto('reference/animation/');
  await expect(page.locator('[data-animation-card]')).toHaveCount(2);expect(requests).toEqual([]);
  await page.locator('[data-animation-search]').fill('nonsense');await expect(page.locator('[data-animation-empty]')).toBeVisible();expect(requests).toEqual([]);
  await page.locator('[data-animation-search]').fill('');await page.locator('[data-animation-driver]').selectOption('phase');
  const root=page.locator('[data-animation="magnetic_island--phase_scan"]');await root.locator('[data-activate]').click();await expect(root.locator('.va-player')).toBeVisible();
  await expect(root.locator('[data-role="rational"]')).toHaveAttribute('stroke','#1c67d2');
  await expect(root).toHaveAttribute('data-state','0');await root.getByRole('button',{name:'Next',exact:true}).click();await expect(root).toHaveAttribute('data-state','1');
  const response=await page.request.get(await root.getAttribute('data-artifact'));const data=await response.json();
  const point=data.states[1].find(r=>r.role==='o_point');if(point){const marker=root.locator(`[data-element="${point.id}"]`);expect(Number(await marker.getAttribute('cx'))).toBeCloseTo(point.at[0],12);}
  await root.getByRole('button',{name:'Reset',exact:true}).click();await expect(root).toHaveAttribute('data-state','0');
  await root.getByRole('button',{name:'Enlarge',exact:true}).click();await expect(root.locator('.va-player')).toHaveClass(/va-enlarged/);await page.keyboard.press('Escape');await expect(root.locator('.va-player')).not.toHaveClass(/va-enlarged/);
});

test('both routes share state URLs; playback is single and cleaned on navigation',async({page})=>{
  test.setTimeout(60000);
  await page.goto('reference/animation/');
  const first=page.locator('[data-animation]').first();
  const url=await first.getAttribute('data-artifact'),id=await first.getAttribute('data-animation');
  // Inspect the generated consumer without laying out hundreds of unrelated SVGs.
  const response=await page.request.get('reference/diagram/');expect(response.ok()).toBe(true);
  const shared=await page.evaluate(({html,id})=>{
    const document=new DOMParser().parseFromString(html,'text/html');
    const root=[...document.querySelectorAll('[data-animation]')].find(root=>root.dataset.animation===id);
    return root?{artifact:root.dataset.artifact,html:root.outerHTML}:null;
  },{html:await response.text(),id});expect(shared?.artifact).toBe(url);
  await first.locator('[data-activate]').click();await expect(first.locator('.va-player')).toBeVisible();
  const second=page.locator('[data-animation]').nth(1);await second.locator('[data-activate]').click();await expect(second.locator('.va-player')).toBeVisible();
  await first.getByRole('button',{name:'Play',exact:true}).click();await second.getByRole('button',{name:'Play',exact:true}).click();await expect(first).toHaveAttribute('data-playing','false');
  // Mount the real Diagram Gallery component in isolation, using the same loader.
  await page.evaluate(html=>{
    const document=new DOMParser().parseFromString(html,'text/html');
    const root=window.document.importNode(document.body.firstElementChild,true);
    root.dataset.testDiagramPlayer='';window.document.querySelector('[data-animation-gallery]').append(root);
  },shared.html);
  const diagram=page.locator('[data-test-diagram-player]');
  await diagram.locator('[data-activate]').click();await expect(diagram.locator('.va-player')).toBeVisible();
  await page.evaluate(()=>window.VAFTAnimationCleanup());await expect(page.locator('.va-player')).toHaveCount(0);await expect(diagram.locator('[data-activate]')).toBeEnabled();await expect(first.locator('[data-activate]')).toBeEnabled();
  await first.locator('[data-activate]').click();await expect(first.locator('.va-player')).toBeVisible();
  async function navigate(route){
    if(await page.evaluate(()=>!document.querySelector('.book').classList.contains('with-summary')))await page.locator('.book-header .fa-align-justify').locator('..').click();
    await page.locator(`.book-summary a[href$="/${route}"]`).first().click();
    await expect(page).toHaveURL(new RegExp(route));
  }
  await navigate('workflows/start-here/');await expect(page.locator('.va-player')).toHaveCount(0);
  await navigate('reference/animation/');
  await expect(page.locator('[data-animation-gallery]')).toBeVisible();await expect(page.locator('.va-player')).toHaveCount(0);
  await page.locator('[data-animation]').first().locator('[data-activate]').click();await expect(page.locator('.va-player')).toHaveCount(1);
});

test('reduced motion and failed data keep the canonical poster usable',async({page})=>{
  await page.emulateMedia({reducedMotion:'reduce'});await page.goto('reference/animation/');
  const root=page.locator('[data-animation]').first();await expect(root.locator('img')).toBeVisible();await expect(root).not.toHaveAttribute('data-playing','true');
  await page.route('**/assets/animations/*.json',route=>route.fulfill({status:503,body:'unavailable'}));await root.locator('[data-activate]').click();await expect(root.locator('[data-animation-status]')).toContainText('unavailable');await expect(root.locator('img')).toBeVisible();await expect(root.locator('[data-activate]')).toBeEnabled();
});

test('charged particles retain scientific positions, species colours and gated downloads',async({page})=>{
  await page.goto('reference/animation/');
  const root=page.locator('[data-animation="exb_drift--particle_motion"]');
  await root.locator('details').evaluate(el=>el.open=true);
  await expect(root.locator('[data-state-download]')).toBeHidden();
  await root.locator('[data-activate]').click();await expect(root.locator('.va-player')).toBeVisible();
  await expect(root.locator('[data-state-download]')).toBeVisible();
  const data=await(await page.request.get(await root.getAttribute('data-artifact'))).json();
  for(const index of [0,180,360]){
    await root.locator('input[type=range]').fill(String(index));
    for(const [name,sign,colour] of [['ion','+','#d63027'],['electron','−','#1c67d2']]){
      const row=[...data.static,...data.states[index]].find(r=>r.role===name+'_particle');
      const marker=root.locator(`[data-element="${row.id}"]`);
      await expect(marker).toHaveAttribute('transform',`translate(${row.at[0]},${row.at[1]})`);
      await expect(marker.locator('text')).toHaveText(sign);
      await expect(marker.locator('circle')).toHaveAttribute('fill',colour);
      await expect(root.locator(`[data-role="${name}_orbit"]`)).toHaveAttribute('stroke',colour);
    }
  }
  await page.evaluate(()=>window.VAFTAnimationCleanup());
  await expect(root.locator('[data-state-download]')).toBeHidden();
});

// One test per (width, route) and a separate menu test: WebKit needs ~28 s for the
// whole route loop, so a shared 30 s budget timed out on whichever step came last.
const ADAPTIVE_ROUTES=['','workflows/experimental-interpretation/','reference/animation/','reference/diagram/','reference/formula/','reference/api/diagram/','reference/process/','reference/dependency-graph/','reference/pipeline-graph/','workflows/automated-pipelines/'];
for(const width of [320,390,768,1024,1440]){
  for(const route of ADAPTIVE_ROUTES)test(`adaptive layout at ${width}px: /${route}`,async({page})=>{
    test.skip(route.startsWith('reference/')&&(await page.request.get(route)).status()===404,'This track has no such reference page');
    await page.setViewportSize({width,height:900});
    await page.goto(route);expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  });
  if(width<=1240)test(`adaptive menu at ${width}px`,async({page})=>{
    await page.setViewportSize({width,height:900});
    await page.goto('workflows/experimental-interpretation/');const menu=page.locator('.book-header .fa-align-justify').locator('..');await menu.click();await expect(page.locator('.book-summary')).toBeVisible();await page.keyboard.press('Tab');expect(await page.evaluate(()=>document.querySelector('.book-summary').contains(document.activeElement))).toBe(true);await page.keyboard.press('Escape');await expect(menu).toBeFocused();await expect(page.locator('.body-inner')).not.toHaveAttribute('inert','');
  });
}
