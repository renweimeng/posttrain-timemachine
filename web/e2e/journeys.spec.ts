import {test,expect} from '@playwright/test';

test('investigate → patch → reviewed contract → regression → report, fully offline',async({page})=>{
  const consoleErrors:string[]=[];page.on('pageerror',e=>consoleErrors.push(e.message));
  await page.route('**/*',route=>route.request().url().startsWith('http://127.0.0.1:8766')?route.continue():route.abort());
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'Training Timeline',exact:true})).toBeVisible();
  await expect(page.getByText('synthetic example',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Reveal Evidence',exact:true}).first().click();
  await expect(page.getByRole('heading',{name:'Rollout Inspector',exact:true})).toBeVisible();
  await expect(page.getByText('Token signals unavailable at L1.',{exact:false})).toBeVisible();
  await page.getByRole('button',{name:'Add to What-if',exact:true}).click();
  await page.getByLabel('Format weight',{exact:true}).fill('0.05');
  await page.getByRole('button',{name:'Recompute Deltas',exact:true}).click();
  await expect(page.getByText('succeeded',{exact:true})).toBeVisible();
  await expect(page.getByText('ADVANTAGE SIGN FLIPS',{exact:true})).toBeVisible();
  await page.getByLabel('Contract ID',{exact:true}).fill('e2e-format-order');
  // Explicitly choose the human-reviewed arithmetic pass/fail fixtures.
  const good=page.getByLabel('Higher reward fixture',{exact:true});
  await good.selectOption({label:'C1 · Verifier pass'});
  await page.getByLabel('Lower reward fixture',{exact:true}).selectOption({label:'C2 · Verifier fail'});
  await page.getByLabel('I reviewed these fixtures',{exact:false}).check();
  await page.getByRole('button',{name:'Save reviewed contract',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Regression Tests',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Use original reward',exact:true}).click();
  await page.getByRole('button',{name:'Run contracts',exact:true}).click();
  await expect(page.getByRole('row').filter({hasText:'e2e-format-order'}).getByText('FAIL',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Use patched reward',exact:true}).click();
  await page.getByRole('button',{name:'Run contracts',exact:true}).click();
  await expect(page.getByRole('row').filter({hasText:'e2e-format-order'}).getByText('PASS',{exact:true})).toBeVisible();
  const pytestDownload=page.waitForEvent('download');
  await page.getByRole('button',{name:'pytest',exact:true}).first().click();
  expect((await pytestDownload).suggestedFilename()).toMatch(/^test_.*\.py$/);
  await page.getByRole('button',{name:'Reports',exact:true}).click();
  const reportDownload=page.waitForEvent('download');
  await page.getByRole('button',{name:'Download reproducibility bundle',exact:true}).click();
  expect((await reportDownload).suggestedFilename()).toMatch(/\.zip$/);
  expect(consoleErrors).toEqual([]);
});

test('data diagnosis, filtering, raw metric evidence and keyboard navigation',async({page})=>{
  await page.goto('/timeline');
  await page.getByPlaceholder('Search prompt or group…').fill('not-a-captured-task');
  await expect(page.getByRole('heading',{name:'No groups match this filter'})).toBeVisible();
  await page.getByPlaceholder('Search prompt or group…').fill('');
  await page.getByText('Accessible data table',{exact:false}).click();
  await page.getByRole('button',{name:'6400',exact:true}).click();
  await expect(page.getByText('Generation step 6400 · explicit batch match')).toBeVisible();
  await page.getByRole('button',{name:'Integrations & data',exact:true}).click();
  await expect(page.getByText('No Token arrays captured',{exact:true})).toBeVisible();
  await page.keyboard.press('Control+k');
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByRole('dialog').getByRole('button',{name:'What-if Lab'}).click();
  await page.getByLabel('Format weight',{exact:true}).fill('0.05');
  await page.getByRole('button',{name:'Recompute Deltas',exact:true}).click();
  await expect(page.getByText('succeeded',{exact:true})).toBeVisible();
  await page.getByLabel('Format weight',{exact:true}).fill('0.10');
  await expect(page.getByText('Stale snapshot',{exact:false})).toBeVisible();
});

test('GPU-free arena, responsive workspace and explicit evidence boundaries',async({page})=>{
  await page.goto('/arena');
  await expect(page.getByRole('heading',{name:'Format Over Truth',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Inspect Zero-Signal Group',exact:true}).click();
  await expect(page.locator('.score-line > div:nth-child(2) > strong')).toHaveText(Array(4).fill('0.000'));
  await page.setViewportSize({width:390,height:844});
  await expect(page.getByRole('button',{name:'Add to What-if',exact:true})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
  await page.screenshot({path:'../output/ui-mobile.png',fullPage:true});
  await page.setViewportSize({width:1600,height:1000});
  await page.goto('/timeline?run=demo-format-truth');
  await expect(page.getByRole('button',{name:'Reveal Evidence',exact:true}).first()).toBeVisible();
  await page.screenshot({path:'../output/ui-preview.png'});
  await page.screenshot({path:'../output/ui-timeline.png',fullPage:true});
});

test('empty data and untrusted completion text have explicit safe states',async({page})=>{
  await page.route('**/api/v1/runs',route=>route.fulfill({json:{items:[]}}));
  await page.goto('/');
  await expect(page.getByRole('heading',{name:'No committed runs'})).toBeVisible();
  await page.unroute('**/api/v1/runs');
  await page.route('**/api/v1/groups/*',async route=>{
    const response=await route.fetch();const data=await response.json();
    data.completions[0].text='<img src=x onerror="window.ptmInjected=true">';
    data.completions[0].grade.value=0.5;
    await route.fulfill({response,json:data});
  });
  await page.goto('/inspector?run=demo-format-truth&group=group-6400-0');
  await expect(page.getByText('<img src=x onerror="window.ptmInjected=true">',{exact:true})).toBeVisible();
  await expect(page.getByText('Verifier score 0.5 · scale unspecified',{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>Object.hasOwn(window,'ptmInjected'))).toBeFalsy();
});

test('actual pinned TRL capture is inspectable end to end',async({page})=>{
  await page.goto('/timeline');
  await page.getByLabel('Select run',{exact:true}).selectOption('tiny-trl-group');
  await expect(page.getByText('real captured trace',{exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'No scalar metrics captured'})).toBeVisible();
  await page.getByRole('button',{name:/^Inspect grp_/}).first().click();
  await expect(page.getByRole('heading',{name:'Rollout Inspector',exact:true})).toBeVisible();
  await expect(page.locator('.completion-card')).toHaveCount(4);
  await page.getByText('Raw group JSON & provenance',{exact:true}).click();
  await expect(page.locator('.raw pre')).toContainText('trl-0.24.0-grpo-group-v1');
});
