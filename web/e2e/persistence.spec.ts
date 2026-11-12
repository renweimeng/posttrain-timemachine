import {test,expect} from '@playwright/test';

test('refresh, deep links, drafts and report history remain usable',async({page})=>{
  await page.goto('/inspector?run=demo-format-truth&group=group-6400-0');
  await page.getByRole('button',{name:'Add to What-if',exact:true}).click();
  await page.getByLabel('Format weight',{exact:true}).fill('0.05');
  await page.getByRole('button',{name:'Recompute Deltas',exact:true}).click();
  await expect(page.getByText('succeeded',{exact:true}).first()).toBeVisible();
  const saved=page.url();expect(saved).toContain('analysis=ana_');
  await page.reload();
  await expect(page.getByText('succeeded',{exact:true}).first()).toBeVisible();
  await expect(page.getByLabel('Format weight',{exact:true})).toHaveValue('0.05');
  await page.getByRole('button',{name:'Reports',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Saved investigation reports'})).toBeVisible();
  await expect(page.getByRole('button',{name:'Report',exact:true}).first()).toBeVisible();
  await page.getByRole('button',{name:'Open',exact:true}).first().click();
  await expect(page).toHaveURL(/\/lab\?/);
  await expect(page.getByText('succeeded',{exact:true}).first()).toBeVisible();
  await page.goBack();
  await expect(page.getByRole('heading',{name:'Saved investigation reports'})).toBeVisible();
});

test('run browser can validate and import a trace without CLI',async({page})=>{
  await page.goto('/runs');
  await page.getByLabel('Trace file',{exact:true}).setInputFiles('../fixtures/trl/captured-events.jsonl');
  await page.getByRole('button',{name:'Import trace',exact:true}).click();
  await expect(page.getByText('Trace validated and imported.',{exact:true})).toBeVisible();
  await expect(page.getByLabel('Select run',{exact:true})).toHaveValue('tiny-trl-group');
});
