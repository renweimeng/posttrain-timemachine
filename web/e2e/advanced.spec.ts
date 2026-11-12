import {test,expect} from '@playwright/test';
import {writeFile} from 'node:fs/promises';
import AxeBuilder from '@axe-core/playwright';

test('all four Arena challenges fail old rewards and pass reviewed patches',async({page})=>{
  await page.goto('/arena');
  const names=['Format Over Truth','Zero-Signal Group','Verbose Wins','Checker Blind Spot'];
  for(const name of names){
    const card=page.locator('.challenge').filter({has:page.getByRole('heading',{name,exact:true})});
    await card.getByRole('button',{name:'Open challenge',exact:true}).click();
    await page.getByRole('button',{name:'Run challenge tests',exact:true}).click();
    await expect(page.getByText('REWARD CONTRACT FAILED',{exact:true})).toBeVisible();
    await page.getByRole('button',{name:'Suggested patch',exact:true}).click();
    await page.getByRole('button',{name:'Run challenge tests',exact:true}).click();
    await expect(page.getByText('KNOWN & GUARD CASES PASS',{exact:true})).toBeVisible();
    await expect(page.getByRole('button',{name:'Save reviewed Arena contract',exact:true})).toBeDisabled();
    await page.getByLabel('I reviewed the fixed cases',{exact:false}).check();
    await page.getByRole('button',{name:'Save reviewed Arena contract',exact:true}).click();
    await expect(page.getByRole('button',{name:'Reviewed contract saved',exact:true})).toBeVisible();
  }
});

test('L2 virtual canvas, exact table, deep links and full objective replay work',async({page})=>{
  await page.goto('/inspector?run=synthetic-token-fixture&group=group-6400-0');
  await expect(page.getByRole('heading',{name:'Token Signal Lens',exact:true})).toBeVisible();
  await expect(page.getByRole('img',{name:/log_ratio for 5000 captured tokens/})).toBeVisible();
  await page.getByLabel('Token signal',{exact:true}).selectOption('sampled_kl');
  await page.getByLabel('Token index',{exact:true}).fill('1');
  await expect(page.locator('.token-detail')).toContainText('🤖');
  await page.reload();
  await expect(page.getByLabel('Token index',{exact:true})).toHaveValue('1');
  await page.getByRole('button',{name:'Replay objective',exact:true}).click();
  await expect(page.locator('.objective-result')).toContainText('Loss:');
  await page.getByText('Accessible Token table and policy provenance',{exact:true}).click();
  await expect(page.getByRole('table').filter({has:page.getByRole('columnheader',{name:'Sampled KL',exact:true})})).toBeVisible();
  await page.getByRole('button',{name:'Next 100',exact:true}).click();
  await expect(page.getByText('Exact captured IDs/pieces; 101–200 of 5000',{exact:true})).toBeVisible();
  const fps=await page.evaluate(async()=>{
    const host=document.querySelector('.token-canvas-scroll') as HTMLElement;
    const times:number[]=[];let previous=performance.now();
    for(let i=0;i<30;i++){host.scrollTop=i*75;await new Promise<void>(resolve=>requestAnimationFrame(()=>resolve()));const now=performance.now();times.push(now-previous);previous=now;}
    return 1000/(times.reduce((a,b)=>a+b,0)/times.length);
  });
  await writeFile('../output/token-canvas-performance.json',JSON.stringify({fps,frames:30,tokens:5000,browser:'Chrome',scope:'local headless scrolling; not a GPU or cross-device claim'},null,2));
  expect(fps).toBeGreaterThan(30); // Record actual frame rate; do not equate this Chrome run with every device.
  await page.screenshot({path:'../output/ui-token-lens.png',fullPage:true});
});

test('primary screens have no serious or critical automated accessibility findings',async({page})=>{
  for(const route of ['/timeline','/runs','/arena','/contracts','/reports','/analyses','/lab?run=demo-format-truth&group=group-6400-0','/updates?run=tiny-trl-group-l2-dapo-adamw','/inspector?run=synthetic-token-fixture&group=group-6400-0','/inspector?run=demo-format-truth&group=group-6400-0']){
    await page.goto(route);
    await expect(page.getByRole('heading',{level:1})).toBeVisible();
    const results=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    const serious=results.violations.filter(v=>v.impact==='serious'||v.impact==='critical');
    expect(serious,JSON.stringify(serious.map(v=>({id:v.id,description:v.description,nodes:v.nodes.map(n=>n.target)})))).toEqual([]);
  }
});


test('actual captured optimizer step executes and matches native weights in the browser',async({page,request})=>{
  const capability=await (await request.get('/api/v1/system/capabilities')).json();
  test.skip(!capability.update_replay_available,'Pinned optional reference environment unavailable');
  await page.goto('/updates?run=tiny-trl-group-l2-dapo-adamw');
  await expect(page.getByRole('heading',{name:'Replay captured step',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Replay captured optimizer step',exact:true}).click();
  await expect(page.locator('.update-result')).toContainText('REPLAYED UPDATE');
  await expect(page.locator('.update-result')).toContainText('Match');
  await page.screenshot({path:'../output/ui-update-replay.png',fullPage:true});
  await page.reload();
  await expect(page.getByRole('heading',{name:'Replay captured step',exact:true})).toBeVisible();
});

test('contract file import requires fresh review and can export portable YAML',async({page})=>{
  const contract={id:'browser-imported-contract',fixtures:[
    {completion_id:'import-good',group_id:'group',sequence_index:0,text:'4',reward_total:1,reward_status:'valid',reward_components:{},inputs:{correctness_score:1,format_score:0}},
    {completion_id:'import-bad',group_id:'group',sequence_index:1,text:'5',reward_total:0,reward_status:'valid',reward_components:{},inputs:{correctness_score:0,format_score:0}}],assertions:[{type:'reward_order',higher:'import-good',lower:'import-bad'}],origin:{run_id:'demo-format-truth',reviewed:true}};
  await page.goto('/contracts?run=demo-format-truth');
  await page.getByLabel('Contract file',{exact:true}).setInputFiles({name:'contract.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(contract))});
  await expect(page.getByRole('button',{name:'Save imported contract',exact:true})).toBeDisabled();
  await page.getByLabel('I reviewed the imported fixtures and expected assertions.',{exact:true}).check();
  await page.getByRole('button',{name:'Save imported contract',exact:true}).click();
  const row=page.getByRole('row').filter({has:page.getByText('browser-imported-contract',{exact:true})});
  await expect(row).toBeVisible();
  const downloaded=page.waitForEvent('download');await row.getByRole('button',{name:'YAML',exact:true}).click();
  expect((await downloaded).suggestedFilename()).toBe('browser-imported-contract.yaml');
});
