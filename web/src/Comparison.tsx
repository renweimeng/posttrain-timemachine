import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { GitCompareArrows, LoaderCircle } from 'lucide-react';
import { api } from './api';
import type { Completion, Group, GroupDetail, Run } from './types';

export function CompletionComparison({selected}:{selected:Completion[]}) {
  const parts=useMemo(()=>{
    if(selected.length!==2)return null;
    const segmenter=new Intl.Segmenter(undefined,{granularity:'grapheme'});
    const a=Array.from(segmenter.segment(selected[0].text??''),x=>x.segment),b=Array.from(segmenter.segment(selected[1].text??''),x=>x.segment);
    let prefix=0,suffix=0;
    while(prefix<Math.min(a.length,b.length)&&a[prefix]===b[prefix])prefix++;
    while(suffix<Math.min(a.length,b.length)-prefix&&a[a.length-1-suffix]===b[b.length-1-suffix])suffix++;
    return [a,b].map(v=>({prefix:v.slice(0,prefix).join(''),changed:v.slice(prefix,v.length-suffix).join(''),suffix:suffix?v.slice(-suffix).join(''):''}));
  },[selected]);
  if(selected.length!==2||!parts)return <div className="empty compact">Select two completion cards to compare their captured text and reward signals.</div>;
  const keys=[...new Set(selected.flatMap(c=>Object.keys(c.reward_components)))].sort();
  return <><div className="compare-grid">{selected.map((c,i)=><div key={c.completion_id}><strong>{c.completion_id}</strong><pre>{c.text===null?'Text not captured':<>{parts[i].prefix}<mark className={i?'text-added':'text-removed'}>{parts[i].changed}</mark>{parts[i].suffix}</>}</pre><p>Reward {c.reward_total?.toFixed(4)??'unavailable'} · Sequence A {c.advantage?.toFixed(4)??'unavailable'}</p></div>)}</div><div className="table-scroll"><table><caption>Contiguous changed span uses Unicode graphemes; captured components remain independently inspectable.</caption><thead><tr><th>Reward component</th><th>Left</th><th>Right</th><th>Right − left</th></tr></thead><tbody>{keys.map(k=>{const a=selected[0].reward_components[k],b=selected[1].reward_components[k];return <tr key={k}><td>{k}</td><td>{a?.toFixed(4)??'unavailable'}</td><td>{b?.toFixed(4)??'unavailable'}</td><td>{a==null||b==null?'unavailable':(b-a).toFixed(4)}</td></tr>;})}</tbody></table></div></>;
}

type CrossResult={left:GroupDetail;right:GroupDetail;prompt_match:boolean;recipe_match:boolean;advantage_comparable:boolean;comparison_level:string;limits:string[]};
export function CrossRunComparison({runId,group,runs}:{runId:string;group:GroupDetail;runs:Run[]}) {
  const [other,setOther]=useState(''),[otherGroup,setOtherGroup]=useState(''),[all,setAll]=useState(false),[result,setResult]=useState<CrossResult|null>(null),[error,setError]=useState(''),[busy,setBusy]=useState(false);
  const options=useQuery({queryKey:['compare-groups',other,all,group.prompt_id],enabled:!!other,
    queryFn:()=>api<{items:Group[]}>(`/runs/${encodeURIComponent(other)}/groups?limit=200&search=${all?'':encodeURIComponent(group.prompt_id)}`)});
  async function compare(){setBusy(true);setError('');try{setResult(await api<CrossResult>('/comparisons/groups',{
    left_run_id:runId,left_group_id:group.group_id,right_run_id:other,right_group_id:otherGroup||options.data?.items[0]?.group_id}));}catch(e){setError(String(e));}finally{setBusy(false);}}
  return <section className="panel cross-run"><details><summary><GitCompareArrows size={17}/> Compare another run / recipe</summary><div className="comparison-controls"><label>Reference run<select aria-label="Comparison run" value={other} onChange={e=>{setOther(e.target.value);setOtherGroup('');setResult(null);}}><option value="">Choose another captured run…</option>{runs.filter(r=>r.run_id!==runId).map(r=><option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}</select></label><label>Reference group<select aria-label="Comparison group" value={otherGroup||options.data?.items[0]?.group_id||''} onChange={e=>setOtherGroup(e.target.value)}>{options.data?.items.map(g=><option key={g.group_id} value={g.group_id}>{g.group_id}</option>)}</select></label><label className="check-label"><input type="checkbox" checked={all} onChange={e=>{setAll(e.target.checked);setOtherGroup('');}}/> Show other prompt IDs</label><button className="primary" disabled={busy||!other||!options.data?.items.length} onClick={()=>void compare()}>{busy?<LoaderCircle className="spin" size={15}/>:<GitCompareArrows size={15}/>} Compare captures</button></div>
    {other&&options.data&&!options.data.items.length&&<p className="boundary">No matching prompt ID in this run. Choose a different run or explicitly enable other prompt IDs.</p>}{(error||options.error)&&<p role="alert" className="error">{String(error||options.error)}</p>}
    {result&&<div className="cross-result"><span className={'badge '+(result.advantage_comparable?'green':'amber')}>{result.recipe_match?result.comparison_level:'Recipe mismatch · descriptive only'}</span>{!result.prompt_match&&<p className="amber">Different prompt IDs — no matched-task comparison.</p>}<div className="compare-grid">{[result.left,result.right].map((g,i)=><div key={i}><h3>{g.run_id}</h3><p>{g.recipe_id}</p><p>{g.completions.length}/{g.expected_group_size} recorded · {g.prompt}</p>{g.completions.map(c=><p key={c.completion_id}>C{c.sequence_index+1}: reward {c.reward_total?.toFixed(4)??'unavailable'}{result.advantage_comparable?` · A ${c.advantage?.toFixed(4)}`:''}</p>)}</div>)}</div>{result.limits.map(x=><p key={x}>{x}</p>)}</div>}</details></section>;
}
