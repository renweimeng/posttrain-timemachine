import { useCallback, useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { ChevronLeft, ChevronRight, LoaderCircle, Play } from 'lucide-react';
import { api } from './api';
import type { Completion } from './types';

type SignalRow={index:number;token_id:number;piece:string;loss_mask:number;log_ratio:number|null;ratio:number|null;sampled_kl:number|null;surrogate:number|null};
type Quantity='log_ratio'|'sampled_kl'|'surrogate';
type TokenResponse={clipping:{epsilon_low:number;epsilon_high:number;source:string;objective_id:string|null};selection:string;observations:{token_blob_id:string;optimizer_step:number;token_count:number}[];
  selected:{token_blob_id:string;optimizer_step:number;source:string;tokenizer_hash:string;current_policy_version:string|null;old_policy_version:string|null;behavior_policy_version:string|null;reference_policy_version:string|null};
  signals:{rows:SignalRow[];sequence_advantage:number|null;missing:Record<string,string[]>;limits:string[]}};
type Objective={objective_id:string;loss_type:string;optimizer_step:number;declared_complete:boolean;global_effective_tokens:number};

function VirtualTokenCanvas({rows,quantity,selected,onSelect}:{rows:SignalRow[];quantity:Quantity;selected:number;onSelect:(i:number)=>void}) {
  const host=useRef<HTMLDivElement>(null),canvas=useRef<HTMLCanvasElement>(null),columns=useRef(1),frame=useRef(0);
  const [height,setHeight]=useState(300),[hover,setHover]=useState<number|null>(null);
  const draw=useCallback(()=>{
    if(!host.current||!canvas.current)return;
    const width=host.current.clientWidth,viewport=300,dpr=window.devicePixelRatio||1;
    columns.current=Math.max(1,Math.floor(width/86));
    const total=Math.max(viewport,Math.ceil(rows.length/columns.current)*34);
    setHeight(h=>h===total?h:total);
    canvas.current.width=width*dpr;canvas.current.height=viewport*dpr;
    const ctx=canvas.current.getContext('2d');if(!ctx)return;ctx.scale(dpr,dpr);ctx.clearRect(0,0,width,viewport);
    ctx.font='11px ui-monospace,monospace';
    const first=Math.floor(host.current.scrollTop/34)*columns.current;
    const last=Math.min(rows.length,first+(Math.ceil(viewport/34)+1)*columns.current);
    for(let i=first;i<last;i++){
      const row=rows[i],v=row[quantity],x=(i%columns.current)*86+3,y=Math.floor(i/columns.current)*34-host.current.scrollTop+3;
      ctx.fillStyle=!row.loss_mask?'#1a2335':v===null?'#243147':v>=0?`rgba(76,232,214,${Math.min(.75,.12+Math.abs(v)*.35)})`:`rgba(161,140,255,${Math.min(.75,.12+Math.abs(v)*.35)})`;
      ctx.fillRect(x,y,80,28);ctx.strokeStyle=i===selected?'#eaf4ff':'#34445e';ctx.lineWidth=i===selected?2:1;
      ctx.setLineDash(row.loss_mask?[]:[3,3]);ctx.strokeRect(x,y,80,28);ctx.setLineDash([]);
      ctx.save();ctx.beginPath();ctx.rect(x+4,y,72,28);ctx.clip();ctx.fillStyle='#eaf4ff';
      ctx.fillText(row.piece.replace(/\n/g,'↵').replace(/ /g,'·'),x+7,y+18);ctx.restore();
    }
  },[rows,quantity,selected]);
  useEffect(()=>{draw();const observer=new ResizeObserver(draw);if(host.current)observer.observe(host.current);return()=>{observer.disconnect();cancelAnimationFrame(frame.current);};},[draw]);
  const locate=(x:number,y:number)=>{if(!host.current||!canvas.current)return 0;const rect=canvas.current.getBoundingClientRect();return Math.min(rows.length-1,Math.max(0,Math.floor((y-rect.top+host.current.scrollTop)/34)*columns.current+Math.floor((x-rect.left)/86)));};
  const inspect=rows[hover??selected];
  return <><div className="token-canvas-scroll" tabIndex={0} role="region" aria-label="Scrollable captured tokens" ref={host} onScroll={()=>{cancelAnimationFrame(frame.current);frame.current=requestAnimationFrame(draw);}}>
    <div style={{height}}><canvas ref={canvas} role="img" aria-label={`${quantity} for ${rows.length} captured tokens. Masked tokens use dashed gray blocks. Use token index controls or the accessible table for exact values.`}
      style={{width:'100%',height:300,position:'sticky',top:0}} onClick={e=>onSelect(locate(e.clientX,e.clientY))}
      onMouseMove={e=>setHover(locate(e.clientX,e.clientY))} onMouseLeave={()=>setHover(null)}/></div></div>
    {inspect&&<div className="token-detail" aria-live="polite"><strong>Token {inspect.index} · ID {inspect.token_id}</strong><code>{inspect.piece}</code><span>mask={inspect.loss_mask} · {quantity}={inspect[quantity]===null?'unavailable':inspect[quantity]?.toFixed(6)}</span></div>}</>;
}

export function TokenLens({runId,completions}:{runId:string;completions:Completion[]}) {
  const [params,setParams]=useSearchParams();
  const completion=completions.find(c=>c.completion_id===params.get('completion'))??completions[0];
  const [quantity,setQuantity]=useState<Quantity>('log_ratio');
  const [page,setPage]=useState(0),[objectiveId,setObjectiveId]=useState(''),[loss,setLoss]=useState<Record<string,unknown>|null>(null);
  const [error,setError]=useState(''),[busy,setBusy]=useState(false);
  const blob=params.get('token_blob');
  const tokens=useQuery({queryKey:['tokens',runId,completion?.completion_id,blob],enabled:!!completion,
    queryFn:()=>api<TokenResponse>(`/completions/${encodeURIComponent(completion.completion_id)}/tokens?run_id=${encodeURIComponent(runId)}${blob?'&token_blob_id='+encodeURIComponent(blob):''}`)});
  const objectives=useQuery({queryKey:['objectives',runId],queryFn:()=>api<{items:Objective[]}>(`/runs/${encodeURIComponent(runId)}/objectives`)});
  useEffect(()=>{setPage(0);setObjectiveId('');setLoss(null);setError('');},[runId,completion?.completion_id,blob]);
  const selected=Math.min(Math.max(Number(params.get('token')||0)||0,0),Math.max((tokens.data?.signals.rows.length??1)-1,0));
  function update(key:string,value:string){const q=new URLSearchParams(params);q.set(key,value);if(key==='completion'){q.delete('token_blob');q.delete('token');}setParams(q,{replace:true});}
  async function replay(){setBusy(true);setError('');try{setLoss(await api<Record<string,unknown>>(`/objectives/${encodeURIComponent(objectiveId||objectives.data?.items[0]?.objective_id||'')}/replay`,{run_id:runId}));}catch(e){setError(String(e));}finally{setBusy(false);}}
  const rows=tokens.data?.signals.rows??[];
  return <section className="panel token-lens"><div className="panel-head"><div><h2>Token Signal Lens</h2><p>Captured tokenizer pieces and masks · recomputed sampled signals · no gradient inference</p></div><span className="badge cyan">L2 · captured tensors</span></div>
    <div className="token-toolbar"><label>Completion<select aria-label="Token completion" value={completion?.completion_id??''} onChange={e=>update('completion',e.target.value)}>{completions.map(c=><option key={c.completion_id} value={c.completion_id}>C{c.sequence_index+1} · {c.completion_id.slice(-14)}</option>)}</select></label>
    <label>Observation<select aria-label="Token observation" value={tokens.data?.selected.token_blob_id??''} onChange={e=>update('token_blob',e.target.value)}>{tokens.data?.observations.map(t=><option key={t.token_blob_id} value={t.token_blob_id}>Optimizer {t.optimizer_step} · {t.token_count} tokens</option>)}</select></label>
    <label>Signal<select aria-label="Token signal" value={quantity} onChange={e=>setQuantity(e.target.value as Quantity)}><option value="log_ratio">log ratio: current − old</option><option value="sampled_kl">Sampled KL estimator</option><option value="surrogate">Clipped surrogate contribution</option></select></label></div>
    {tokens.isLoading?<div className="loading">Loading captured Token arrays…</div>:tokens.error?<div className="error" role="alert">{String(tokens.error)}</div>:tokens.data&&<>
      <p className="token-provenance">{tokens.data.selection} · sequence A={tokens.data.signals.sequence_advantage?.toFixed(6)??'unavailable'} · {tokens.data.selected.source}</p>
      {tokens.data.signals.missing[quantity==='log_ratio'?'ratio':quantity]&&<p className="boundary">Insufficient capture: {tokens.data.signals.missing[quantity==='log_ratio'?'ratio':quantity]?.join(', ')}</p>}
      <p className="token-provenance">Surrogate clipping: [{(1-tokens.data.clipping.epsilon_low).toFixed(4)}, {(1+tokens.data.clipping.epsilon_high).toFixed(4)}] · {tokens.data.clipping.source}{tokens.data.clipping.objective_id?` · ${tokens.data.clipping.objective_id}`:''}</p>
      <VirtualTokenCanvas rows={rows} quantity={quantity} selected={selected} onSelect={i=>update('token',String(i))}/>
      <div className="token-index"><button className="secondary" aria-label="Previous token" disabled={selected===0} onClick={()=>update('token',String(selected-1))}><ChevronLeft size={14}/></button><label>Token index<input aria-label="Token index" type="range" min="0" max={Math.max(rows.length-1,0)} value={selected} onChange={e=>update('token',e.target.value)}/></label><span>{selected+1}/{rows.length}</span><button className="secondary" aria-label="Next token" disabled={selected>=rows.length-1} onClick={()=>update('token',String(selected+1))}><ChevronRight size={14}/></button></div>
      <div className="token-legend"><span className="cyan">Positive / higher</span><span className="violet">Negative / lower</span><span>Dashed gray = loss_mask 0 · unavailable ≠ zero</span></div>
      <details className="raw"><summary>Accessible Token table and policy provenance</summary><div className="policy-grid"><span>current: {tokens.data.selected.current_policy_version??'missing'}</span><span>old: {tokens.data.selected.old_policy_version??'missing'}</span><span>behavior: {tokens.data.selected.behavior_policy_version??'missing'}</span><span>reference: {tokens.data.selected.reference_policy_version??'missing'}</span></div><p className="token-provenance">Tokenizer hash: {tokens.data.selected.tokenizer_hash}. Different IDs do not imply a known policy-step distance.</p>
      <div className="table-scroll"><table><caption>Exact captured IDs/pieces; {page*100+1}–{Math.min((page+1)*100,rows.length)} of {rows.length}</caption><thead><tr><th>Index / ID</th><th>Piece</th><th>Mask</th><th>Ratio</th><th>Sampled KL</th><th>Surrogate</th></tr></thead><tbody>{rows.slice(page*100,(page+1)*100).map(r=><tr key={r.index}><td><button className="text-btn" onClick={()=>update('token',String(r.index))}>{r.index} / {r.token_id}</button></td><td><code>{r.piece}</code></td><td>{r.loss_mask}</td><td>{r.ratio?.toFixed(6)??'unavailable'}</td><td>{r.sampled_kl?.toFixed(6)??'unavailable'}</td><td>{r.surrogate?.toFixed(6)??'unavailable'}</td></tr>)}</tbody></table></div><div className="pagination"><button disabled={!page} onClick={()=>setPage(page-1)}>Previous 100</button><button disabled={(page+1)*100>=rows.length} onClick={()=>setPage(page+1)}>Next 100</button></div></details></>}
    <div className="objective-controls"><div><h3>Objective replay</h3><p>Full optimizer batch and explicit denominator required. This computes a scalar loss, without updating weights.</p></div><label>Captured batch<select aria-label="Objective batch" value={objectiveId||objectives.data?.items[0]?.objective_id||''} onChange={e=>{setObjectiveId(e.target.value);setLoss(null);}} disabled={busy}>{objectives.data?.items.map(o=><option key={o.objective_id} value={o.objective_id}>{o.loss_type} · optimizer {o.optimizer_step} · {o.global_effective_tokens} effective tokens</option>)}</select></label><button className="primary" disabled={busy||!objectives.data?.items.length} onClick={()=>void replay()}>{busy?<LoaderCircle size={15} className="spin"/>:<Play size={15}/>} Replay objective</button></div>
    {error&&<p className="error" role="alert">{error}</p>}{loss&&<div className="objective-result" role="status"><span className="badge green">RECOMPUTED</span><strong>Loss: {Number(loss.loss).toFixed(8)}</strong><span>{String(loss.loss_type)} · {String(loss.normalization_scope)}</span><details><summary>Recipe, denominator and per-completion evidence</summary><pre>{JSON.stringify(loss,null,2)}</pre></details></div>}
  </section>;
}

export function TrajectoryView({runId,completions}:{runId:string;completions:Completion[]}) {
  const [choice,setChoice]=useState('');
  const completionId=completions.find(c=>c.completion_id===choice)?.completion_id??completions[0]?.completion_id??'';
  const trace=useQuery({queryKey:['trajectory',runId,completionId],enabled:!!completionId,
    queryFn:()=>api<{items:{trajectory_id:string;source:string;turns:{turn_id:string;role:string;content:string;tool_name?:string;policy_version?:string}[]}[]}>(`/completions/${encodeURIComponent(completionId)}/trajectory?run_id=${encodeURIComponent(runId)}`)});
  return <section className="panel trajectory"><div className="panel-head"><h2>Recorded multi-turn trajectory</h2><span className="badge">OBSERVED · no tool execution</span></div><label>Trajectory completion<select aria-label="Trajectory completion" value={completionId} onChange={e=>setChoice(e.target.value)}>{completions.map(c=><option key={c.completion_id} value={c.completion_id}>C{c.sequence_index+1} · {c.completion_id}</option>)}</select></label>{trace.error&&<p role="alert" className="error">{String(trace.error)}</p>}{trace.isLoading?<p>Loading recorded trajectory…</p>:!trace.data?.items.length?<p>No multi-turn trace was captured for this completion.</p>:trace.data.items.map(t=><div key={t.trajectory_id}><p>{t.source}</p>{t.turns.map(turn=><article key={turn.turn_id}><span className="badge">{turn.role}{turn.tool_name?' / '+turn.tool_name:''}</span><small>Policy: {turn.policy_version??'not recorded'}</small><pre>{turn.content}</pre></article>)}</div>)}</section>;
}
