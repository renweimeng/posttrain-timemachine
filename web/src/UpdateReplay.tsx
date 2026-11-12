import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams } from 'react-router-dom';
import { ArrowRight, GitBranch, LoaderCircle, Play } from 'lucide-react';
import { api } from './api';
import type { RewardSpec } from './types';

type Capture={capture_id:string;run_id:string;optimizer_step:number;optimizer:string;scope:string;
  model_type:string;environment:{device:string;torch:string;platform:string}};
type Result={analysis_id:string;evidence:string;loss:number;parameter_delta_l2:number;parameters_changed:boolean;
  reference_match:boolean|null;max_reference_weight_error:number|null;source_loss_delta:number;limits:string[];scope:string};

export function UpdateReplay({runId,specText,onSelect}:{runId:string;specText:string;onSelect:(run:string,capture:string)=>void}) {
  const [params]=useSearchParams();
  const environment=useQuery({queryKey:['reference-environment'],queryFn:()=>api<{update_replay_available:boolean;reason:string|null;installed:Record<string,string|null>}>('/system/capabilities')});
  const captures=useQuery({queryKey:['update-captures'],queryFn:()=>api<{items:Capture[]}>('/update-captures')});
  const selected=captures.data?.items.find(c=>c.capture_id===params.get('capture')&&c.run_id===runId)??captures.data?.items.find(c=>c.run_id===runId);
  const [patched,setPatched]=useState(false),[busy,setBusy]=useState(false),[result,setResult]=useState<Result|null>(null),[error,setError]=useState('');
  async function replay(){if(!selected)return;setBusy(true);setError('');setResult(null);try{
    const spec=patched?JSON.parse(specText) as RewardSpec:null;
    setResult(await api<Result>(`/update-captures/${selected.capture_id}/replay`,{run_id:selected.run_id,reward_spec:spec}));
  }catch(e){setError(String(e));}finally{setBusy(false);}}
  return <><section className="panel update-intro"><GitBranch size={28}/><div><h2>One real optimizer step. An explicit evidence boundary.</h2><p>Restore captured numeric model and optimizer state, reuse the fixed Token inputs, then execute SGD or AdamW. No new rollout or future-quality prediction.</p></div><span className="badge violet">CPU reference / bounded GPT-2</span></section>
    {(captures.error||environment.error||error)&&<p role="alert" className="error">{String(captures.error||environment.error||error)}</p>}
    {captures.isLoading?<div className="loading">Verifying local checkpoint manifests…</div>:!captures.data?.items.length?<div className="empty"><h3>No verified update captures</h3><p>Run the included actual deep-capture example with --deep --capture-update. A trace alone cannot reconstruct missing model or optimizer state.</p></div>:<div className="run-grid">{captures.data.items.map(c=><article key={c.capture_id} className="panel run-card"><span className="badge green">real captured optimizer state</span><h2>{c.run_id}</h2><p>{c.model_type} · {c.optimizer} · optimizer step {c.optimizer_step}</p><small className="block mono">{c.capture_id.slice(0,28)}…</small><p>{c.environment.device} · Torch {c.environment.torch}</p><button className="secondary" disabled={busy} onClick={()=>{setResult(null);setError('');onSelect(c.run_id,c.capture_id);}}>Select capture <ArrowRight size={14}/></button></article>)}</div>}
    {environment.data&&!environment.data.update_replay_available&&<div className="boundary"><p>{environment.data.reason}</p><code>pip install 'posttrain-timemachine[trl]==0.2.0'</code><details><summary>Installed reference package versions</summary><pre>{JSON.stringify(environment.data.installed,null,2)}</pre></details></div>}
    {selected&&<section className="panel update-workbench"><div className="panel-head"><div><h2>Replay captured step</h2><p>{selected.run_id} · {selected.optimizer} · optimizer {selected.optimizer_step}</p></div><span className="badge cyan">Hash-verified safetensors</span></div><div className="update-options"><label className="check-label"><input type="checkbox" checked={patched} disabled={busy} onChange={e=>{setPatched(e.target.checked);setResult(null);}}/> Recompute sequence advantages using the current What-if reward draft</label><p>{patched?'Changed rewards affect only this saved input batch. No future training performance is inferred.':'Original captured advantages: compare the actual replay against the recorded native post-update weights.'}</p>{patched&&<details><summary>Inspect reward draft used for this branch</summary><pre>{specText}</pre></details>}<button className="primary" disabled={busy||!environment.data?.update_replay_available} onClick={()=>void replay()}>{busy?<LoaderCircle size={16} className="spin"/>:<Play size={16}/>} {busy?'Executing bounded optimizer step…':'Replay captured optimizer step'}</button></div>
    {result&&<div className="update-result" role="status"><span className={'badge '+(result.parameters_changed?'green':'amber')}>{result.parameters_changed?'REPLAYED UPDATE':'STEP EXECUTED · ZERO WEIGHT DELTA'}</span><div className="diff-kpis"><div><small>ACTUAL PARAMETER Δ L2</small><strong>{result.parameter_delta_l2.toExponential(4)}</strong></div><div><small>REPLAYED LOSS</small><strong>{result.loss.toExponential(4)}</strong></div><div><small>NATIVE REFERENCE</small><strong>{result.reference_match===null?'Branch':result.reference_match?'Match':'Mismatch'}</strong></div></div><p>Maximum native reference weight error: {result.max_reference_weight_error?.toExponential(6)??'Not comparable after a reward patch'}.</p><p>{result.scope}</p><details><summary>Replay evidence and numerical limits</summary><pre>{JSON.stringify(result,null,2)}</pre></details></div>}
    <div className="boundary">This is one actual update of saved weights. It does not predict accuracy after retraining. Weights and optimizer state are excluded from ordinary investigation reports.</div></section>}
  </>;
}
