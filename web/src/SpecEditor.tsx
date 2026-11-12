import { useEffect, useRef } from 'react';
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
import 'monaco-editor/esm/vs/language/json/monaco.contribution';
import EditorWorker from 'monaco-editor/esm/vs/editor/editor.worker?worker';
import JsonWorker from 'monaco-editor/esm/vs/language/json/json.worker?worker';
import { api } from './api';

(self as unknown as {MonacoEnvironment:unknown}).MonacoEnvironment={getWorker:(_id:string,label:string)=>label==='json'?new JsonWorker():new EditorWorker()};
export default function SpecEditor({value,onChange}:{value:string;onChange:(v:string)=>void}) {
  const ref=useRef<HTMLDivElement>(null), editor=useRef<monaco.editor.IStandaloneCodeEditor|null>(null);
  const change=useRef(onChange);change.current=onChange;
  useEffect(()=>{
    if (!ref.current) return;
    void api<Record<string,unknown>>('/schemas/reward-spec').then(schema=>{
      monaco.languages.json.jsonDefaults.setDiagnosticsOptions({validate:true,enableSchemaRequest:false,
        schemas:[{uri:'ptm://schemas/reward-spec',fileMatch:['*'],schema}]});
    }).catch(()=>{/* Server-side validation remains authoritative when schema loading fails. */});
    monaco.editor.defineTheme('ptm',{base:'vs-dark',inherit:true,rules:[],colors:{'editor.background':'#0b1222','editorLineNumber.foreground':'#8192ad'}});
    const instance=monaco.editor.create(ref.current,{value,language:'json',theme:'ptm',automaticLayout:true,
      minimap:{enabled:false},fontSize:12,lineNumbers:'on',scrollBeyondLastLine:false,wordWrap:'on',
      ariaLabel:'RewardSpec JSON editor',tabSize:2});
    editor.current=instance;
    const listener=instance.onDidChangeModelContent(()=>change.current(instance.getValue()));
    return()=>{listener.dispose();instance.getModel()?.dispose();instance.dispose();};
  },[]);
  useEffect(()=>{if(editor.current&&editor.current.getValue()!==value)editor.current.setValue(value);},[value]);
  return <div ref={ref} className="spec-editor"/>;
}
