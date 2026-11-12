import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, DataZoomComponent, MarkLineComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { Metric } from './types';

echarts.use([LineChart, GridComponent, TooltipComponent, DataZoomComponent, MarkLineComponent, CanvasRenderer]);
export function Timeline({metrics, onStep}: {metrics: Metric[]; onStep:(step:number)=>void}) {
  const ref = useRef<HTMLDivElement>(null);
  const accuracyKey=metrics.some(m=>m.name==='heldout_accuracy')?'heldout_accuracy':'verifier_accuracy';
  const accuracyLabel=accuracyKey==='heldout_accuracy'?'Held-out accuracy':'Saved-sample verifier accuracy';
  const steps = [...new Set(metrics.map(m=>m.generation_step))].sort((a,b)=>a-b);
  useEffect(()=>{
    if (!ref.current || !metrics.length) return;
    const chart = echarts.init(ref.current, undefined, {renderer:'canvas'});
    const steps = [...new Set(metrics.map(m=>m.generation_step))].sort((a,b)=>a-b);
    chart.setOption({animation:false, backgroundColor:'transparent',
      grid:{left:48,right:48,top:30,bottom:62},
      tooltip:{trigger:'axis',backgroundColor:'#172138',borderColor:'#33445e',textStyle:{color:'#eaf4ff'},confine:true},
      xAxis:{type:'category',data:steps,axisLabel:{color:'#9aaec9',fontFamily:'monospace'},axisLine:{lineStyle:{color:'#25324a'}}},
      yAxis:[{type:'value',name:'Reward',nameTextStyle:{color:'#4ce8d6'},axisLabel:{color:'#9aaec9'},splitLine:{lineStyle:{color:'#1c2941',type:'dashed'}}},
        {type:'value',min:0,max:1,name:'Accuracy',nameTextStyle:{color:'#bcaaff'},axisLabel:{color:'#9aaec9',formatter:'{value}'},splitLine:{show:false}}],
      dataZoom:[{type:'inside',filterMode:'none'},{type:'slider',height:18,bottom:8,borderColor:'#25324a',backgroundColor:'#0b1120',
        fillerColor:'#4ce8d614',dataBackground:{lineStyle:{color:'#4ce8d6'}},textStyle:{color:'#9aaec9'}}],
      series:[{name:'Reward (left axis)',type:'line',yAxisIndex:0,symbol:'circle',symbolSize:6,lineStyle:{width:3,color:'#4ce8d6'},
        itemStyle:{color:'#4ce8d6'},areaStyle:{color:new echarts.graphic.LinearGradient(0,0,0,1,[{offset:0,color:'#4ce8d61c'},{offset:1,color:'#4ce8d600'}])},
        data:steps.map(s=>metrics.find(m=>m.name==='reward'&&m.generation_step===s)?.value??null)},
        {name:accuracyLabel+' (right axis)',type:'line',yAxisIndex:1,symbol:'diamond',symbolSize:7,lineStyle:{width:2,color:'#ac97ff',type:'dashed'},
        itemStyle:{color:'#ac97ff'},data:steps.map(s=>metrics.find(m=>m.name===accuracyKey&&m.generation_step===s)?.value??null)}]});
    chart.on('click', p=>{if (p.componentType==='series') onStep(steps[p.dataIndex]);});
    const observer=new ResizeObserver(()=>chart.resize()); observer.observe(ref.current);
    return ()=>{observer.disconnect();chart.dispose();};
  },[metrics,onStep]);
  return <><div ref={ref} className="chart" role="img" aria-label={`Reward and ${accuracyLabel} on separate axes. Raw points, aligned by generation step. Click a point to reveal groups.`}/>
    <details className="chart-table"><summary>Accessible data table · {steps.length} raw generation steps</summary>
      <table><caption>Observed metric values; no smoothing. Independent axes in chart.</caption><thead><tr><th>Generation step</th><th>Reward</th><th>{accuracyLabel}</th></tr></thead>
      <tbody>{steps.map(s=><tr key={s}><td><button onClick={()=>onStep(s)}>{s}</button></td><td>{metrics.find(m=>m.name==='reward'&&m.generation_step===s)?.value.toFixed(3)??'Unavailable'}</td><td>{metrics.find(m=>m.name===accuracyKey&&m.generation_step===s)?.value.toFixed(3)??'Unavailable'}</td></tr>)}</tbody></table></details></>;
}
