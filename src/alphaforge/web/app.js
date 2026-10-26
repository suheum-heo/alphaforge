'use strict';
const labels={zero_return:'Zero return',training_mean:'Training mean',ridge:'Ridge',lstm:'LSTM',transformer:'Transformer',buy_hold:'Always long',cash:'Cash'};
const colors={zero_return:'#a4aeb8',training_mean:'#c5a353',ridge:'#cf7548',lstm:'#187b69',transformer:'#7978b9',buy_hold:'#9ba8b5'};
const $=id=>document.getElementById(id);
const pct=n=>n==null?'—':`${(n*100).toFixed(2)}%`;
const bps=n=>n==null?'—':(n*10000).toFixed(2);
const name=k=>labels[k]||k;
const fmtDate=s=>new Date(s).toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric',timeZone:'UTC'});
function el(tag,text,cls){const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;}
function metric(label,value,detail,cls=''){const n=el('article',null,'metric');n.append(el('div',label,'label'),el('strong',value,cls),el('small',detail));return n;}
function legend(id,series){$(id).replaceChildren(...series.map(s=>{const n=el('span');const dot=el('i');dot.style.background=s.color;n.append(dot,document.createTextNode(s.label));return n;}));}
function chart(id,series,format,title,{zero=false}={}){
 const host=$(id);host.replaceChildren();const valid=series.filter(s=>s.points.length);if(!valid.length){host.append(el('p','No observations available.','muted'));return;}
 const all=valid.flatMap(s=>s.points),w=1040,h=270,left=70,right=22,top=16,bottom=32;
 let min=Math.min(...all.map(p=>p.value)),max=Math.max(...all.map(p=>p.value));
 if(zero){min=Math.min(min,0);max=Math.max(max,0);}if(min===max){min-=.01;max+=.01;}
 const pad=(max-min)*.08;min-=pad;max+=pad;
 const times=all.map(p=>Date.parse(p.time)),start=Math.min(...times),end=Math.max(...times);
 const x=t=>left+(Date.parse(t)-start)/Math.max(end-start,1)*(w-left-right),y=v=>top+(max-v)/(max-min)*(h-top-bottom);
 const ns='http://www.w3.org/2000/svg';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',title);
 function node(tag,attrs,text){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!=null)n.textContent=text;svg.append(n);return n;}
 node('title',{},title);node('desc',{},valid.map(s=>`${s.label}: last value ${format(s.points.at(-1).value)}`).join('. '));
 for(let i=0;i<5;i++){const value=min+(max-min)*i/4,yy=y(value);node('line',{x1:left,x2:w-right,y1:yy,y2:yy,stroke:'#edf0f3'});node('text',{x:left-12,y:yy+4,'text-anchor':'end'},format(value));}
 for(let i=0;i<4;i++){const t=new Date(start+(end-start)*i/3);node('text',{x:left+(w-left-right)*i/3,y:h-6,'text-anchor':i===0?'start':i===3?'end':'middle'},t.toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'}));}
 if(zero&&min<0&&max>0)node('line',{x1:left,x2:w-right,y1:y(0),y2:y(0),stroke:'#cbd3dc','stroke-dasharray':'4 4'});
 valid.forEach(s=>node('path',{d:s.points.map((p,i)=>`${i?'L':'M'}${x(p.time).toFixed(2)},${y(p.value).toFixed(2)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':s.width||2,'stroke-linejoin':'round','stroke-linecap':'round'}));
 host.append(svg);
}
let data;
function renderModels(){
 const symbol=$('asset').value,split=$('split').value,result=data.comparison[symbol],winner=result.selected_by_validation_rmse;
 const entries=Object.entries(result.models);const max=Math.max(...entries.map(([,m])=>m[split].rmse));
 $('error-bars').replaceChildren(...entries.map(([kind,m])=>{const row=el('div',null,`bar-row ${kind===winner?'selected':''}`),track=el('div',null,'bar-track'),fill=el('div',null,'bar-fill');fill.style.width=`${m[split].rmse/max*100}%`;track.append(fill);row.append(el('span',name(kind)),track,el('strong',bps(m[split].rmse)));return row;}));
 $('model-table').replaceChildren(...entries.map(([kind,m])=>{const row=el('tr'),first=el('td',name(kind));if(kind===winner)first.append(el('span','SELECTED','tag'));row.append(first,...[bps(m[split].rmse),bps(m[split].mae),pct(m[split].direction_accuracy),pct(m[split].direction_coverage)].map(v=>el('td',v)));return row;}));
 $('winner').textContent=name(winner);$('winner-score').textContent=`${bps(result.models[winner].test.rmse)} bps`;
 const delta=result.models[winner].test.rmse/result.models.zero_return.test.rmse-1;$('winner-delta').textContent=`${delta>=0?'+':''}${pct(delta)} error`;$('winner-delta').className=delta>0?'negative':'positive';
 renderForecast();
}
function renderForecast(){const rows=data.forecasts[$('asset').value][$('forecast-model').value]||[];chart('forecast-chart',[{label:'Observed return',color:'#b9c3ce',width:1.5,points:rows.map(r=>({time:r.time,value:r.actual}))},{label:'Forecast',color:'#187b69',points:rows.map(r=>({time:r.time,value:r.prediction}))}],v=>`${(v*100).toFixed(1)}%`,'Historical observed and predicted hourly returns',{zero:true});}
function renderBacktest(){const choice=$('backtest-model').value,entries=Object.entries(data.backtests).filter(([k])=>choice==='all'||k===choice);const series=entries.map(([k,v])=>({label:name(k),color:colors[k],points:v.equity}));legend('equity-legend',series);chart('equity-chart',series,v=>`$${Math.round(v*10000).toLocaleString()}`,'Historical cost-adjusted walk-forward equity');$('backtest-stats').replaceChildren(...entries.map(([k,v])=>metric(`${name(k)} · net return`,pct(v.stats.total_return),`${pct(v.stats.max_drawdown)} drawdown · ${v.folds} folds`,v.stats.total_return<0?'negative':'positive')));}
function renderStudy(){const s=data.study;if(!s)return;$('study-pending').hidden=true;$('study-ready').hidden=false;$('study-winner').textContent=name(s.selected_portfolio);if(s.selected_portfolio!=='cash'&&s.results[s.selected_portfolio].holdout_strategy.turnover===0){$('study-description').textContent='The frozen rule produced no positions in the holdout. A 0% return here means staying in cash, not a demonstrated trading edge.';}$('study-return').textContent=pct(s.selected_holdout_return);$('study-return').className=s.selected_holdout_return<0?'negative':'positive';const series=Object.entries(s.equity).map(([k,points])=>({label:name(k),color:colors[k],points}));legend('study-legend',series);chart('study-chart',series,v=>`$${Math.round(v*10000).toLocaleString()}`,'Fresh July through September 2026 holdout equity');$('study-table').replaceChildren(...Object.entries(s.results).map(([kind,r])=>{const row=el('tr'),first=el('td',name(kind));if(kind===s.selected_portfolio)first.append(el('span','FROZEN CHOICE','tag'));row.append(first,el('td',bps(s.selection[kind].threshold)),el('td',pct(r.holdout_strategy.total_return),r.holdout_strategy.total_return<0?'negative':'positive'),el('td',pct(r.zero_threshold_strategy.total_return)),el('td',r.holdout_strategy.turnover.toFixed(0)));return row;}));}
async function main(){try{const response=await fetch('dashboard-data.json');if(!response.ok)throw new Error('Missing dashboard export');data=await response.json();$('date-range').textContent=`Benchmark · ${fmtDate(data.dataset.start)} — ${fmtDate(data.dataset.end)}`;$('metrics').replaceChildren(metric('Historical candles',data.dataset.rows.toLocaleString(),'BTC + ETH · benchmark dataset'),metric('Model families','5','Baselines, LSTM & Transformer'),metric('Prediction interval','1 hour','UTC-aligned, closed candles'),metric('Evaluation','Out of sample','Chronological splits · purged labels'));renderModels();renderBacktest();renderStudy();$('asset').addEventListener('change',renderModels);$('split').addEventListener('change',renderModels);$('forecast-model').addEventListener('change',renderForecast);$('backtest-model').addEventListener('change',renderBacktest);const nav=[...document.querySelectorAll('nav a')];nav.forEach(a=>a.addEventListener('click',()=>{nav.forEach(n=>n.classList.remove('active'));a.classList.add('active');}));}catch(error){$('error').hidden=false;$('error').textContent='Could not load experiment data. Serve this folder over HTTP, or run the dashboard export again. '+error.message;}}
main();
