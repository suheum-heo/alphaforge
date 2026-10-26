export function health(data, now=Date.now()) {
  if(data?.schema_version!==1 || !Array.isArray(data.metrics) || !Number.isFinite(Date.parse(data.updated_at))) throw Error('Invalid performance feed');
  if(now-Date.parse(data.updated_at)>90*60000) return 'Updates delayed';
  if(data.last_attempt?.forecast!=='ok' || data.last_attempt?.scoring!=='ok') return 'Data update needs attention';
  return data.metrics.some(r=>r.missing_candles>0)?'Missing target candles':'Tracking hourly outcomes';
}
if(typeof document!=='undefined') {
  let data=null, failed=false;
  const status=document.getElementById('performance-status');
  function render(){
    if(!data)return;
    status.textContent=failed?'Performance update unavailable':health(data);
    const total=data.metrics.reduce((n,r)=>n+r.scored,0);
    document.getElementById('performance-health').textContent=`${total} scored forecasts · ${data.missing_forecast_slots} missed asset/hour slots since tracking began · updated ${new Date(data.updated_at).toISOString().replace('T',' ').replace('.000Z',' UTC')}. ${total<672?'Early sample — results are not yet conclusive.':''}`;
    const body=document.getElementById('performance-table');body.replaceChildren();
    for(const row of data.metrics){
      const tr=document.createElement('tr');
      const metric=k=>Number.isFinite(row[k])?row[k].toFixed(2):'Awaiting outcomes';
      for(const text of [`${row.symbol} · ${row.model_version}`,row.scored,metric('rmse_bps'),metric('baseline_rmse_bps'),metric('mae_bps'),metric('baseline_mae_bps'),`${row.pending} / ${row.missing_candles}`]){
        const td=document.createElement('td');td.textContent=text;tr.append(td);
      }body.append(tr);
    }
    if(!data.metrics.length){const tr=document.createElement('tr'),td=document.createElement('td');td.colSpan=7;td.textContent='Waiting for the first prospective forecasts.';tr.append(td);body.append(tr);}
  }
  async function load(){try{const res=await fetch(`live-performance.json?t=${Date.now()}`,{cache:'no-store'});if(!res.ok)throw Error();const next=await res.json();health(next);data=next;failed=false;render();}catch{failed=true;status.textContent='Performance update unavailable';}}
  load();setInterval(load,60000);setInterval(()=>{if(data)render();},15000);
}
