// Independent of historical charts: an old or unavailable feed never looks current.
export function validateFeed(feed) {
  if (feed?.schema_version !== 1 || !Array.isArray(feed.forecasts) || feed.forecasts.length !== 2) throw Error('Invalid forecast feed');
  const start=Date.parse(feed.forecast_start), end=Date.parse(feed.expires_at), generated=Date.parse(feed.generated_at);
  if (![start,end,generated].every(Number.isFinite) || end-start!==3600000 || generated<start || generated>=end) throw Error('Invalid forecast timestamps');
  if (new Set(feed.forecasts.map(f=>f.symbol)).size!==2 || !feed.forecasts.every(f=>['BTC-USD','ETH-USD'].includes(f.symbol))) throw Error('Invalid assets');
  for (const f of feed.forecasts) {
    if (!Number.isFinite(f.predicted_log_return) || !Number.isFinite(f.last_close_usd) || f.last_close_usd<=0 || !f.model_version || Date.parse(f.decision_time)!==start || Date.parse(f.target_time)!==end) throw Error('Invalid forecast values');
  }
  return feed;
}
export function freshness(feed, now=Date.now()) {
  validateFeed(feed);
  if (now < Date.parse(feed.generated_at)-60000) return 'clock-mismatch';
  return now >= Date.parse(feed.expires_at) ? 'expired' : 'current';
}
if (typeof document !== 'undefined') {
  const root=document.getElementById('hourly');
  if (root) {
    let feed=null, error=false;
    const $=id=>document.getElementById(id);
    const make=(tag,text,cls)=>{const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n;};
    const utc=s=>new Date(s).toLocaleString('en-US',{year:'numeric',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'UTC'})+' UTC';
    function render(){
      const state=feed?freshness(feed):'unavailable';
      $('hourly-status').textContent=error?'Update check failed':state==='current'?'Current hourly forecast':state==='expired'?'Expired · awaiting update':state==='clock-mismatch'?'Check your device clock':'Forecasts unavailable';
      $('hourly-status').className='badge '+(state==='current'&&!error?'hourly-current':'hourly-warning');
      $('hourly-warning').hidden=state==='current'&&!error;
      $('hourly-warning').textContent=error?'Could not check for an update. Any forecasts below are the last successfully loaded values; check their validity window.':state==='expired'?'These forecasts have expired. The next scheduled update may be delayed; do not treat the old values as current.':state==='clock-mismatch'?'Your device clock appears earlier than the forecast generation time. Verify the timestamps before using this display.':'The hourly feed is not available yet. Historical research below is unchanged.';
      if(!feed)return;
      $('hourly-updated').textContent=`Generated ${utc(feed.generated_at)} · forecast ends ${utc(feed.expires_at)}`;
      $('hourly-cards').replaceChildren(...feed.forecasts.map(f=>{
        const card=make('article','','card hourly-card');
        card.append(make('div',f.symbol.replace('-',' / '),'eyebrow'),make('h3',`${f.predicted_log_return>=0?'+':''}${(f.predicted_log_return*10000).toFixed(2)} bps`),make('p','Predicted next-hour log return','muted'));
        const rows=[['Last closed candle',new Intl.NumberFormat('en-US',{style:'currency',currency:'USD'}).format(f.last_close_usd)],['Forecast window',`${utc(f.decision_time)} → ${utc(f.target_time)}`],['Model',f.model_version],['Training cutoff',utc(f.trained_through)],['Validation cutoff',utc(f.validation_through)]];
        for(const [label,value] of rows){const row=make('div','','hourly-detail');row.append(make('span',label),make('strong',value));card.append(row);}
        return card;
      }));
    }
    async function load(){try{const response=await fetch(`latest-forecasts.json?t=${Date.now()}`,{cache:'no-store'});if(!response.ok)throw Error('Feed unavailable');const next=validateFeed(await response.json());if(feed&&Date.parse(next.generated_at)<Date.parse(feed.generated_at))throw Error('Older cached feed');feed=next;error=false;}catch{error=true;}render();}
    load();setInterval(load,60000);setInterval(render,15000);
  }
}
