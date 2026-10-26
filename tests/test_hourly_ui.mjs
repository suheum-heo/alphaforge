import test from 'node:test';
import assert from 'node:assert/strict';
import {freshness,validateFeed} from '../src/alphaforge/web/hourly.mjs';
const base={schema_version:1,forecast_start:'2026-10-26T05:00:00Z',expires_at:'2026-10-26T06:00:00Z',generated_at:'2026-10-26T05:17:00Z',forecasts:['BTC-USD','ETH-USD'].map(symbol=>({symbol,decision_time:'2026-10-26T05:00:00Z',target_time:'2026-10-26T06:00:00Z',predicted_log_return:.001,last_close_usd:1000,model_version:'test-v1'}))};
test('forecast expires exactly at target time',()=>{assert.equal(freshness(base,Date.parse('2026-10-26T05:59:59Z')),'current');assert.equal(freshness(base,Date.parse('2026-10-26T06:00:00Z')),'expired');});
test('clock skew is explicit',()=>assert.equal(freshness(base,Date.parse('2026-10-26T04:00:00Z')),'clock-mismatch'));
test('reject corrupt or partial feeds',()=>{assert.throws(()=>validateFeed({...base,forecasts:base.forecasts.slice(1)}));assert.throws(()=>validateFeed({...base,expires_at:'bad'}));assert.throws(()=>validateFeed({...base,forecasts:[base.forecasts[0],base.forecasts[0]]}));});

import {health} from '../src/alphaforge/web/performance.mjs';
test('performance health exposes stale updates and provider failures',()=>{
  const now=Date.parse('2026-10-26T06:17:00Z');
  const data={schema_version:1,metrics:[],updated_at:new Date(now).toISOString(),last_attempt:{forecast:'ok',scoring:'ok'}};
  assert.equal(health(data,now),'Tracking hourly outcomes');
  assert.equal(health(data,now+91*60000),'Updates delayed');
  data.last_attempt.forecast='failed';
  assert.equal(health(data,now),'Data update needs attention');
  assert.throws(()=>health({},now));
});
