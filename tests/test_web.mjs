import test from 'node:test';
import assert from 'node:assert/strict';
import {softmax, probabilityBounds, eventBounds, refine, makeFixture, exact} from '../web/core.mjs';

test('extreme logits, one class and event complements', ()=>{
  for (const a of [[1000,-1000,0],[1e6,1e6-1,-1e6],[0]]) {
    const p=softmax(a),[lo,hi]=probabilityBounds(a,a);
    for(let i=0;i<a.length;i++) {assert.ok(Math.abs(lo[i]-p[i])<1e-10);assert.ok(Math.abs(hi[i]-p[i])<1e-10);}
  }
  const l=[-1,0,2],u=[1,2,3];
  const a=eventBounds(l,u,[0]),b=eventBounds(l,u,[1,2]);
  assert.ok(Math.abs(a[0]+b[1]-1)<1e-14);
});
test('adaptive envelopes contain full distribution at every budget',()=>{
  for(let seed=0;seed<12;seed++) {
    const f=makeFixture(61,11,seed%2 ? .3 : 4,seed),p=exact(f);
    for(let rounds=0;rounds<28;rounds++) {
      const r=refine(f,{maxRounds:rounds,tolerance:0,batchSize:3});
      const [lo,hi]=probabilityBounds(r.low,r.high);
      for(let i=0;i<p.length;i++) assert.ok(p[i]>=lo[i]-1e-10&&p[i]<=hi[i]+1e-10);
      if(r.certified) assert.equal(r.value,p.indexOf(Math.max(...p)));
    }
    const r=refine(f,{tolerance:.01});
    assert.ok(r.certified&&Math.abs(r.probability-p[r.value])<=.01+1e-10);
  }
});
