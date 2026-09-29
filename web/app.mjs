import {refine, makeFixture, exact} from './core.mjs';
import {runNeuralBenchmark} from './neural.mjs';

window.runNumericBenchmark=()=>{
  const n=Number(document.querySelector('#options').value), margin=Number(document.querySelector('#margin').value);
  const fixture=makeFixture(n,64,margin), start=performance.now();
  const result=refine(fixture,{batchSize:64}), ms=performance.now()-start;
  const full=exact(fixture), contained=full[result.value]>=result.interval[0]-1e-10&&full[result.value]<=result.interval[1]+1e-10;
  const r={status:'synthetic_uncalibrated',options:n,chunks:64,scheduler_ms:ms,selected:result.value,
    approximate_probability:result.probability,full_probability:full[result.value],probability_interval:result.interval,
    contains_reference:contained,argmax_certified:result.certified,chunks_refined:result.chunksRefined,
    options_refined:result.optionsRefined,rounds:result.rounds};
  const target=document.querySelector('#result');target.replaceChildren();
  const p=document.createElement('p');p.className='number';p.textContent=`${(result.probability*100).toFixed(3)} %`;target.append(p);
  const label=document.createElement('p');label.textContent=`Option ${result.value} · distribution synthétique`;target.append(label);
  const pre=document.createElement('pre');pre.textContent=JSON.stringify(r,null,2);target.append(pre);
  return r;
};
window.runNeuralBenchmark=runNeuralBenchmark;
document.querySelector('#numeric').onclick=window.runNumericBenchmark;
document.querySelector('#neural').onclick=async()=>{
  const btn=document.querySelector('#neural');btn.disabled=true;
  try {await runNeuralBenchmark();} catch(e) {document.querySelector('#neural-status').textContent=String(e);}
  finally{btn.disabled=false;}
};
window.dexReady=true;
