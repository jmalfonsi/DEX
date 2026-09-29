export async function runNeuralBenchmark() {
  const status=document.querySelector('#neural-status');
  status.textContent='Chargement des graphes locaux…';
  const {default:ort}=await import('./vendor/ort.wasm.bundle.min.mjs');
  ort.env.wasm.numThreads=1;
  ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
  const get=async path=>{const res=await fetch(path);if(!res.ok)throw Error(`Fichier manquant : ${path}`);return res.json();};
  const [manifest,fixture]=await Promise.all([get('../artifacts/model-manifest.json'),get('../artifacts/browser-fixture.json')]);
  const startLoad=performance.now();
  const encoder=await ort.InferenceSession.create('../artifacts/encoder.int8.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'basic'});
  const core=await ort.InferenceSession.create('../artifacts/core.fp32.onnx',{executionProviders:['wasm'],graphOptimizationLevel:'basic'});
  const loadMs=performance.now()-startLoad;
  const intTensor=(data,shape)=>new ort.Tensor('int64',BigInt64Array.from(data,BigInt),shape);
  const floatTensor=(data,shape)=>new ort.Tensor('float32',Float32Array.from(data),shape);
  const encode=async(data,shape,kind)=>encoder.run({ids:intTensor(data,shape),kinds:intTensor(Array(shape[0]).fill(kind),[shape[0]])});
  const tSchema=performance.now();
  const questions=await encode(fixture.questions,fixture.question_shape,1);
  const options=await encode(fixture.options,fixture.option_shape,2);
  const bank=Float32Array.from(options.summaries.data),rank=manifest.config.rank;
  for(let j=0;j<fixture.option_shape[0];j++) {let norm=0;for(let k=0;k<rank;k++)norm+=bank[j*rank+k]**2;norm=Math.sqrt(norm);for(let k=0;k<rank;k++)bank[j*rank+k]/=Math.max(norm,1e-12);}
  const schemaMs=performance.now()-tSchema;
  const stateFeed={ids:intTensor(fixture.state,fixture.state_shape),kinds:intTensor(Array(fixture.state_shape[0]).fill(0),[fixture.state_shape[0]])};
  const read=()=>encoder.run(stateFeed);
  let state=await read();
  const decide=()=>core.run({questions:questions.summaries,summaries:state.summaries,tokens:state.tokens,
    valid:new ort.Tensor('bool',Uint8Array.from(fixture.state,x=>x!==0),fixture.state_shape),
    bank:floatTensor(bank,[fixture.option_shape[0],rank])});
  await decide();
  const encoderMs=[],coreMs=[];
  for(let i=0;i<5;i++){
    status.textContent=`Mesure ${i+1}/5…`;
    let t=performance.now();state=await read();encoderMs.push(performance.now()-t);
    t=performance.now();await decide();coreMs.push(performance.now()-t);
  }
  const actual=(await decide()).logits.data;
  const referenceCore=(await core.run({
    questions:floatTensor(fixture.expected_questions,[fixture.question_shape[0],rank]),
    summaries:floatTensor(fixture.expected_summaries,[fixture.state_shape[0],rank]),
    tokens:floatTensor(fixture.expected_tokens,[...fixture.state_shape,rank]),
    valid:new ort.Tensor('bool',Uint8Array.from(fixture.state,x=>x!==0),fixture.state_shape),
    bank:floatTensor(fixture.expected_bank,[fixture.option_shape[0],rank])})).logits.data;
  const error=(a,b)=>a.reduce((m,x,i)=>Math.max(m,Math.abs(x-b[i])),0);
  const diagnostics={token_error:error(state.tokens.data,fixture.expected_tokens),
    summary_error:error(state.summaries.data,fixture.expected_summaries),
    question_error:error(questions.summaries.data,fixture.expected_questions),
    bank_error:error(bank,fixture.expected_bank),core_only_error:error(referenceCore,fixture.expected_logits)};
  let maxError=0;for(let i=0;i<actual.length;i++)maxError=Math.max(maxError,Math.abs(actual[i]-fixture.expected_logits[i]));
  const median=a=>[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)];
  const result={status:'random_weights_untrained_uncalibrated',backend:'WASM CPU 1 thread',
    user_agent:navigator.userAgent,cross_origin_isolated:crossOriginIsolated,
    parameters:manifest.parameters,model_bytes:Object.values(manifest.files).reduce((a,b)=>a+b,0),
    context_tokens:fixture.state.length,fields:fixture.question_shape[0],options:fixture.option_shape[0],
    session_load_ms:loadMs,schema_compile_ms:schemaMs,encoder_ms:encoderMs,core_ms:coreMs,
    encoder_median_ms:median(encoderMs),core_median_ms:median(coreMs),
    max_abs_error_vs_onnx_cpu:maxError,diagnostics,parity_passed:maxError<1e-4,
    note:'5 observations; no reliable p95. Excludes download, tokenization, calibration and adaptive scheduler.'};
  const pre=document.querySelector('#neural-result');pre.hidden=false;pre.textContent=JSON.stringify(result,null,2);
  status.textContent=result.parity_passed?'Exécution locale vérifiée ; qualité sémantique non mesurée.':'Exécution terminée ; écart numérique à examiner.';
  await encoder.release();await core.release();
  return result;
}
