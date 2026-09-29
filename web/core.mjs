// DEX numerical research reference. No language model is hidden in these kernels.
const lse2 = (a, b) => {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const m = Math.max(a, b);
  return m + Math.log1p(Math.exp(-Math.abs(a - b)));
};
function validateVector(a) {
  if (!a.length || !Array.from(a).every(Number.isFinite)) throw Error('Invalid finite vector');
}
export function softmax(a) {
  validateVector(a);
  let max = -Infinity;
  for (const x of a) max = Math.max(max, x);
  const out = Float64Array.from(a, x => Math.exp(x - max));
  let total = 0;
  for (const x of out) total += x;
  return out.map(x => x / total);
}
function excluding(a) {
  const out = new Float64Array(a.length);
  let left = -Infinity;
  for (let i = 0; i < a.length; i++) { out[i] = left; left = lse2(left, a[i]); }
  let right = -Infinity;
  for (let i = a.length - 1; i >= 0; i--) { out[i] = lse2(out[i], right); right = lse2(right, a[i]); }
  return out;
}
export function probabilityBounds(low, high) {
  validateVector(low); validateVector(high);
  if (low.length !== high.length || low.some((x, i) => x > high[i])) throw Error('Invalid intervals');
  const exceptHi = excluding(high), exceptLo = excluding(low);
  return [Float64Array.from(low, (x, i) => Math.exp(x - lse2(x, exceptHi[i]))),
          Float64Array.from(high, (x, i) => Math.exp(x - lse2(x, exceptLo[i])))];
}
export function eventBounds(low, high, members) {
  probabilityBounds(low, high); // shape/finite validation
  const selected = new Set(members);
  if (selected.size !== members.length || members.some(i => !Number.isInteger(i) || i < 0 || i >= low.length)) throw Error('Invalid event');
  if (!selected.size) return [0, 0];
  if (selected.size === low.length) return [1, 1];
  let a = -Infinity, b = -Infinity, c = -Infinity, d = -Infinity;
  for (let i = 0; i < low.length; i++) {
    if (selected.has(i)) { a = lse2(a, low[i]); c = lse2(c, high[i]); }
    else { b = lse2(b, high[i]); d = lse2(d, low[i]); }
  }
  return [Math.exp(a - lse2(a, b)), Math.exp(c - lse2(c, d))];
}
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const argmax = a => a.reduce((best, x, i) => x > a[best] ? i : best, 0);

export function refine(f, {tolerance = .01, batchSize = 16, maxRounds = Infinity} = {}) {
  const {coarse, weights, bank, delta, correction, rho = 1, optionBound = 1, scale = 8, temperature = 1} = f;
  const c = weights.length, n = bank.length, rank = coarse.length;
  validateVector(coarse); validateVector(weights); validateVector(correction);
  if (!n || !rank || delta.length !== c || correction.length !== n || temperature <= 0 || rho < 0 || optionBound < 0 || scale <= 0 ||
      ![temperature, rho, optionBound, scale, tolerance, batchSize].every(Number.isFinite) || tolerance < 0 || batchSize < 1 || !Number.isInteger(batchSize) ||
      weights.some(x => x < 0) || Math.abs(weights.reduce((a,b)=>a+b,0)-1)>1e-8) throw Error('Invalid fixture');
  for (const row of [...bank, ...delta]) { validateVector(row); if (row.length !== rank) throw Error('Invalid rank'); }
  if (delta.some(row => Math.sqrt(dot(row,row)) > rho) || correction.some(x => Math.abs(x) > optionBound)) throw Error('Invalid residual bound');
  const stateSeen = new Uint8Array(c).fill(rho === 0 ? 1 : 0);
  const optionSeen = new Uint8Array(n).fill(optionBound === 0 ? 1 : 0);
  const latent = Float64Array.from(coarse), corrections = new Float64Array(n);
  let previousLow = new Float64Array(n).fill(-Infinity), previousHigh = new Float64Array(n).fill(Infinity);
  const norms = bank.map(row => Math.sqrt(dot(row, row)));
  let chunksRefined = 0, optionsRefined = 0, rounds = 0;
  while (true) {
    let tail = 0;
    for (let j = 0; j < c; j++) if (!stateSeen[j]) tail += weights[j];
    const full = stateSeen.every(Boolean) && optionSeen.every(Boolean);
    const low = new Float64Array(n), high = new Float64Array(n), stateRadius = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const center = (scale * dot(bank[i], latent) + corrections[i]) / temperature;
      stateRadius[i] = scale * rho * tail * norms[i];
      const radius = (stateRadius[i] + (optionSeen[i] ? 0 : optionBound)) / temperature + (full ? 0 : 1e-7);
      low[i] = Math.max(previousLow[i], center - radius);
      high[i] = Math.min(previousHigh[i], center + radius);
      if (low[i] > high[i] + 1e-10) throw Error('Inconsistent envelope');
      low[i] = Math.min(low[i], high[i]);
    }
    const [pLow, pHigh] = probabilityBounds(low, high);
    let best = argmax(low), otherMax = -Infinity;
    for (let i = 0; i < n; i++) if (i !== best) otherMax = Math.max(otherMax, high[i]);
    const certified = low[best] >= otherMax;
    const toleranceMet = certified && pHigh[best] - pLow[best] <= tolerance;
    if (toleranceMet || full || rounds >= maxRounds) {
      const probabilities = softmax(low.map((x, i) => (x + high[i]) / 2));
      if (!certified) best = argmax(probabilities);
      return {value: best, probability: probabilities[best], interval: [pLow[best], pHigh[best]],
        probabilities, low, high, certified, toleranceMet, chunksRefined, optionsRefined, rounds, calibrated: false};
    }
    previousLow = low; previousHigh = high;
    if (!stateSeen.every(Boolean) && (optionSeen.every(Boolean) || stateRadius[best] >= optionBound)) {
      const ids = Array.from({length:c}, (_,i)=>i).filter(i=>!stateSeen[i]).sort((a,b)=>weights[b]-weights[a]).slice(0,batchSize);
      for (const id of ids) { for (let k=0;k<rank;k++) latent[k] += weights[id] * delta[id][k]; stateSeen[id]=1; }
      chunksRefined += ids.length;
    } else {
      const ids = Array.from({length:n},(_,i)=>i).filter(i=>!optionSeen[i]).sort((a,b)=>high[b]-high[a]).slice(0,batchSize);
      for (const id of ids) { corrections[id]=correction[id]; optionSeen[id]=1; }
      optionsRefined += ids.length;
    }
    rounds++;
  }
}

export function makeFixture(n=1024, c=64, margin=5, seed=29) {
  const rank=16;
  let s=seed>>>0;
  const random=()=>{s=(Math.imul(1664525,s)+1013904223)>>>0;return (s+.5)/4294967296;};
  const normal=()=>Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random());
  const bank=Array.from({length:n},()=>{const v=Array.from({length:rank},normal);const norm=Math.sqrt(dot(v,v));return v.map(x=>x/norm);});
  const coarse=bank[0].map(x=>margin*x);
  const weights=Array.from(softmax(Array.from({length:c},()=>normal()*3)));
  const delta=Array.from({length:c},()=>Array.from({length:rank},()=>Math.tanh(normal())/Math.sqrt(rank)));
  const correction=Array.from({length:n},()=>Math.tanh(normal()));
  return {coarse,weights,bank,delta,correction};
}
export function exact(f) {
  const latent=f.coarse.map((x,k)=>x+f.weights.reduce((s,w,j)=>s+w*f.delta[j][k],0));
  return softmax(f.bank.map((row,i)=>((f.scale??8)*dot(row,latent)+f.correction[i])/(f.temperature??1)));
}
