"""Prepare a deterministic end-to-end ONNX fixture and local ORT Web runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-source", type=Path, help="Optional directory containing ORT Web 1.30.0 WASM bundle")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact = root/"artifacts"
    manifest = json.loads((artifact/"model-manifest.json").read_text())
    vocab, rank = manifest["config"]["vocab"], manifest["config"]["rank"]
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    encoder = ort.InferenceSession(str(artifact/"encoder.int8.onnx"), sess_options=session_options)
    core = ort.InferenceSession(str(artifact/"core.fp32.onnx"), sess_options=session_options)
    def inputs(rows, length, seed):
        return (np.arange(rows*length, dtype=np.int64).reshape(rows,length)*17 + seed) % (vocab-1) + 1
    state, questions, options = inputs(4,128,7), inputs(4,32,17), inputs(32,32,29)
    def enc(ids, kind):
        return encoder.run(None, {"ids":ids,"kinds":np.full(ids.shape[0],kind,dtype=np.int64)})
    tokens, summaries = enc(state,0)
    _, qs = enc(questions,1)
    _, bank = enc(options,2)
    bank /= np.maximum(np.linalg.norm(bank,axis=1,keepdims=True),1e-12)
    logits = core.run(None,{"questions":qs,"summaries":summaries,"tokens":tokens,
                            "valid":state!=0,"bank":bank})[0]
    data={"status":"random_weights_uncalibrated","expected_logits":logits.ravel().tolist(),
          "expected_tokens":tokens.ravel().tolist(), "expected_summaries":summaries.ravel().tolist(),
          "expected_questions":qs.ravel().tolist(), "expected_bank":bank.ravel().tolist()}
    for name,array in (("state",state),("questions",questions),("options",options)):
        data[name]=array.ravel().tolist()
        data[{"questions":"question","options":"option"}.get(name,name)+"_shape"]=list(array.shape)
    (artifact/"browser-fixture.json").write_text(json.dumps(data)+"\n")
    vendor=root/"web"/"vendor"
    vendor.mkdir(exist_ok=True)
    provenance=[]
    for name in ("ort.wasm.bundle.min.mjs","ort-wasm-simd-threaded.mjs","ort-wasm-simd-threaded.wasm"):
        path=vendor/name
        url=f"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/{name}"
        if args.runtime_source and (args.runtime_source/name).exists():
            if (args.runtime_source/name).resolve()!=path.resolve():
                shutil.copyfile(args.runtime_source/name,path)
        else:
            urllib.request.urlretrieve(url,path)
        provenance.append({"file":name,"upstream":url,"bytes":path.stat().st_size,
                           "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    if "ONNX Runtime Web v1.30.0" not in (vendor/"ort.wasm.bundle.min.mjs").read_text()[:200]:
        raise ValueError("unexpected runtime version")
    (artifact/"runtime-manifest.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps({"fixture":"artifacts/browser-fixture.json","runtime_bytes":sum(x["bytes"] for x in provenance)},indent=2))


if __name__ == "__main__":
    main()
