"""Check cache/batch independence and dynamic shapes on the exported graphs."""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'artifacts/model-manifest.json').read_text())
cfg=manifest['config']
session=ort.SessionOptions()
session.intra_op_num_threads=4
session.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
encoder=ort.InferenceSession(str(root/'artifacts/encoder.int8.onnx'),sess_options=session)
core=ort.InferenceSession(str(root/'artifacts/core.fp32.onnx'),sess_options=session)
rng=np.random.default_rng(39)
ids=rng.integers(1,cfg['vocab'],size=(5,20),dtype=np.int64)
kinds=np.zeros(5,dtype=np.int64)
tokens,summaries=encoder.run(None,{'ids':ids,'kinds':kinds})
single=[encoder.run(None,{'ids':ids[i:i+1],'kinds':kinds[i:i+1]}) for i in range(5)]
single_tokens=np.concatenate([x[0] for x in single])
single_summaries=np.concatenate([x[1] for x in single])
rank=cfg['rank']
questions=rng.normal(size=(7,rank)).astype(np.float32)
bank=rng.normal(size=(37,rank)).astype(np.float32)
bank/=np.linalg.norm(bank,axis=1,keepdims=True)
def run(q,b=bank):
    return core.run(None,{'questions':q,'summaries':summaries,'tokens':tokens,
                          'valid':ids!=0,'bank':b})[0]
together=run(questions)
alone=np.concatenate([run(q[None]) for q in questions])
order=rng.permutation(len(bank))
permuted=run(questions,bank[order])
report={'status':'untrained_export_numerical_audit',
    'encoder_batch_max_abs_error':float(np.max(np.abs(tokens-single_tokens))),
    'summary_batch_max_abs_error':float(np.max(np.abs(summaries-single_summaries))),
    'field_batch_max_abs_error':float(np.max(np.abs(together-alone))),
    'option_permutation_max_abs_error':float(np.max(np.abs(permuted-together[:,order]))),
    'shapes':{'chunks':5,'length':20,'fields':7,'options':37},
    'tolerance':2e-5}
report['passed']=all(report[k]<=report['tolerance'] for k in report if k.endswith('max_abs_error'))
(root/'artifacts/export-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert report['passed']
