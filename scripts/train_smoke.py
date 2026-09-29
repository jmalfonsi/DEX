"""Tiny CPU training smoke test; demonstrates optimization, not language quality."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dex.model import Config,DEX,training_loss

torch.set_num_threads(4)
torch.manual_seed(29)
model=DEX(Config(vocab=96,width=32,heads=4,ff=64,depth=1,rank=16,window=16))
state=torch.randint(1,96,(3,12))
questions=torch.randint(1,96,(4,8))
options=torch.randint(1,96,(5,8))
targets=torch.tensor([0,1,2,3])
optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
history=[]
for step in range(80):
    optimizer.zero_grad()
    loss=training_loss(model(state,questions,options),targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    history.append(float(loss.detach()))
report={"status":"single_batch_overfit_smoke_only","steps":80,"initial_loss":history[0],
        "final_loss":history[-1],"no_semantic_or_generalization_claim":True}
assert history[-1]<history[0]*.25
path=Path(__file__).resolve().parents[1]/"artifacts"/"training-smoke.json"
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
