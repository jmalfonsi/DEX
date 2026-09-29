"""Post-hoc temperature fitting and diagnostics, on independent labelled data.

These utilities do not make an untrained model calibrated. A deployment needs a
versioned held-out calibration corpus and a separate test of the final policy.
"""
import numpy as np


def _validate(logits, labels):
    z=np.asarray(logits,dtype=np.float64)
    y=np.asarray(labels)
    if (z.ndim!=2 or min(z.shape)<1 or not np.isfinite(z).all() or
        y.shape!=(len(z),) or y.dtype.kind not in "iu" or
        np.any(y<0) or np.any(y>=z.shape[1])):
        raise ValueError("invalid logits or labels")
    return z,y


def nll(logits, labels, temperature=1.):
    z,y=_validate(logits,labels)
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError("invalid temperature")
    z=z/temperature
    return float(np.mean(np.logaddexp.reduce(z,axis=1)-z[np.arange(len(y)),y]))


def fit_temperature(logits, labels):
    z,y=_validate(logits,labels)
    # Deterministic bounded golden-section minimization over log-temperature.
    left,right=np.log(.05),np.log(20.)
    ratio=(np.sqrt(5)-1)/2
    a,b=right-ratio*(right-left),left+ratio*(right-left)
    fa,fb=nll(z,y,np.exp(a)),nll(z,y,np.exp(b))
    for _ in range(80):
        if fa<fb:
            right,b,fb=b,a,fa
            a=right-ratio*(right-left)
            fa=nll(z,y,np.exp(a))
        else:
            left,a,fa=a,b,fb
            b=left+ratio*(right-left)
            fb=nll(z,y,np.exp(b))
    candidate=float(np.exp((left+right)/2))
    # Include 1 explicitly to avoid a pathological optimizer worsening its own set.
    return candidate if nll(z,y,candidate)<nll(z,y,1.) else 1.


def metrics(logits, labels, temperature=1., bins=15):
    z,y=_validate(logits,labels)
    if not np.isfinite(temperature) or temperature<=0 or type(bins) is not int or bins<1:
        raise ValueError("invalid metrics settings")
    z=z/temperature
    p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    guess=p.argmax(1);conf=p[np.arange(len(y)),guess];correct=guess==y
    target=np.zeros_like(p);target[np.arange(len(y)),y]=1
    bucket=np.minimum((conf*bins).astype(int),bins-1)
    reliability=[];ece=0.
    for i in range(bins):
        selected=bucket==i
        count=int(selected.sum())
        if count:
            accuracy,confidence=float(correct[selected].mean()),float(conf[selected].mean())
            ece+=count/len(y)*abs(accuracy-confidence)
            reliability.append({"bin":i,"n":count,"accuracy":accuracy,"confidence":confidence})
    return {"n":len(y),"accuracy":float(correct.mean()),"nll":nll(logits,y,temperature),
        "brier_sum_over_classes":float(((p-target)**2).sum(1).mean()),
        "ece_equal_width_top_label":float(ece),"bins":bins,"reliability":reliability}

