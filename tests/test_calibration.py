import numpy as np
import pytest
from dex.calibration import fit_temperature, metrics, nll


def test_temperature_recovers_synthetic_overconfidence_on_separate_test():
    rng=np.random.default_rng(12)
    z=rng.normal(size=(12000,4))*1.5
    p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    y=np.array([rng.choice(4,p=row) for row in p])
    temperature=fit_temperature(3*z[:6000],y[:6000])
    assert 2.7<temperature<3.3
    before=metrics(3*z[6000:],y[6000:])
    after=metrics(3*z[6000:],y[6000:],temperature)
    assert after['nll']<before['nll']
    assert after['accuracy']==before['accuracy']
    assert after['ece_equal_width_top_label']<.04


def test_rejects_unusable_calibration():
    with pytest.raises(ValueError):
        fit_temperature([],[])
    with pytest.raises(ValueError):
        nll([[1,2]],[2])
