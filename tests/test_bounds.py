import itertools
import numpy as np
import pytest

from dex.bounds import softmax, probability_bounds, event_bounds, typed_views
from dex.refine import ResidualDecision


def test_bounds_contain_all_vertices_and_are_tight():
    rng = np.random.default_rng(28)
    for n in (1, 2, 4, 6):
        for _ in range(12):
            low = rng.normal(size=n) * 7
            high = low + rng.random(n) * 5
            vertices = np.array([softmax(np.where(bits, high, low))
                                 for bits in itertools.product([False, True], repeat=n)])
            p_low, p_high = probability_bounds(low, high)
            np.testing.assert_allclose(p_low, vertices.min(0), atol=1e-13)
            np.testing.assert_allclose(p_high, vertices.max(0), atol=1e-13)
            members = list(range(0, n, 2))
            bounds = event_bounds(low, high, members)
            mass = vertices[:, members].sum(1)
            np.testing.assert_allclose(bounds, (mass.min(), mass.max()), atol=1e-13)


def test_numerical_extremes_and_dominant_class():
    for logits in ([1000, -1000, 0], [1e6, 1e6-1, -1e6], [-1e6, -1e6], [0]):
        lo, hi = probability_bounds(logits, logits)
        np.testing.assert_allclose(lo, softmax(logits), atol=1e-10)
        np.testing.assert_allclose(hi, softmax(logits), atol=1e-10)


def fixture(seed, c=11, n=37, r=8):
    rng = np.random.default_rng(seed)
    bank = rng.normal(size=(n, r))
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    delta = np.tanh(rng.normal(size=(c, r))) / np.sqrt(r)
    weights, correction = softmax(rng.normal(size=c)), np.tanh(rng.normal(size=n))
    coarse = rng.normal(size=r)
    ref = ResidualDecision(coarse, weights, bank, lambda ids: delta[ids], lambda ids: correction[ids])
    exact = softmax(8 * bank @ (coarse + weights @ delta) + correction)
    return ref, exact


def test_every_intermediate_envelope_contains_full_model():
    for seed in range(12):
        ref, exact = fixture(seed)
        previous_width = np.inf
        for rounds in range(19):
            out = ref.run(tolerance=0, max_rounds=rounds, batch_size=3)
            low, high = probability_bounds(out.logit_lower, out.logit_upper)
            assert np.all(low <= exact + 1e-10)
            assert np.all(high >= exact - 1e-10)
            width = np.max(high - low)
            assert width <= previous_width + 1e-10
            previous_width = width
            if out.argmax_certified:
                assert out.value == int(np.argmax(exact))
        np.testing.assert_allclose(out.probabilities, exact, atol=1e-10)


def test_adaptive_stopping_meets_probability_error():
    for seed in range(20):
        ref, exact = fixture(seed)
        out = ref.run(tolerance=.01)
        assert out.argmax_certified and out.tolerance_met
        assert out.value == np.argmax(exact)
        assert abs(out.probability - exact[out.value]) <= .01 + 1e-10


def test_uniform_distribution_does_not_invent_confidence():
    n = 10000
    ref = ResidualDecision(np.zeros(2), [1], np.zeros((n, 2)),
                           lambda ids: np.zeros((len(ids), 2)), lambda ids: np.zeros(len(ids)),
                           rho=0, option_bound=0)
    out = ref.run()
    assert out.probability == pytest.approx(1/n)
    assert out.argmax_certified
    assert not out.calibrated


def test_coarse_loser_can_become_winner():
    # A fixed top-1 filter misses option 1, our option bounds do not.
    ref = ResidualDecision([1.0], [1.0], [[1.0], [.95]],
                           lambda ids: np.zeros((len(ids), 1)),
                           lambda ids: np.array([-.9, .9])[ids], rho=0)
    out = ref.run(tolerance=1e-7, batch_size=1)
    assert out.value == 1
    assert out.options_refined == 2


def test_contract_violations_and_bad_inputs_fail():
    with pytest.raises(ValueError):
        probability_bounds([1], [0])
    with pytest.raises(ValueError):
        ResidualDecision([0], [.9], [[1]], lambda _: None, lambda _: None)
    ref = ResidualDecision([0], [1], [[1], [-1]],
                           lambda ids: np.full((len(ids), 1), 1.00000005), lambda ids: np.zeros(len(ids)))
    with pytest.raises(ValueError, match="norm contract"):
        ref.run(tolerance=0)
    with pytest.raises(ValueError):
        event_bounds([0, 0], [1, 1], [-1])


def test_views_share_probability_measure():
    p = [.2, .3, .5]
    out = typed_views(p, ["a", "b", "c"], positive=["a", "b"], levels=[0, 1, 10])
    assert out["boolean"]["p_true"] + out["boolean"]["p_false"] == 1
    assert out["score"]["value"] == 10
    assert out["score"]["probability"] == .5
    assert out["score"]["expectation"] == 5.3
    assert event_bounds(np.log(p), np.log(p), [0, 1]) == pytest.approx((.5, .5))

