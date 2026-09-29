"""Bounds relative to the full model, NOT bounds on correctness in the world.

The formulas are valid over the reals. NumPy/JS evaluations use floating point;
the current reference is not a formally verified interval-arithmetic library.
"""
from __future__ import annotations

import numpy as np


def _vector(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("expected a non-empty finite vector")
    return x


def softmax(x):
    x = _vector(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return -np.inf
    return np.logaddexp.reduce(x)


def _excluding_logsumexp(x):
    # Prefix/suffix avoids total - exp(x_i) cancellation for a dominant class.
    left = np.concatenate(([-np.inf], np.logaddexp.accumulate(x)[:-1]))
    right = np.concatenate((np.logaddexp.accumulate(x[::-1])[::-1][1:], [-np.inf]))
    return np.logaddexp(left, right)


def probability_bounds(lower, upper):
    lower, upper = _vector(lower), _vector(upper)
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("invalid logit intervals")
    lo = np.exp(lower - np.logaddexp(lower, _excluding_logsumexp(upper)))
    hi = np.exp(upper - np.logaddexp(upper, _excluding_logsumexp(lower)))
    return lo, hi


def event_bounds(lower, upper, members):
    """Tight probability bounds over an interval box for an event/subset."""
    lower, upper = _vector(lower), _vector(upper)
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("invalid logit intervals")
    ids = list(members)
    if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 or i >= len(lower) for i in ids):
        raise ValueError("event must contain distinct valid integer indices")
    mask = np.zeros(len(lower), dtype=bool)
    mask[ids] = True
    if not mask.any():
        return 0.0, 0.0
    if mask.all():
        return 1.0, 1.0
    a, b = logsumexp(lower[mask]), logsumexp(upper[~mask])
    c, d = logsumexp(upper[mask]), logsumexp(lower[~mask])
    return float(np.exp(a - np.logaddexp(a, b))), float(np.exp(c - np.logaddexp(c, d)))


def typed_views(probabilities, option_ids, *, positive=None, levels=None):
    """Views of ONE explicitly declared categorical variable, no hidden independence.

    A score's expectation has no point probability. Report the modal level and
    its probability separately. Boolean false is the declared complement.
    """
    p = _vector(probabilities)
    if len(option_ids) != len(p) or len(set(option_ids)) != len(p):
        raise ValueError("option ids must be unique and match probabilities")
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("probabilities must sum to one")
    best = int(np.argmax(p))
    result = {"choice": {"value": option_ids[best], "probability": float(p[best])}}
    if positive is not None:
        if len(set(positive)) != len(positive) or not set(positive).issubset(option_ids):
            raise ValueError("invalid event")
        yes = sum(float(p[i]) for i, name in enumerate(option_ids) if name in positive)
        result["boolean"] = {"value": yes >= .5, "probability": max(yes, 1 - yes),
                             "p_true": yes, "p_false": 1 - yes}
    if levels is not None:
        levels = _vector(levels)
        if len(levels) != len(p) or np.any(np.diff(levels) <= 0):
            raise ValueError("levels must be strictly increasing")
        result["score"] = {"value": float(levels[best]), "probability": float(p[best]),
                           "expectation": float(p @ levels)}
    return result

