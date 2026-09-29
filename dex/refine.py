"""Lazy evaluation of a single fixed bounded-residual decision function.

The two callbacks may perform neural inference, and are called only for selected
indices. Coarse state summaries and all coarse option embeddings are already read.
This is a correctness reference, not an optimized GPU scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .bounds import probability_bounds, softmax


@dataclass
class Result:
    value: int
    probability: float
    probability_interval: tuple[float, float]
    probabilities: np.ndarray
    logit_lower: np.ndarray
    logit_upper: np.ndarray
    argmax_certified: bool
    tolerance_met: bool
    chunks_refined: int
    options_refined: int
    rounds: int
    calibrated: bool = False


class ResidualDecision:
    def __init__(self, coarse, weights, bank, state_refiner: Callable,
                 option_refiner: Callable, *, rho=1.0, option_bound=1.0,
                 scale=8.0, temperature=1.0, numerical_slack=1e-7):
        self.coarse = np.asarray(coarse, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bank = np.asarray(bank, dtype=np.float64)
        if (self.coarse.ndim != 1 or self.coarse.size == 0 or
            self.bank.ndim != 2 or self.bank.shape[0] == 0 or
            self.bank.shape[1] != self.coarse.size or self.weights.ndim != 1 or
            self.weights.size == 0):
            raise ValueError("invalid shapes")
        if not all(np.isfinite(v).all() for v in (self.coarse, self.weights, self.bank)):
            raise ValueError("nonfinite inputs")
        if (np.any(self.weights < 0) or
            not np.isclose(self.weights.sum(), 1, rtol=0, atol=1e-10)):
            raise ValueError("routing weights must form a distribution")
        if (not np.isfinite([rho, option_bound, scale, temperature, numerical_slack]).all()
            or min(rho, option_bound, numerical_slack) < 0 or min(scale, temperature) <= 0):
            raise ValueError("invalid bounds or temperature")
        self.rho, self.option_bound = rho, option_bound
        self.scale, self.temperature, self.slack = scale, temperature, numerical_slack
        self.state_refiner, self.option_refiner = state_refiner, option_refiner

    def run(self, *, tolerance=.01, batch_size=8, max_rounds=None):
        if not np.isfinite(tolerance) or tolerance < 0 or batch_size < 1:
            raise ValueError("invalid scheduler settings")
        if max_rounds is not None and (type(max_rounds) is not int or max_rounds < 0):
            raise ValueError("invalid round budget")
        c, n = self.weights.size, self.bank.shape[0]
        state_seen, option_seen = np.zeros(c, bool), np.zeros(n, bool)
        if self.rho == 0:
            state_seen[:] = True
        if self.option_bound == 0:
            option_seen[:] = True
        latent = self.coarse.copy()
        corrections = np.zeros(n)
        norms = np.linalg.norm(self.bank, axis=1)
        previous_lower, previous_upper = np.full(n, -np.inf), np.full(n, np.inf)
        rounds, state_count, option_count = 0, 0, 0
        while True:
            logits = (self.scale * (self.bank @ latent) + corrections) / self.temperature
            # Sum remaining weights directly: avoid catastrophic subtraction 1-sum(seen).
            tail = self.weights[~state_seen].sum()
            state_radius = self.scale * self.rho * tail * norms
            radius = (state_radius + self.option_bound * (~option_seen)) / self.temperature
            # A practical numerical guard, NOT a formal floating-point proof.
            guard = self.slack if (not state_seen.all() or not option_seen.all()) else 0.0
            lower = np.maximum(previous_lower, logits - radius - guard)
            upper = np.minimum(previous_upper, logits + radius + guard)
            if np.any(lower > upper + 1e-10):
                raise ArithmeticError("inconsistent refinement envelope")
            lower = np.minimum(lower, upper)
            p_low, p_high = probability_bounds(lower, upper)
            candidate = int(np.argmax(lower))
            other = np.delete(upper, candidate)
            certified = not other.size or lower[candidate] >= np.max(other)
            met = certified and (p_high[candidate] - p_low[candidate] <= tolerance)
            full = state_seen.all() and option_seen.all()
            if met or full or (max_rounds is not None and rounds >= max_rounds):
                # Midpoint of the *intersected* box defines one normalized estimate.
                # Its selected probability lies inside the returned interval.
                p = softmax((lower + upper) / 2)
                if not certified:
                    candidate = int(np.argmax(p))
                return Result(candidate, float(p[candidate]),
                              (float(p_low[candidate]), float(p_high[candidate])), p,
                              lower, upper, bool(certified), bool(met), state_count,
                              option_count, rounds)
            previous_lower, previous_upper = lower, upper
            # Remove the larger source of uncertainty; finite worst-case termination.
            do_state = (not state_seen.all() and
                        (option_seen.all() or state_radius[candidate] >= self.option_bound))
            if do_state:
                pending = np.flatnonzero(~state_seen)
                ids = pending[np.argsort(-self.weights[pending], kind="stable")[:batch_size]]
                values = np.asarray(self.state_refiner(ids), dtype=np.float64)
                if (values.shape != (len(ids), len(latent)) or not np.isfinite(values).all()
                    or np.any(np.linalg.norm(values, axis=1) > self.rho)):
                    raise ValueError("state refiner violated its norm contract")
                latent += self.weights[ids] @ values
                state_seen[ids] = True
                state_count += len(ids)
            else:
                pending = np.flatnonzero(~option_seen)
                ids = pending[np.argsort(-upper[pending], kind="stable")[:batch_size]]
                values = np.asarray(self.option_refiner(ids), dtype=np.float64)
                if (values.shape != (len(ids),) or not np.isfinite(values).all()
                    or np.any(np.abs(values) > self.option_bound)):
                    raise ValueError("option refiner violated its scalar contract")
                corrections[ids] = values
                option_seen[ids] = True
                option_count += len(ids)
            rounds += 1

