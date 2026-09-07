"""
Shotgun stochastic search over the space of active MVR constraint sets.

Treats the constraint set itself as the unknown: given a pool of candidate MVRs and
one observed sequence drawn from an unknown constrained distribution, search for the
subset that best explains the data. Scores are exact Bayes factors, read off the
satisfaction weights that ``sat_prob_mvr`` already computes.
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import torch
from munch import Munch

from conin.exceptions import InvalidInputError

from .chmm_mvr import MVR_CHMM
from .learning.baum_welch_mvr import forward_backward_mvr_chmm
from .mvr_common import _build_sumprod_ctx, _model_parts, _resolve_horizon
from .other_queries.sat_prob_mvr import sat_prob_torch_mvr_chmm
from .other_queries.sat_time_mvr import _resolve_target

# ======================================================================
# Scoring
# ======================================================================


def _reject_misconfiguration(model, observed, T, dtype, device):
    """Raise for a model that is invalid rather than merely infeasible.

    A window past the horizon and an unknown observed label raise the same class as an
    infeasible lattice, but only the latter is an answer. Building the context runs
    every such check up front, where the two are still distinguishable.
    """
    _build_sumprod_ctx(model, observed, time_horizon=T, dtype=dtype, device=device)


def _conditional(joint, prior):
    """``log P(observed | event)``, or ``-inf`` for an event carrying no prior mass."""
    prior = float(prior)

    # Both terms vanish for a constraint set nothing satisfies. The conditional is then
    # undefined rather than small, and -inf is what keeps it out of a search.
    return float(joint) - prior if math.isfinite(prior) else -math.inf


def _evidence_pair(model, observed, target, T, dtype, device):
    """``(log P(D | rest), log P(D | rest + target))`` from one pair of sat_prob calls.

    The two satisfaction branches partition the lattice, so their total is the evidence
    of the model without the target and the satisfied branch alone is the evidence with
    it. Running the pair a second time over an empty observation set supplies the
    matching normalizers, which is what turns a joint into a conditional.
    """
    options = dict(
        target=target,
        time_horizon=T,
        dtype=dtype,
        device=device,
        return_log_weights=True,
    )

    _, with_data = sat_prob_torch_mvr_chmm(model, observed, **options)
    _, without_data = sat_prob_torch_mvr_chmm(model, {}, **options)

    rest = _conditional(
        torch.logsumexp(with_data, dim=0), torch.logsumexp(without_data, dim=0)
    )

    return rest, _conditional(with_data[0], without_data[0])


def log_evidence_mvr_chmm(
    model, observed, *, time_horizon=None, dtype=torch.float64, device="cpu"
):
    """Log evidence of a constraint set, ``log P(observed | constraints satisfied)``.

    Evaluates ``log P(observed, constraints satisfied) - log P(constraints satisfied)``
    with two forward passes, the second over an empty observation set. Returns ``-inf``
    for a set the data cannot satisfy rather than raising, so an infeasible model can
    be scored alongside feasible ones. A *misconfigured* model still raises: a window
    past the horizon or an unknown label is not the same thing as an infeasible one.
    """
    T, _ = _resolve_horizon(observed, time_horizon)
    _reject_misconfiguration(model, observed, T, dtype, device)

    try:
        joint = forward_backward_mvr_chmm(
            model, observed, time_horizon=T, dtype=dtype, device=device
        )[2]
        prior = forward_backward_mvr_chmm(
            model, {}, time_horizon=T, dtype=dtype, device=device
        )[2]
    except InvalidInputError:
        return -math.inf

    return joint - prior


def bayes_factor_mvr_chmm(
    model, observed, *, target, time_horizon=None, dtype=torch.float64, device="cpu"
):
    """Log Bayes factor for including the target constraint, given all the others.

    Returns ``log P(C | observed, rest) - log P(C | rest)``, which equals
    ``log P(observed | rest + C) - log P(observed | rest)`` exactly. ``target`` selects
    ``C`` by index or name just as in ``sat_prob_torch_mvr_chmm``; every other
    constraint on the model forms ``rest``.

    A target that cannot hold under ``rest`` scores ``-inf`` rather than the ``0`` of
    the note this implements: that model has no feasible path at all, so a search must
    never enter it. A misconfigured model raises, as in ``log_evidence_mvr_chmm``.
    """
    T, _ = _resolve_horizon(observed, time_horizon)
    index = _resolve_target(_model_parts(model)[1], target)
    _reject_misconfiguration(model, observed, T, dtype, device)

    try:
        rest, full = _evidence_pair(model, observed, index, T, dtype, device)
    except InvalidInputError:
        return -math.inf

    if not math.isfinite(rest):
        return -math.inf

    return full - rest


# ======================================================================
# Shotgun stochastic search
# ======================================================================


def _log_prior(size, num_candidates, inclusion_prob):
    """Log Bernoulli inclusion prior for a constraint set of the given size."""
    return size * math.log(inclusion_prob) + (num_candidates - size) * math.log(
        1.0 - inclusion_prob
    )


def _sample(rng, items, deltas):
    """One member of ``items`` drawn with probability proportional to ``exp(delta)``."""
    if not items:
        return None

    weights = np.array([deltas[i] for i in items], dtype=float)
    finite = np.isfinite(weights)

    if not finite.any():
        return None

    probs = np.exp(weights - weights[finite].max())

    return items[rng.choice(len(items), p=probs / probs.sum())]


def _validate(candidates, num_iterations, inclusion_prob, elite_size):
    """Check the search arguments, one message per rejected argument."""
    if not candidates:
        raise InvalidInputError("Shotgun search needs at least one candidate MVR.")

    if num_iterations < 1:
        raise InvalidInputError(
            f"num_iterations must be a positive integer, got {num_iterations!r}."
        )

    if not 0.0 < inclusion_prob < 1.0:
        raise InvalidInputError(
            f"inclusion_prob must lie strictly between 0 and 1, got {inclusion_prob!r}."
        )

    if elite_size < 1:
        raise InvalidInputError(
            f"elite_size must be a positive integer, got {elite_size!r}."
        )


def sss_torch_mvr_chmm(
    hidden_markov_model,
    candidates,
    observed,
    *,
    time_horizon=None,
    num_iterations=50,
    inclusion_prob=0.5,
    initial=None,
    elite_size=100,
    rng=None,
    dtype=torch.float64,
    device="cpu",
):
    """Search for the active constraint set behind an observed sequence.

    Runs Shotgun Stochastic Search (Hans, Dobra and West, 2007) over subsets of
    ``candidates``, scoring each model by ``log P(observed | A) + log p(A)`` under a
    Bernoulli(``inclusion_prob``) inclusion prior. The neighborhood of the current set
    is every one-constraint addition and deletion; each iteration samples one
    representative from each of those two sets in proportion to score, then samples
    between them. Every model scored on the way enters the elite set, so the result
    ranks far more models than the chain itself visits.

    Returns a ``Munch`` carrying ``best`` and ``best_score``, an ``elite`` list of
    ``(model, score)`` sorted by score, and a per-iteration ``trace``. A model is a
    ``frozenset`` of indices into ``candidates``. Raises if the starting model is
    infeasible, since every score is measured relative to it.
    """
    candidates = list(candidates)
    _validate(candidates, num_iterations, inclusion_prob, elite_size)

    rng = np.random.default_rng() if rng is None else rng
    T, _ = _resolve_horizon(observed, time_horizon)

    # One construction aligns every candidate to the HMM's hidden-state ordering; the
    # subset models below are shallow copies, so they inherit that alignment.
    pool = MVR_CHMM(
        hidden_markov_model=hidden_markov_model, constraints=list(candidates)
    )
    aligned = list(pool.constraints)
    num_candidates = len(aligned)

    def model_for(subset):
        model = copy.copy(pool)
        model.constraints = [aligned[i] for i in subset]
        return model

    _reject_misconfiguration(pool, observed, T, dtype, device)

    current = frozenset(initial or ())

    if any(i not in range(num_candidates) for i in current):
        raise InvalidInputError(
            f"initial={sorted(current)} is out of range for "
            f"{num_candidates} candidates."
        )

    start = log_evidence_mvr_chmm(
        model_for(sorted(current)), observed, time_horizon=T, dtype=dtype, device=device
    )

    if not math.isfinite(start):
        raise InvalidInputError(
            f"The starting model {sorted(current)} is infeasible for these "
            "observations, so no move from it can be scored."
        )

    prior_odds = math.log(inclusion_prob) - math.log(1.0 - inclusion_prob)
    evaluated = {}

    def evaluate(base, candidate):
        """``(factor, evidence without candidate, evidence with it)``, memoized."""
        key = (base, candidate)

        if key not in evaluated:
            subset = sorted(base | {candidate})

            try:
                rest, full = _evidence_pair(
                    model_for(subset),
                    observed,
                    subset.index(candidate),
                    T,
                    dtype,
                    device,
                )
            except InvalidInputError:
                rest, full = -math.inf, -math.inf

            factor = full - rest if math.isfinite(rest) else -math.inf
            evaluated[key] = (factor, rest, full)

        return evaluated[key]

    elite = {}
    trace = []

    for iteration in range(num_iterations):
        deltas = {}
        anchor = None

        for i in range(num_candidates):
            deleting = i in current
            base = current - {i} if deleting else current

            factor, rest, full = evaluate(base, i)

            # A move we cannot score is a move we do not take, in either direction.
            deltas[i] = (
                -math.inf
                if not math.isfinite(factor)
                else (-(factor + prior_odds) if deleting else factor + prior_odds)
            )

            # The current model's own evidence falls out of either move: a deletion
            # holds it in the satisfied branch, an addition in the branch total.
            if anchor is None and math.isfinite(full if deleting else rest):
                anchor = full if deleting else rest

        score = anchor + _log_prior(len(current), num_candidates, inclusion_prob)
        trace.append(Munch(iteration=iteration, model=current, score=score))

        # The paper records the whole neighborhood, not just the sampled move.
        elite[current] = max(elite.get(current, -math.inf), score)

        for i, delta in deltas.items():
            if not math.isfinite(delta):
                continue

            neighbor = current - {i} if i in current else current | {i}
            elite[neighbor] = max(elite.get(neighbor, -math.inf), score + delta)

        finalists = [
            pick
            for pick in (
                _sample(rng, sorted(set(range(num_candidates)) - current), deltas),
                _sample(rng, sorted(current), deltas),
            )
            if pick is not None
        ]

        if not finalists:
            warnings.warn(
                f"No feasible move from the model at iteration {iteration}; "
                "stopping early.",
                UserWarning,
            )
            break

        move = _sample(rng, finalists, deltas)
        current = current - {move} if move in current else current | {move}

    ranked = sorted(elite.items(), key=lambda item: item[1], reverse=True)[:elite_size]

    return Munch(
        best=ranked[0][0],
        best_score=ranked[0][1],
        elite=ranked,
        trace=trace,
    )
