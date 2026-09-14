"""Generalized EM for the conditional likelihood P(observations | constraints)."""

from __future__ import annotations

import copy
import math
import warnings
from collections import Counter

import torch

from conin.exceptions import InvalidInputError

from ..mvr_common import (
    ACCUM_DTYPE,
    _build_static_context,
    _build_sumprod_ctx,
    _forward_messages,
    _hmm_to_torch,
    _model_parts,
    _resolve_horizon,
)
from .baum_welch_mvr import _e_step_counts, _resolve_time_horizons, _support_masks


def _parameter_context(ctx, log_params):
    """Attach current parameters to a context with fixed observation indices."""
    start, transition, emission = log_params
    weights = torch.zeros_like(ctx["log_emit_weights"])
    if ctx["observed_index"]:
        times, observed = zip(*ctx["observed_index"].items())
        weights[list(times)] = emission[:, list(observed)].T
    return dict(
        ctx,
        log_start_vec=start,
        transition_mat=transition.exp(),
        log_transition_mat=transition,
        log_transition_t=transition.T.contiguous(),
        log_emit_weights=weights,
    )


def _constraint_statistics(log_params, contexts, multiplicities, *, counts=False):
    """Sum constraint-only statistics with one pass per distinct horizon."""
    total = [torch.zeros_like(p, dtype=ACCUM_DTYPE) for p in log_params[:2]]
    normalizer = torch.zeros((), dtype=ACCUM_DTYPE, device=log_params[0].device)
    for horizon, base in contexts.items():
        ctx = _parameter_context(base, log_params)
        if counts:
            initial, transition, _, score = _e_step_counts(ctx)
            total[0] += multiplicities[horizon] * initial
            total[1] += multiplicities[horizon] * transition
        else:
            alpha, shift = _forward_messages(ctx)
            score = shift + torch.logsumexp(alpha[-1].reshape(-1), dim=0)
        normalizer += multiplicities[horizon] * score
    return total, normalizer


def _surrogate(log_params, counts, normalizer):
    """Expected complete-data log likelihood minus the constraint normalizer."""
    return (
        sum((p[c > 0] * c[c > 0]).sum() for p, c in zip(log_params, counts))
        - normalizer
    )


def _chain_gradient(log_params, data_counts, prior_counts, update):
    """Gradient of the conditional surrogate in start and transition logits."""
    gradients = []
    for name, p, data, prior in zip(
        ("start", "transition"), log_params, data_counts, prior_counts
    ):
        difference = data - prior
        gradient = difference - p.exp() * difference.sum(dim=-1, keepdim=True)
        gradients.append(gradient.to(p.dtype) if name in update else torch.zeros_like(p))
    return gradients


def generalized_em_mvr_chmm(
    model,
    observations,
    *,
    time_horizons=None,
    max_iter=50,
    tol=1e-6,
    update=("start", "transition", "emission"),
    dtype=torch.float64,
    device="cpu",
    verbose=False,
    inner_max_iter=10,
    inner_tol=1e-6,
    step_size=1.0,
    max_backtracks=30,
):
    """Fit ``sum log P(observations | constraints)`` with constrained GEM.

    Accepts the same batch of dense lists or partial ``{time: label}`` maps as
    Baum–Welch; maps require explicit ``time_horizons``. Missing times still
    drive transitions and constraints. Fits a copy and returns ``(hmm, history)``
    with Baum–Welch's start-of-iteration history convention: on budget exhaustion
    the returned model is one update ahead of the last score.

    Initial zeros are structural and warn. Enabled emission rows use exact count
    normalization without pseudocounts; chain parameters use backtracked logit
    steps. Inner tolerances apply to the batch-mean surrogate. Failed backtracking
    warns and returns the last accepted parameters, with their score recorded.
    """
    update = tuple(update)
    unknown = set(update) - {"start", "transition", "emission"}
    if unknown:
        raise InvalidInputError(f"Unknown update targets: {sorted(unknown)}")
    if (
        any(
            not isinstance(v, int) or v < minimum
            for v, minimum in (
                (max_iter, 0), (inner_max_iter, 1), (max_backtracks, 1)
            )
        )
        or not math.isfinite(step_size)
        or step_size <= 0
        or not math.isfinite(inner_tol)
        or inner_tol < 0
        or not math.isfinite(tol)
    ):
        raise InvalidInputError(
            "GEM iteration budgets, step_size, and tolerances must be valid."
        )
    observations = list(observations)
    if not observations:
        raise InvalidInputError("observations must contain at least one sequence.")
    horizons = _resolve_time_horizons(observations, time_horizons)
    working = copy.deepcopy(model)
    hmm, constraints = _model_parts(working)
    _support_masks(hmm)
    log_params = list(_hmm_to_torch(hmm, log=True, dtype=dtype, device=device))
    contexts, prior_contexts, resolved = [], {}, []
    for index, (observed, horizon) in enumerate(zip(observations, horizons)):
        try:
            T, _ = _resolve_horizon(observed, horizon)
            if T not in prior_contexts:
                static = _build_static_context(
                    constraints, hmm.num_hidden_states, T,
                    log=True, dtype=dtype, device=device,
                )
                prior_contexts[T] = _build_sumprod_ctx(
                    working, {}, time_horizon=T, dtype=dtype, device=device, static=static
                )
            contexts.append(
                _build_sumprod_ctx(
                    working, observed, time_horizon=T, dtype=dtype, device=device,
                    static=prior_contexts[T],
                )
            )
            resolved.append(T)
        except InvalidInputError as exc:
            raise InvalidInputError(f"Sequence {index}: {exc}") from exc
    multiplicities = Counter(resolved)
    n = len(observations)
    history = []
    converged = failed = False
    for iteration in range(max_iter + 1):
        if iteration == max_iter and not failed:
            break
        data_counts = [torch.zeros_like(p, dtype=ACCUM_DTYPE) for p in log_params]
        joint = 0.0
        for index, ctx in enumerate(contexts):
            try:
                counts = _e_step_counts(_parameter_context(ctx, log_params))
            except InvalidInputError as exc:
                raise InvalidInputError(f"Sequence {index}: {exc}") from exc
            for dest, source in zip(data_counts, counts[:3]):
                dest += source
            joint += counts[3]
        _, normalizer = _constraint_statistics(log_params, prior_contexts, multiplicities)
        likelihood = joint - float(normalizer)
        if history and likelihood < history[-1] - 1e-6 * max(1.0, abs(history[-1])):
            warnings.warn(
                "GEM conditional log-likelihood decreased; try dtype=torch.float64.",
                UserWarning,
                stacklevel=2,
            )
        history.append(likelihood)
        if verbose:
            print(f"GEM iter {iteration:3d}  loglik = {likelihood:.10f}")
        if failed:
            break
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break
        data_counts = [c / n for c in data_counts]
        if "emission" in update:
            totals = data_counts[2].sum(dim=-1, keepdim=True)
            log_params[2] = torch.where(
                totals > 0, data_counts[2].log() - totals.log(), log_params[2]
            ).to(dtype)
        for _ in range(inner_max_iter):
            prior_counts, normalizer = _constraint_statistics(
                log_params, prior_contexts, multiplicities, counts=True
            )
            value = _surrogate(log_params, data_counts, normalizer / n)
            gradient = _chain_gradient(
                log_params, data_counts, [c / n for c in prior_counts], update
            )
            if max(float(g.abs().max()) for g in gradient) <= inner_tol:
                break
            norm_sq = sum(g.to(ACCUM_DTYPE).square().sum() for g in gradient)
            step = step_size
            for _ in range(max_backtracks):
                candidate = [
                    torch.log_softmax(p + step * g, dim=-1) if name in update else p
                    for name, p, g in zip(("start", "transition"), log_params, gradient)
                ] + [log_params[2]]
                try:
                    _, candidate_z = _constraint_statistics(
                        candidate, prior_contexts, multiplicities
                    )
                    candidate_value = _surrogate(candidate, data_counts, candidate_z / n)
                except InvalidInputError:
                    candidate_value = value.new_tensor(-torch.inf)
                if (
                    torch.isfinite(candidate_value)
                    and candidate_value >= value + 1e-4 * step * norm_sq
                ):
                    break
                step *= 0.5
            else:
                warnings.warn(
                    "GEM backtracking failed; retaining last accepted parameters.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                failed = True
                break
            log_params = candidate
            if float(candidate_value - value) <= inner_tol * max(1.0, abs(float(value))):
                break
    if tol > 0 and not converged and not failed and len(history) > 1:
        warnings.warn(
            f"GEM stopped at max_iter={max_iter} without reaching tol={tol}; "
            "the returned model has had one more update than history records.",
            UserWarning,
            stacklevel=2,
        )
    if max_iter:
        for name, attr, p in zip(
            ("start", "transition", "emission"),
            ("start_vec", "transition_mat", "emission_mat"),
            log_params,
        ):
            if name in update:
                setattr(hmm, attr, p.exp().cpu().tolist())
        hmm.initialize(avoid_reinitialization=False)
    return hmm, history
