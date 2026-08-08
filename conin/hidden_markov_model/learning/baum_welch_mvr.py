"""
Baum-Welch / EM for an MVR-constrained hidden Markov model.

The generative model is the **unconstrained** HMM; feasibility is an observed
fact about each realized hidden path. The complete-data likelihood is therefore
``P(x, y | theta) * 1[x feasible]`` Hence, maximizing ``E_q[log P(x, y | theta)]``
is the ordinary exact, closed-form, monotone EM on

    log P(y, constraints satisfied | theta)
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import torch

from conin.exceptions import InvalidInputError

from ..mvr_common import (
    ACCUM_DTYPE,
    _build_static_context,
    _build_sumprod_ctx,
    _dest_at,
    _initial_sumprod_message,
    _model_parts,
    _resolve_horizon,
    _sum_step,
    _sum_step_backward,
)

# ======================================================================
# Forward-backward
# ======================================================================


def _forward_messages(ctx):
    """
    Log-space forward recursion, returning every message and the accumulated
    shift. Messages are a list because the augmented shape varies with ``t``.
    """
    log_P = _initial_sumprod_message(ctx)

    log_alpha = []
    log_norm = torch.zeros((), dtype=ACCUM_DTYPE, device=ctx["device"])

    for t in range(ctx["T"]):
        if t:
            log_P = _sum_step(ctx, log_P, t)

        shift = log_P.max()

        # Must run before the subtraction: -inf - -inf is NaN.
        if not torch.isfinite(shift):
            raise InvalidInputError(f"No feasible augmented path at time {t}.")

        log_norm = log_norm + shift
        log_P = log_P - shift

        log_alpha.append(log_P[0])

    return log_alpha, log_norm


def _normalized_exp(log_values):
    """
    Exponentiate log-weights, normalized to sum to ``1``. The total is always
    finite: ``_forward_messages`` raises unless a complete feasible path exists.
    """
    total = torch.logsumexp(log_values.reshape(-1), dim=0)

    return (log_values - total).exp()


def _transition_posterior(ctx, log_alpha_prev, log_beta_tilde, t):
    """
    Posterior over ``(h_prev, h_curr)`` across ``t-1 -> t``, mediation summed out.
    ``upd`` is deterministic, so ``m_curr`` is a function of ``m_prev`` and there
    is no sum over it -- hence a gather rather than an augmented outer product.
    """
    K = ctx["K"]

    # log_beta_tilde is [h_curr, m_curr]; dest is [h_curr, m_prev] -> m_curr.
    gathered = log_beta_tilde.gather(1, _dest_at(ctx, t))

    # (h_prev, h_curr, m_prev), contracted down to (h_prev, h_curr).
    joint = log_alpha_prev.unsqueeze(1) + gathered.unsqueeze(0)
    log_xi = torch.logsumexp(joint, dim=2)

    log_xi = log_xi + ctx["log_transition_mat"]
    log_xi = log_xi + ctx["log_emit_weights"][t].reshape(1, K)

    return _normalized_exp(log_xi)


def _forward_backward(ctx, *, want_xi=False, want_augmented=False):
    """
    Constrained forward-backward, streaming the backward messages so only two
    slices are live at a time. Returns
    ``(gamma (T, K), trans_counts (K, K), loglik, xi, gamma_augmented)``, the
    last two ``None`` unless asked for.
    """
    K, T = ctx["K"], ctx["T"]
    dtype, device = ctx["dtype"], ctx["device"]

    log_alpha, log_norm = _forward_messages(ctx)

    loglik = float(log_norm + torch.logsumexp(log_alpha[-1].reshape(-1), dim=0))

    gamma = torch.zeros((T, K), dtype=ACCUM_DTYPE, device=device)
    trans_counts = torch.zeros((K, K), dtype=ACCUM_DTYPE, device=device)

    gamma_augmented = [None] * T if want_augmented else None
    xi_reversed = [] if want_xi else None

    def record(t, log_beta):
        # gamma[t] normalizes to 1, so the arbitrary per-step rescalings of
        # alpha and beta cancel here and never need to be tracked.
        posterior = _normalized_exp(log_alpha[t] + log_beta)

        gamma[t] = posterior.sum(dim=1).to(ACCUM_DTYPE)

        if want_augmented:
            gamma_augmented[t] = posterior.reshape(ctx["shapes"][t])

    # log 1 -- the empty future contributes nothing.
    log_beta = torch.zeros(
        (K, math.prod(ctx["dims_by_time"][T - 1])),
        dtype=dtype,
        device=device,
    )
    record(T - 1, log_beta)

    for t in range(T - 1, 0, -1):
        log_beta_prev, log_beta_tilde = _sum_step_backward(ctx, log_beta, t)

        xi_t = _transition_posterior(ctx, log_alpha[t - 1], log_beta_tilde, t)
        trans_counts += xi_t.to(ACCUM_DTYPE)

        if want_xi:
            xi_reversed.append(xi_t)

        # Unlike the forward pass, an all -inf slice is not an error here: a
        # state can simply have no feasible future.
        shift = log_beta_prev.max()
        log_beta = log_beta_prev - shift if torch.isfinite(shift) else log_beta_prev

        record(t - 1, log_beta)

    xi = None

    if want_xi:
        xi_reversed.reverse()
        xi = (
            torch.stack(xi_reversed)
            if xi_reversed
            else torch.zeros((0, K, K), dtype=dtype, device=device)
        )

    return gamma, trans_counts, loglik, xi, gamma_augmented


# ======================================================================
# E step
# ======================================================================


def _e_step_counts(ctx):
    """
    Expected sufficient statistics for one sequence, as
    ``(init_counts, trans_counts, emit_counts, loglik)``.
    """
    K = ctx["K"]
    device = ctx["device"]

    gamma, trans_counts, loglik, _, _ = _forward_backward(ctx)

    # Only observed times carry an emission statistic; an unobserved one still
    # drives the chain and every MVR active there.
    num_observed = ctx["hmm"].num_observed_states
    emit_counts = torch.zeros((K, num_observed), dtype=ACCUM_DTYPE, device=device)

    for t, o in ctx["observed_index"].items():
        emit_counts[:, o] += gamma[t]

    return gamma[0], trans_counts, emit_counts, loglik


# ======================================================================
# M step
# ======================================================================


def _as_array(values):
    return np.asarray(values, dtype=np.float64)


def _support_masks(hmm):
    """
    Binary masks of the entries an initial HMM puts positive mass on. Zeros are
    permanent, so warn: an unlucky initialization looks like deliberate structure.
    """
    masks = {
        "start": (_as_array(hmm.start_vec) > 0.0).astype(np.float64),
        "transition": (_as_array(hmm.transition_mat) > 0.0).astype(np.float64),
        "emission": (_as_array(hmm.emission_mat) > 0.0).astype(np.float64),
    }

    zeros = {
        name: int(mask.size - mask.sum())
        for name, mask in masks.items()
        if mask.sum() < mask.size
    }

    if zeros:
        detail = ", ".join(f"{name} ({count})" for name, count in zeros.items())
        warnings.warn(
            f"Initial model has zero-probability entries in {detail}. These "
            f"are treated as structural and stay zero for the whole EM run. "
            f"Pass a model with strictly positive parameters (see "
            f"HiddenMarkovModel.make_non_zero) if they were not intended.",
            UserWarning,
            stacklevel=3,
        )

    return masks


def _normalize_on_support(counts, support, eps, fallback):
    """
    Normalize rows over their support, forcing off-support entries to zero. A row
    that collects no mass -- an unreachable state emits and leaves nothing -- has
    a flat objective, so it keeps its current value rather than going uniform.
    """
    mat = counts * support

    # Emptiness is a property of the counts, so it is decided before smoothing.
    # Otherwise the pseudocount alone would carry the row and normalize it to
    # uniform, which is exactly the outcome the fallback exists to avoid.
    empty = mat.sum(axis=1) <= 0

    if eps > 0:
        mat = mat + eps * support

    mat[empty] = fallback[empty]

    return mat / mat.sum(axis=1, keepdims=True)


def _m_step(hmm, counts, masks, pseudocount, update):
    """Normalize the expected counts on support and install them in ``hmm``."""
    init_counts, trans_counts, emit_counts = (c.cpu().numpy() for c in counts)

    start = _as_array(hmm.start_vec)
    transition = _as_array(hmm.transition_mat)
    emission = _as_array(hmm.emission_mat)

    if "start" in update:
        start = _normalize_on_support(
            init_counts.reshape(1, -1),
            masks["start"].reshape(1, -1),
            pseudocount,
            start.reshape(1, -1),
        ).reshape(-1)

    if "transition" in update:
        transition = _normalize_on_support(
            trans_counts, masks["transition"], pseudocount, transition
        )

    if "emission" in update:
        emission = _normalize_on_support(
            emit_counts, masks["emission"], pseudocount, emission
        )

    # Deliberately not load_model: that re-derives the hidden order from
    # sorted(transition_probs) and the observed order from sorted(emission_probs),
    # so a round trip can permute the internal indices and silently break the
    # constraint alignment MVR_CHMM established.
    hmm.start_vec = [float(x) for x in start]
    hmm.transition_mat = [[float(x) for x in row] for row in transition]
    hmm.emission_mat = [[float(x) for x in row] for row in emission]

    hmm.initialize(avoid_reinitialization=False)


# ======================================================================
# Public interface
# ======================================================================


def forward_backward_mvr_chmm(
    model,
    observed,
    *,
    time_horizon=None,
    dtype=torch.float64,
    device="cpu",
    return_augmented=False,
):
    """Constrained forward-backward over the HMM x MVR product.

    Computes the posterior over hidden states given the observations **and** the
    fact that the hidden path satisfies every constraint.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model carrying a hidden Markov model and MVR constraints.
    observed : list or dict
        Observed sequence in external observed-state labels, either a dense list
        timed by position or a sparse ``{time: label}`` map. A time with no
        observation still consumes a transition and still drives every MVR
        active there.
    time_horizon : int, optional
        Number of time steps. Defaults to ``len(observed)`` for a dense list and
        is required for a map.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors. Defaults to ``float64``: unlike
        Viterbi, this recursion carries constraint-satisfaction probabilities
        whose spread grows with the horizon.
    device : str or torch.device, optional
        Torch device.
    return_augmented : bool, optional
        If ``True``, also return the per-time posterior over the full augmented
        state, as a list of tensors shaped ``(K,) + dims(t)``.

    Returns
    -------
    gamma : torch.Tensor
        ``(T, K)`` posterior over hidden states, mediation summed out.
    xi : torch.Tensor
        ``(T - 1, K, K)`` posterior over consecutive hidden-state pairs, indexed
        ``[t, h_prev, h_curr]`` for the transition into time ``t + 1``.
    loglik : float
        ``log P(observed, constraints satisfied | theta)``.
    gamma_augmented : list[torch.Tensor], optional
        Returned when ``return_augmented`` is ``True``.
    """
    ctx = _build_sumprod_ctx(
        model,
        observed,
        time_horizon=time_horizon,
        dtype=dtype,
        device=device,
    )

    gamma, _, loglik, xi, gamma_augmented = _forward_backward(
        ctx, want_xi=True, want_augmented=return_augmented
    )

    if return_augmented:
        return gamma, xi, loglik, gamma_augmented

    return gamma, xi, loglik


def baum_welch_mvr_chmm(
    model,
    observations,
    *,
    time_horizons=None,
    max_iter=50,
    tol=1e-6,
    pseudocount=1e-8,
    update=("start", "transition", "emission"),
    dtype=torch.float64,
    device="cpu",
    verbose=False,
):
    """Fit an HMM by EM, conditioning on the MVR constraints being satisfied.

    Maximizes ``sum_i log P(y_i, constraints satisfied | theta)`` over a batch
    of sequences. The constraint enters the E-step -- expected counts are taken
    under the posterior restricted to feasible paths -- and the M-step is then
    the ordinary normalization of those counts.

    WARNING: Entries that are zero in ``model``'s HMM are treated as structural and stay
    zero; a ``UserWarning`` will be raised in this case.

    Parameters
    ----------
    model : MVR_CHMM
        Constrained model. It is not modified; the fit runs on a copy, which
        keeps the constraints aligned to the hidden-state ordering without
        rebuilding them.
    observations : sequence
        Observed sequences, each a dense list or a sparse ``{time: label}`` map.
    time_horizons : int or sequence of int, optional
        Horizon per sequence. A single int applies to all. Required for any
        sequence given as a map.
    max_iter : int, optional
        Maximum EM iterations. One iteration is one E step followed by one M
        step.
    tol : float, optional
        Stop when the total log-likelihood changes by less than this. Stopping
        without reaching it warns. Pass ``0`` to run exactly ``max_iter``
        iterations without that warning.
    pseudocount : float, optional
        Added to expected counts on their support before normalizing.
    update : tuple of str, optional
        Which parameter blocks the M-step rewrites; the rest are carried over
        unchanged.
    dtype : torch.dtype, optional
        Floating dtype for torch tensors.
    device : str or torch.device, optional
        Torch device.
    verbose : bool, optional
        Print the log-likelihood each iteration.

    Returns
    -------
    hmm : HiddenMarkovModel
        Fitted model, with the label indexing of the input preserved.
    history : list[float]
        Total log-likelihood at the **start** of each iteration, before that
        iteration's update. So ``history[0]`` scores the model passed in. If EM
        converged, no update followed the last entry and ``history[-1]`` scores
        the returned model exactly; if it stopped at ``max_iter`` instead, one
        further update was applied and the returned model is one step ahead of
        ``history[-1]``.
    """
    unknown = set(update) - {"start", "transition", "emission"}

    if unknown:
        raise InvalidInputError(f"Unknown update targets: {sorted(unknown)}")

    observations = list(observations)

    if not observations:
        raise InvalidInputError("observations must contain at least one sequence.")

    horizons = _resolve_time_horizons(observations, time_horizons)

    working = copy.deepcopy(model)
    hmm, constraints = _model_parts(working)

    K = hmm.num_hidden_states
    num_observed = hmm.num_observed_states

    masks = _support_masks(hmm)

    # Neither the MVR factors nor the augmented axes depend on the parameters,
    # so they are built once per distinct horizon and reused every iteration.
    static_cache = {}
    history = []
    converged = False

    for iteration in range(max_iter):
        init_counts = torch.zeros(K, dtype=ACCUM_DTYPE, device=device)
        trans_counts = torch.zeros((K, K), dtype=ACCUM_DTYPE, device=device)
        emit_counts = torch.zeros((K, num_observed), dtype=ACCUM_DTYPE, device=device)
        total_loglik = 0.0

        for index, (observed, horizon) in enumerate(zip(observations, horizons)):
            try:
                T, _ = _resolve_horizon(observed, horizon)

                if T not in static_cache:
                    static_cache[T] = _build_static_context(
                        constraints, K, T, log=True, dtype=dtype, device=device
                    )

                counts = _e_step_counts(
                    _build_sumprod_ctx(
                        working,
                        observed,
                        time_horizon=horizon,
                        dtype=dtype,
                        device=device,
                        static=static_cache[T],
                    )
                )
            except InvalidInputError as exc:
                raise InvalidInputError(f"Sequence {index}: {exc}") from exc

            init_counts += counts[0]
            trans_counts += counts[1]
            emit_counts += counts[2]
            total_loglik += counts[3]

        slack = 1e-6 * max(1.0, abs(history[-1])) if history else 0.0

        if history and total_loglik < history[-1] - slack:
            warnings.warn(
                f"Log-likelihood decreased at iteration {iteration}: "
                f"{history[-1]} -> {total_loglik}. EM on this objective is "
                f"monotone, so this indicates a numerical problem; try "
                f"dtype=torch.float64.",
                UserWarning,
                stacklevel=2,
            )

        history.append(total_loglik)

        if verbose:
            print(f"EM iter {iteration:3d}  loglik = {total_loglik:.10f}")

        # Break before the M step. On this path the returned model is exactly
        # the one whose likelihood is history[-1]; when the loop instead runs out
        # of iterations, one more M step has been applied than history records.
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            converged = True
            break

        _m_step(
            hmm,
            (init_counts, trans_counts, emit_counts),
            masks,
            pseudocount,
            update,
        )

    # tol <= 0 is an explicit request for exactly max_iter iterations, so
    # falling out of the loop is the intended outcome rather than a surprise.
    if tol > 0 and not converged and len(history) > 1:
        warnings.warn(
            f"EM stopped at max_iter={max_iter} without reaching tol={tol}; the "
            f"last change was {abs(history[-1] - history[-2]):.3e}. The returned "
            f"model has had one more update than history records.",
            UserWarning,
            stacklevel=2,
        )

    return hmm, history


def _resolve_time_horizons(observations, time_horizons):
    """Normalize the horizon argument to one entry per sequence."""
    if time_horizons is None:
        return [None] * len(observations)

    if isinstance(time_horizons, (int, np.integer)):
        return [int(time_horizons)] * len(observations)

    horizons = list(time_horizons)

    if len(horizons) != len(observations):
        raise InvalidInputError(
            f"time_horizons has {len(horizons)} entries but there are "
            f"{len(observations)} sequences."
        )

    return horizons


