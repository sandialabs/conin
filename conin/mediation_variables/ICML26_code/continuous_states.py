import math
import numpy as np
import torch
from scipy.stats import norm


def gaussian_bin_prob(left, right, mean, var):
    std = math.sqrt(var)
    return norm.cdf(right, loc=mean, scale=std) - norm.cdf(left, loc=mean, scale=std)


def build_1d_grid(xmin, xmax, K):
    edges = np.linspace(xmin, xmax, K + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def build_1d_transition_matrix(centers, edges, a, q):
    K = len(centers)
    P = np.zeros((K, K), dtype=np.float64)

    for j in range(K):
        mean = a * centers[j]
        for k in range(K):
            P[j, k] = gaussian_bin_prob(edges[k], edges[k + 1], mean, q)

        row_sum = P[j].sum()
        if row_sum > 0:
            P[j] /= row_sum
        else:
            nearest = np.argmin(np.abs(centers - mean))
            P[j, nearest] = 1.0

    return P


def build_1d_initial_prob(edges, mean0, var0):
    K = len(edges) - 1
    pi = np.zeros(K, dtype=np.float64)

    for k in range(K):
        pi[k] = gaussian_bin_prob(edges[k], edges[k + 1], mean0, var0)

    s = pi.sum()
    if s > 0:
        pi /= s
    else:
        pi[:] = 1.0 / K

    return pi


def build_1d_emission_loglike(y_obs, centers, b, r):
    y_obs = np.asarray(y_obs, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    means = b * centers[None, :]
    resid2 = (y_obs[:, None] - means) ** 2
    const = -0.5 * np.log(2.0 * np.pi * r)
    return const - 0.5 * resid2 / r


def build_1d_emission_loglike_partial(
    T,
    centers,
    b,
    r,
    obs_times,
    obs_values,
):
    """
    Build log emission likelihoods when observations exist only at selected times.

    Parameters
    ----------
    T : int
        Total time horizon.
    centers : array-like, shape (K,)
        Discretized latent-state centers.
    b : float
        Emission coefficient in y_t = b x_t + nu_t.
    r : float
        Emission noise variance.
    obs_times : array-like, shape (M,)
        Integer time indices where observations are available.
    obs_values : array-like, shape (M,)
        Observation values corresponding to obs_times.

    Returns
    -------
    emit_loglik : np.ndarray, shape (T, K)
        Log emission likelihood matrix. For unobserved times t,
        emit_loglik[t, :] = 0.
    """
    centers = np.asarray(centers, dtype=np.float64)
    obs_times = np.asarray(obs_times, dtype=int)
    obs_values = np.asarray(obs_values, dtype=np.float64)

    K = len(centers)
    emit_loglik = np.zeros((T, K), dtype=np.float64)

    const = -0.5 * np.log(2.0 * np.pi * r)

    for t, y_t in zip(obs_times, obs_values):
        mean = b * centers
        emit_loglik[t, :] = const - 0.5 * ((y_t - mean) ** 2) / r

    return emit_loglik


def resource_map_scalar(x, resource_limit):
    if x >= 0:
        val = math.floor(x)
    else:
        val = math.ceil(x)

    return int(max(-resource_limit, min(resource_limit, val)))


def build_resource_values(x_centers, resource_limit):
    return np.array(
        [resource_map_scalar(x, resource_limit) for x in x_centers], dtype=np.int64
    )


def build_resource_budget_tensors_simplified(
    resource_values, lower_bounds, upper_bounds
):
    resource_values = np.asarray(resource_values, dtype=np.int64)
    lower_bounds = np.asarray(lower_bounds, dtype=np.int64)
    upper_bounds = np.asarray(upper_bounds, dtype=np.int64)

    T = len(lower_bounds)
    K = len(resource_values)

    widths = upper_bounds - lower_bounds + 1
    fail_index = widths.copy()
    dims = fail_index + 1
    Rdim_max = int(np.max(dims))

    init_next_r = np.zeros(K, dtype=np.int64)
    next_r = np.zeros((T, Rdim_max, K), dtype=np.int64)
    eval_t = np.zeros((T, Rdim_max), dtype=np.float64)
    valid_mask = np.zeros((T, Rdim_max), dtype=np.float64)

    for t in range(T):
        d = dims[t]
        f = fail_index[t]
        valid_mask[t, :d] = 1.0
        eval_t[t, :f] = 1.0
        eval_t[t, f] = 0.0
        next_r[t, :, :] = f

    L0, U0 = lower_bounds[0], upper_bounds[0]
    f0 = fail_index[0]
    for k in range(K):
        s0 = int(resource_values[k])
        if L0 <= s0 <= U0:
            init_next_r[k] = s0 - L0
        else:
            init_next_r[k] = f0

    for t in range(1, T):
        L_prev, U_prev = lower_bounds[t - 1], upper_bounds[t - 1]
        L_cur, U_cur = lower_bounds[t], upper_bounds[t]
        f_prev = fail_index[t - 1]
        f_cur = fail_index[t]

        next_r[t, :, :] = f_cur

        for s_prev in range(L_prev, U_prev + 1):
            r_prev = s_prev - L_prev
            for k in range(K):
                s_cur = s_prev + int(resource_values[k])
                if L_cur <= s_cur <= U_cur:
                    r_cur = s_cur - L_cur
                else:
                    r_cur = f_cur
                next_r[t, r_prev, k] = r_cur

        next_r[t, f_prev, :] = f_cur

    return {
        "init_next_r": init_next_r,
        "next_r": next_r,
        "eval": eval_t,
        "fail_index": fail_index,
        "low": lower_bounds,
        "high": upper_bounds,
        "valid_mask": valid_mask,
        "Rdim_max": Rdim_max,
    }


def viterbi_torch_resource_budget_simplified(
    init_prob,
    trans_mat,
    emit_loglik,
    cst,
    apply_eval_every_t=False,
    num_corr=1e-300,
    dtype=torch.float64,
    device="cpu",
    debug=False,
):
    init_prob = torch.as_tensor(init_prob, dtype=dtype, device=device)
    trans_mat = torch.as_tensor(trans_mat, dtype=dtype, device=device)
    emit_loglik = torch.as_tensor(emit_loglik, dtype=dtype, device=device)

    init_next_r = torch.as_tensor(cst["init_next_r"], dtype=torch.long, device=device)
    next_r = torch.as_tensor(cst["next_r"], dtype=torch.long, device=device)
    eval_t = torch.as_tensor(cst["eval"], dtype=dtype, device=device)
    valid_mask = torch.as_tensor(cst["valid_mask"], dtype=dtype, device=device)

    T, K = emit_loglik.shape
    Rdim_max = cst["Rdim_max"]

    log_init = torch.log(init_prob + num_corr)
    log_trans = torch.log(trans_mat + num_corr)
    log_eval = torch.log(eval_t + num_corr)
    log_valid = torch.log(valid_mask + num_corr)

    val = torch.full((T, K, Rdim_max), -torch.inf, dtype=dtype, device=device)
    ptr_k = torch.full((T, K, Rdim_max), -1, dtype=torch.long, device=device)
    ptr_r = torch.full((T, K, Rdim_max), -1, dtype=torch.long, device=device)

    v0 = torch.full((K, Rdim_max), -torch.inf, dtype=dtype, device=device)
    base0 = log_init + emit_loglik[0]
    for k in range(K):
        r0 = int(init_next_r[k].item())
        v0[k, r0] = base0[k]

    v0 = v0 + log_valid[0].unsqueeze(0)
    if apply_eval_every_t:
        v0 = v0 + log_eval[0].unsqueeze(0)
    val[0] = v0

    for t in range(1, T):
        cur = torch.full((K, Rdim_max), -torch.inf, dtype=dtype, device=device)

        for k in range(K):
            score_jr = val[t - 1] + log_trans[:, k].unsqueeze(1)
            best_val_rprev, best_j_rprev = torch.max(score_jr, dim=0)
            r_cur_idx = next_r[t, :, k]

            for r_prev in range(Rdim_max):
                r_cur = int(r_cur_idx[r_prev].item())
                cand = best_val_rprev[r_prev]
                if cand > cur[k, r_cur]:
                    cur[k, r_cur] = cand
                    ptr_k[t, k, r_cur] = best_j_rprev[r_prev]
                    ptr_r[t, k, r_cur] = r_prev

        cur = cur + emit_loglik[t].unsqueeze(1)
        cur = cur + log_valid[t].unsqueeze(0)

        if apply_eval_every_t:
            cur = cur + log_eval[t].unsqueeze(0)

        val[t] = cur

    if not apply_eval_every_t:
        val[T - 1] = val[T - 1] + log_eval[T - 1].unsqueeze(0)

    flat_last = int(torch.argmax(val[T - 1].reshape(-1)).item())
    k_cur = flat_last // Rdim_max
    r_cur = flat_last % Rdim_max

    path = [(k_cur, r_cur)]
    for t in range(T - 1, 0, -1):
        k_prev = int(ptr_k[t, k_cur, r_cur].item())
        r_prev = int(ptr_r[t, k_cur, r_cur].item())
        k_cur, r_cur = k_prev, r_prev
        path.append((k_cur, r_cur))

    path.reverse()
    hidden_path = [k for k, r in path]
    resource_state_path = [r for k, r in path]

    if debug:
        return hidden_path, resource_state_path, val, ptr_k, ptr_r
    return hidden_path, resource_state_path


def recover_cumulative_resource_path(resource_state_path, lower_bounds, fail_index):
    T = len(resource_state_path)
    vals = np.empty(T, dtype=np.float64)

    for t in range(T):
        r = resource_state_path[t]
        if r == fail_index[t]:
            vals[t] = np.nan
        elif 0 <= r < fail_index[t]:
            vals[t] = lower_bounds[t] + r
        else:
            vals[t] = np.nan

    return vals


def simulate_1d_ssm(T, a, q, b, r, x0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    y = np.zeros(T)

    prev = x0
    for t in range(T):
        xt = a * prev + rng.normal(0.0, math.sqrt(q))
        yt = b * xt + rng.normal(0.0, math.sqrt(r))
        x[t] = xt
        y[t] = yt
        prev = xt

    return x, y


def resource_value_to_interval(resource_value, resource_limit):
    """
    Map an integer resource value to the corresponding latent-state interval.

    Resource map convention:
      - x >= 0 -> floor(x)
      - x < 0  -> ceil(x)
      - clip to [-resource_limit, resource_limit]

    Thus:
      m = -R -> (-inf, -R]
      -R < m < 0 -> (m-1, m]
      m = 0 -> (-1, 1)
      0 < m < R -> [m, m+1)
      m = R -> [R, inf)

    For continuous Gaussian sampling, endpoint inclusion does not matter.
    """
    R = int(resource_limit)
    m = int(resource_value)

    if m < -R or m > R:
        raise ValueError(
            "resource_value must be within [-resource_limit, resource_limit]."
        )

    if m == -R:
        return -np.inf, -R
    elif -R < m < 0:
        return m - 1, m
    elif m == 0:
        return -1, 1
    elif 0 < m < R:
        return m, m + 1
    else:  # m == R
        return R, np.inf


def gaussian_interval_prob(left, right, mean, std):
    """
    Gaussian probability mass over an interval.
    """
    return norm.cdf(right, loc=mean, scale=std) - norm.cdf(left, loc=mean, scale=std)


def sample_truncated_normal_interval(mean, std, left, right, rng):
    """
    Sample from N(mean, std^2) truncated to the interval [left, right]
    in the continuous sense via inverse-CDF sampling.
    """
    cdf_left = norm.cdf(left, loc=mean, scale=std)
    cdf_right = norm.cdf(right, loc=mean, scale=std)

    if cdf_right <= cdf_left:
        raise RuntimeError(
            "Degenerate truncated-normal interval with zero probability mass."
        )

    u = rng.uniform(cdf_left, cdf_right)
    return norm.ppf(u, loc=mean, scale=std)


def sample_latent_given_budget(
    mean,
    std,
    prev_cum_resource,
    lower_t,
    upper_t,
    resource_limit,
    rng,
):
    """
    Sample x_t from N(mean, std^2), truncated to the set of latent values whose
    mapped resource keeps the cumulative budget within [lower_t, upper_t].

    Returns
    -------
    x_sample : float
    resource_sample : int
    cum_resource_sample : int
    """
    min_res_needed = int(lower_t - prev_cum_resource)
    max_res_needed = int(upper_t - prev_cum_resource)

    feasible_resources = np.arange(
        max(-resource_limit, min_res_needed),
        min(resource_limit, max_res_needed) + 1,
        dtype=int,
    )

    if feasible_resources.size == 0:
        raise RuntimeError("No feasible resource values for this time step.")

    intervals = []
    probs = []

    for m in feasible_resources:
        left, right = resource_value_to_interval(m, resource_limit)
        p = gaussian_interval_prob(left, right, mean, std)
        if p > 0:
            intervals.append((int(m), left, right))
            probs.append(p)

    if len(intervals) == 0:
        raise RuntimeError(
            "Feasible resource values exist, but the transition places negligible mass on all of them."
        )

    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()

    idx = rng.choice(len(intervals), p=probs)
    m, left, right = intervals[idx]

    x_sample = sample_truncated_normal_interval(mean, std, left, right, rng)
    cum_resource_sample = int(prev_cum_resource + m)

    return x_sample, int(m), cum_resource_sample


def simulate_1d_ssm_budget_constrained_truncated(
    T,
    a,
    q,
    b,
    r,
    lower_bounds,
    upper_bounds,
    resource_limit,
    x0=0.0,
    seed=0,
):
    """
    Simulate a 1D linear Gaussian state-space model while enforcing a
    time-varying cumulative-resource budget constraint by sampling directly
    from the Gaussian transition truncated to the feasible latent region.

    Model
    -----
      x_t = a x_{t-1} + eps_t,   eps_t ~ N(0, q)
      y_t = b x_t + nu_t,        nu_t  ~ N(0, r)

    Constraint
    ----------
      Let resource_t = resource_map_scalar(x_t, resource_limit), where
        - x >= 0 -> floor(x)
        - x < 0  -> ceil(x)
        - clip to [-resource_limit, resource_limit]

      Then cumulative resource must satisfy:
        lower_bounds[t] <= sum_{s=0}^t resource_s <= upper_bounds[t]

    Parameters
    ----------
    T : int
    a, q : float
    b, r : float
    lower_bounds, upper_bounds : array-like of shape (T,)
        Integer cumulative budget bounds
    resource_limit : int
    x0 : float
        Previous latent value used to generate x_0 via x_0 ~ N(a*x0, q)
    seed : int

    Returns
    -------
    x : np.ndarray, shape (T,)
    y : np.ndarray, shape (T,)
    resource : np.ndarray, shape (T,)
    cum_resource : np.ndarray, shape (T,)
    """
    lower_bounds = np.asarray(lower_bounds, dtype=int)
    upper_bounds = np.asarray(upper_bounds, dtype=int)

    if len(lower_bounds) != T or len(upper_bounds) != T:
        raise ValueError("lower_bounds and upper_bounds must each have length T.")

    if np.any(lower_bounds > upper_bounds):
        raise ValueError("Each lower bound must be <= the corresponding upper bound.")

    rng = np.random.default_rng(seed)

    x = np.zeros(T, dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    resource = np.zeros(T, dtype=int)
    cum_resource = np.zeros(T, dtype=int)

    prev_x = float(x0)
    prev_cum_resource = 0
    std_x = math.sqrt(q)
    std_y = math.sqrt(r)

    for t in range(T):
        mean_t = a * prev_x

        x_t, res_t, cum_t = sample_latent_given_budget(
            mean=mean_t,
            std=std_x,
            prev_cum_resource=prev_cum_resource,
            lower_t=lower_bounds[t],
            upper_t=upper_bounds[t],
            resource_limit=resource_limit,
            rng=rng,
        )

        y_t = b * x_t + rng.normal(0.0, std_y)

        x[t] = x_t
        y[t] = y_t
        resource[t] = res_t
        cum_resource[t] = cum_t

        prev_x = x_t
        prev_cum_resource = cum_t

    return x, y, resource, cum_resource


def viterbi_torch_unconstrained(
    init_prob,
    trans_mat,
    emit_loglik,
    num_corr=1e-300,
    dtype=torch.float64,
    device="cpu",
    debug=False,
):
    """
    Plain Viterbi for a discretized HMM without any budget/resource constraints.

    Parameters
    ----------
    init_prob : array-like, shape (K,)
        Initial state probabilities.
    trans_mat : array-like, shape (K, K)
        Transition matrix with trans_mat[j, k] = P(z_t = k | z_{t-1} = j).
    emit_loglik : array-like, shape (T, K)
        Log emission likelihoods, emit_loglik[t, k] = log p(y_t | z_t = k).

    Returns
    -------
    path : list[int]
        Most likely hidden-state sequence.
    If debug=True, also returns:
        val : torch.Tensor, shape (T, K)
        ptr : torch.Tensor, shape (T, K)
    """
    init_prob = torch.as_tensor(init_prob, dtype=dtype, device=device)
    trans_mat = torch.as_tensor(trans_mat, dtype=dtype, device=device)
    emit_loglik = torch.as_tensor(emit_loglik, dtype=dtype, device=device)

    T, K = emit_loglik.shape

    log_init = torch.log(init_prob + num_corr)
    log_trans = torch.log(trans_mat + num_corr)

    val = torch.full((T, K), -torch.inf, dtype=dtype, device=device)
    ptr = torch.full((T, K), -1, dtype=torch.long, device=device)

    # t = 0
    val[0] = log_init + emit_loglik[0]

    # recursion
    for t in range(1, T):
        score = val[t - 1].unsqueeze(1) + log_trans  # shape (K_prev, K_cur)
        best_prev_val, best_prev_ix = torch.max(score, dim=0)

        val[t] = best_prev_val + emit_loglik[t]
        ptr[t] = best_prev_ix

    # backtrack
    last_state = int(torch.argmax(val[T - 1]).item())
    path = [last_state]

    for t in range(T - 1, 0, -1):
        last_state = int(ptr[t, last_state].item())
        path.append(last_state)

    path.reverse()

    if debug:
        return path, val, ptr
    return path
