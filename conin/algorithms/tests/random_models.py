"""Small random models, count constraints in every family, and brute force.

The randomized cross-method tests (``test_random_cross_method.py``) solve
each random case with every method that applies and compare the results
with enumeration.  The models are small enough to enumerate (at most a few
thousand assignments).

A *count constraint* says that between ``lb`` and ``ub`` of the variables in
``targets`` (a ``{variable: value}`` dict) take their target value.  It can
be written in every constraint family, so the same constrained model can be
solved by every method.  Variables are keyed as the methods return them:
``"A"`` (MN, BN), ``("A", t)`` (DBN) and ``t`` (the hidden state of an HMM
at time ``t``).
"""

import itertools
import math

import munch
import pyomo.environ as pe

from conin import (
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
    DiscreteBayesianNetwork,
    DiscreteCPD,
    DiscreteFactor,
    DiscreteMarkovNetwork,
    DynamicDiscreteBayesianNetwork,
    log_potential,
    log_probability,
    oracle_constraint_fn,
    pyomo_constraint_fn,
    smoek_constraint_fn,
    toulbar2_constraint_fn,
)
from conin.constraints import OracleConstraint, RecursiveConstraint
from conin.constraints.mvr import HomMVR, mvr_constraint_fn
from conin.graphical_models import TimeIndex, create_bn_from_dbn, random_hmm

CONSTRAINED = {
    DiscreteMarkovNetwork: ConstrainedDiscreteMarkovNetwork,
    DiscreteBayesianNetwork: ConstrainedDiscreteBayesianNetwork,
    DynamicDiscreteBayesianNetwork: ConstrainedDynamicDiscreteBayesianNetwork,
}


# ---------------------------------------------------------------------
# Random models
# ---------------------------------------------------------------------


def _weights(rng, n, zeros):
    """``n`` positive weights; with ``zeros``, some of them (not all) are 0."""
    w = rng.uniform(0.05, 1.0, size=n)
    if zeros:
        w[rng.random(n) < 0.25] = 0.0
        if not w.any():
            w[rng.integers(n)] = 1.0
    return w


def _distribution(rng, states, zeros):
    w = _weights(rng, len(states), zeros)
    return {s: float(p) for s, p in zip(states, w / w.sum())}


def _states(rng, nodes):
    return {n: list(range(int(rng.integers(2, 4)))) for n in nodes}


def random_mn(rng, *, zeros=False):
    """A Markov network over 3-4 nodes: unary factors and random pair factors."""
    nodes = ["A", "B", "C", "D"][: int(rng.integers(3, 5))]
    states = _states(rng, nodes)
    scopes = [[n] for n in nodes]
    scopes += [list(p) for p in itertools.combinations(nodes, 2) if rng.random() < 0.6]
    if rng.random() < 0.3:
        scopes.append(list(rng.choice(nodes, size=3, replace=False)))
    factors = []
    for scope in scopes:
        keys = list(itertools.product(*(states[n] for n in scope)))
        w = _weights(rng, len(keys), zeros)
        values = {k[0] if len(k) == 1 else k: float(v) for k, v in zip(keys, w)}
        factors.append(DiscreteFactor(nodes=[str(n) for n in scope], values=values))
    return DiscreteMarkovNetwork(states=states, factors=factors)


def random_bn(rng, *, zeros=False):
    """A Bayesian network over 3-5 nodes; each node has up to 2 earlier parents."""
    nodes = ["A", "B", "C", "D", "E"][: int(rng.integers(3, 6))]
    states = _states(rng, nodes)
    cpds = []
    for i, node in enumerate(nodes):
        k = int(rng.integers(0, min(i, 2) + 1))
        parents = [str(p) for p in rng.choice(nodes[:i], size=k, replace=False)]
        cpds.append(_cpd(rng, node, parents, states, zeros))
    return DiscreteBayesianNetwork(states=states, cpds=cpds)


def _cpd(rng, node, parents, states, zeros, key=lambda n: n):
    if not parents:
        return DiscreteCPD(
            node=node, values=_distribution(rng, states[key(node)], zeros)
        )
    values = {}
    for pa in itertools.product(*(states[key(p)] for p in parents)):
        values[pa[0] if len(pa) == 1 else pa] = _distribution(
            rng, states[key(node)], zeros
        )
    return DiscreteCPD(node=node, parents=parents, values=values)


def random_dbn(rng, *, zeros=False):
    """A DBN with dynamic nodes A and B: A_t | A_{t-1} and B_t | A_t (, B_{t-1})."""
    t = TimeIndex()
    states = _states(rng, ["A", "B"])
    name = lambda node: node[0]  # noqa: E731
    cpds = [
        _cpd(rng, ("A", 0), [], states, zeros, name),
        _cpd(rng, ("A", t), [("A", t - 1)], states, zeros, name),
    ]
    if rng.random() < 0.5:
        # B_t | A_t, B_{t-1}, with its own CPD at time 0
        cpds.append(_cpd(rng, ("B", 0), [("A", 0)], states, zeros, name))
        cpds.append(_cpd(rng, ("B", t), [("A", t), ("B", t - 1)], states, zeros, name))
    else:
        # B_t | A_t at every time
        cpds.append(_cpd(rng, ("B", t), [("A", t)], states, zeros, name))
    return DynamicDiscreteBayesianNetwork(dynamic_states=states, cpds=cpds, t=t)


def random_hmm_case(rng, *, zeros=False):
    """A random HMM with 2-3 hidden and 2-3 observable states."""
    hidden = [f"h{i}" for i in range(int(rng.integers(2, 4)))]
    observable = [f"o{i}" for i in range(int(rng.integers(2, 4)))]
    hmm = random_hmm(hidden_states=hidden, observable_states=observable, rng=rng)
    if not zeros:
        return hmm
    # Zero out some transitions and emissions (each row keeps one entry).
    tables = {}
    for name, rows, cols in [
        ("transition_probs", hidden, hidden),
        ("emission_probs", hidden, observable),
    ]:
        table = dict(getattr(hmm, name))
        for a in rows:
            keep = cols[int(rng.integers(len(cols)))]
            for b in cols:
                if b != keep and rng.random() < 0.25:
                    table[a, b] = 0.0
            total = sum(table[a, b] for b in cols)
            for b in cols:
                table[a, b] /= total
        tables[name] = table
    return type(hmm)(
        start_probs=hmm.start_probs,
        hidden_states=hidden,
        observable_states=observable,
        **tables,
    )


# ---------------------------------------------------------------------
# Variables and brute force
# ---------------------------------------------------------------------


def variables(pgm, stop=None, T=None):
    """``{variable: states}`` of a model, keyed as the MAP solutions are."""
    if isinstance(pgm, DynamicDiscreteBayesianNetwork):
        return dict(create_bn_from_dbn(dbn=pgm, start=0, stop=stop).states)
    if T is not None:
        return {t: list(pgm.hidden_states) for t in range(T)}
    return dict(pgm.states)


def assignments(domains, fixed=None):
    """Every assignment of ``domains`` (``{variable: states}``) that extends ``fixed``."""
    fixed = fixed or {}
    free = [v for v in domains if v not in fixed]
    for values in itertools.product(*(domains[v] for v in free)):
        x = dict(fixed)
        x.update(zip(free, values))
        yield x


def score(pgm, x, observed=None):
    """log P (or the log-potential of an MN) of a total assignment ``x``."""
    if observed is not None:
        return log_probability(
            pgm, [x[t] for t in range(len(observed))], observed=observed
        )
    return log_potential(pgm, x)


def logsumexp(values):
    values = [v for v in values if v != -math.inf]
    if not values:
        return -math.inf
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


# ---------------------------------------------------------------------
# Count constraints
# ---------------------------------------------------------------------


def random_count(rng, domains, *, size=None):
    """A random count constraint over ``domains`` (``{variable: states}``).

    ``lb..ub`` is a random sub-range of ``0..len(targets)``, so the constraint
    may be infeasible together with the evidence.
    """
    names = list(domains)
    size = size or int(rng.integers(2, min(len(names), 4) + 1))
    chosen = [
        names[i] for i in sorted(rng.choice(len(names), size=size, replace=False))
    ]
    targets = {v: domains[v][int(rng.integers(len(domains[v])))] for v in chosen}
    lb = int(rng.integers(0, size + 1))
    ub = int(rng.integers(lb, size + 1))
    if lb == 0 and ub == size:
        # Not a constraint: tighten one side.
        ub = size - 1
    return munch.Munch(targets=targets, lb=lb, ub=ub)


def count_holds(count, x):
    n = sum(1 for v, value in count.targets.items() if x[v] == value)
    return count.lb <= n <= count.ub


def _var(M, v, value):
    """The indicator variable ``v == value`` of a Pyomo, smoek or toulbar2 model."""
    if isinstance(v, int):
        return M.V("H", v, value)
    if isinstance(v, tuple):
        return M.V(*v, value)
    return M.V(v, value)


def _ve_node(v):
    return ("H", v) if isinstance(v, int) else v


def count_constraint(count, family, *, hmm=None, T=None):
    """The count constraint in ``family``.

    ``family`` is ``"pyomo"``, ``"smoek"``, ``"toulbar2"``,
    ``"factor_oracle"`` (static models, DBNs and HMMs) or ``"oracle"``,
    ``"recursive"``, ``"mvr"`` (HMMs: ``hmm`` and the horizon ``T`` are
    needed, and the targets must be all the times ``0..T-1`` with the same
    value).
    """
    targets, lb, ub = count.targets, count.lb, count.ub

    if family == "pyomo":

        @pyomo_constraint_fn()
        def pyomo_count(M):
            total = sum(_var(M, v, value) for v, value in targets.items())
            M.count_lb = pe.Constraint(expr=total >= lb)
            M.count_ub = pe.Constraint(expr=total <= ub)

        return pyomo_count

    if family == "smoek":

        @smoek_constraint_fn()
        def smoek_count(M):
            total = sum(_var(M, v, value) for v, value in targets.items())
            return [total >= lb, total <= ub]

        return smoek_count

    if family == "toulbar2":

        @toulbar2_constraint_fn()
        def toulbar2_count(M):
            terms = [_var(M, v, value) for v, value in targets.items()]
            M.AddGeneralizedLinearConstraint(terms, ">=", lb)
            M.AddGeneralizedLinearConstraint(terms, "<=", ub)

        return toulbar2_count

    if family == "factor_oracle":
        nodes = {_ve_node(v): value for v, value in targets.items()}

        @oracle_constraint_fn(nodes=list(nodes))
        def factor_oracle_count(states):
            n = sum(1 for v, value in nodes.items() if states[v] == value)
            return lb <= n <= ub

        return factor_oracle_count

    # The HMM families: count the times with the hidden state h.
    (h,) = set(targets.values())
    assert sorted(targets) == list(range(T))

    if family == "oracle":

        def seen(states):
            return sum(1 for s in states.values() if s == h)

        return OracleConstraint(
            func=lambda states: lb <= seen(states) <= ub,
            partial_func=lambda T_, states: seen(states) <= ub
            and seen(states) + (T - len(states)) >= lb,
            name="oracle_count",
        )

    if family == "recursive":
        # The state is (number of states so far, count).
        return RecursiveConstraint(
            initial=lambda s: (1, int(s == h)),
            update=lambda c, s: (c[0] + 1, c[1] + int(s == h)),
            feasible=lambda c: lb <= c[1] <= ub,
            feasible_partial=lambda c, t, T_: c[1] <= ub and c[1] + (T - c[0]) >= lb,
            name="recursive_count",
        )

    if family == "mvr":
        hidden = list(hmm.hidden_states)

        @mvr_constraint_fn(name="mvr_count")
        def mvr_count(model):
            # The count so far, capped at ub + 1.
            m_states = list(range(ub + 2))
            return HomMVR(
                hidden_states=hidden,
                mediation_states=m_states,
                ini={s: min(int(s == h), ub + 1) for s in hidden},
                upd={
                    (m, s): min(m + int(s == h), ub + 1)
                    for m in m_states
                    for s in hidden
                },
                evl={m: lb <= m <= ub for m in m_states},
            )

        return mvr_count

    raise ValueError(family)


def constrain(pgm, constraint):
    """The constrained model of ``pgm`` with ``constraint`` (``pgm`` if it is ``None``)."""
    if constraint is None:
        return pgm
    if hasattr(pgm, "hidden_states"):
        return ConstrainedHiddenMarkovModel(pgm, constraints=[constraint])
    return CONSTRAINED[type(pgm)](pgm, constraints=[constraint])
