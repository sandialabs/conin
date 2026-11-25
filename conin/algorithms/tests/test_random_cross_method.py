"""Randomized cross-method tests: every method agrees with brute force.

Each case is a small random model (MN, BN, DBN or HMM, sometimes with zero
probabilities), random evidence and a random count constraint
(``random_models.py``).  The ``tiny-*`` kinds are degenerate models: 1-2
nodes, nodes with a single state, HMMs with a single hidden or observable
state, and HMM horizons from 1.  ``map_query`` solves it with every method and
constraint family that applies, and each result is compared with
enumeration: the best score, a feasible argmax, or no solution if the
evidence and the constraint exclude every assignment.  A* top-k, partial
MAP and ``log_probability_query`` are compared the same way.

The cases are reproducible: case ``(kind, i)`` is built from the seed
``(i, kind)``.  ``CONIN_RANDOM_CASES`` sets the number of cases of each
kind (default 10, which takes a few seconds); run more for a longer sweep,
e.g. ``CONIN_RANDOM_CASES=500 pytest conin/algorithms/tests -n 4``.
"""

import functools
import math
import os
import zlib

import munch
import numpy as np
import pytest

from conin import log_probability_query, map_query
from conin.config import default_mip_solver
from conin.exceptions import InvalidInputError
from conin.graphical_models import create_bn_from_dbn
from conin.tests.optional import torch_installed as torch_available
from conin.util import try_import

from . import random_models as rm

with try_import() as pytoulbar2_available:
    import pytoulbar2  # noqa: F401

NUM_CASES = int(os.environ.get("CONIN_RANDOM_CASES", "10"))
BASE_KINDS = ["mn", "bn", "dbn", "hmm"]
KINDS = BASE_KINDS + [f"tiny-{kind}" for kind in BASE_KINDS]
TOL = 1e-6

# (name, method, options, constraint families; None is no constraint)
STATIC_METHODS = [
    ("integer_program", "integer_program", {}, [None, "pyomo", "smoek"]),
    ("toulbar2", "toulbar2", {}, [None, "toulbar2", "smoek"]),
    (
        "variable_elimination",
        "variable_elimination",
        {},
        [None, "factor_oracle", "smoek"],
    ),
]
HMM_METHODS = STATIC_METHODS + [
    (
        "network_flow",
        "integer_program",
        {"ip_formulation": "network_flow"},
        [None, "pyomo", "smoek"],
    ),
    ("viterbi", "viterbi", {}, [None]),
    ("a_star", "a_star", {}, [None, "oracle", "recursive"]),
    ("mvr_viterbi", "mvr_viterbi", {}, [None, "mvr"]),
]


def _available(name):
    if name in ("integer_program", "network_flow"):
        return default_mip_solver is not None
    if name == "toulbar2":
        return bool(pytoulbar2_available)
    if name == "mvr_viterbi":
        return bool(torch_available)
    return True


def _rng(kind, i):
    return np.random.default_rng([i, zlib.crc32(kind.encode())])


@functools.cache
def case(kind, i):
    """The random case ``(kind, i)``.

    Returns a Munch with ``pgm``, ``query`` (the ``map_query`` keywords
    besides the method), ``domains`` (``{variable: states}`` of the MAP
    variables), ``evidence`` (fixed MAP variables), ``count`` (the count
    constraint), ``score`` (``x -> log P``) and, for HMMs, ``T``.
    """
    rng = _rng(kind, i)
    tiny = kind.startswith("tiny-")
    base = kind.removeprefix("tiny-")
    zeros = rng.random() < 0.3
    if base == "hmm":
        pgm = rm.random_hmm_case(rng, zeros=zeros, tiny=tiny)
        T = int(rng.integers(1, 4)) if tiny else int(rng.integers(2, 6))
        observed = [
            pgm.observable_states[int(k)]
            for k in rng.integers(len(pgm.observable_states), size=T)
        ]
        h = pgm.hidden_states[int(rng.integers(len(pgm.hidden_states)))]
        lb = int(rng.integers(0, T + 1))
        ub = int(rng.integers(lb, T + 1))
        if lb == 0 and ub == T:
            ub = T - 1
        return munch.Munch(
            kind=base,
            pgm=pgm,
            T=T,
            observed=observed,
            query=dict(evidence=observed),
            domains=rm.variables(pgm, T=T),
            evidence={},
            count=munch.Munch(targets={t: h for t in range(T)}, lb=lb, ub=ub),
            score=lambda x: rm.score(pgm, x, observed),
        )

    make = {"mn": rm.random_mn, "bn": rm.random_bn, "dbn": rm.random_dbn}[base]
    pgm = make(rng, zeros=zeros, tiny=tiny)
    stop = int(rng.integers(1, 3)) if base == "dbn" else None
    domains = rm.variables(pgm, stop=stop)
    names = list(domains)
    k = int(rng.integers(0, max(0, min(2, len(names) - 2)) + 1))
    evidence = {}
    for j in rng.choice(len(names), size=k, replace=False):
        v = names[j]
        evidence[v] = domains[v][int(rng.integers(len(domains[v])))]
    free = {v: d for v, d in domains.items() if v not in evidence}
    unrolled = create_bn_from_dbn(dbn=pgm, start=0, stop=stop) if stop else pgm
    query = dict(evidence=evidence)
    if stop:
        query["stop"] = stop
    return munch.Munch(
        kind=base,
        pgm=pgm,
        stop=stop,
        query=query,
        domains=domains,
        evidence=evidence,
        count=rm.random_count(rng, free),
        score=lambda x: rm.score(unrolled, x),
    )


@functools.cache
def scores(kind, i, constrained):
    """``[(score, x)]`` of every (feasible) MAP assignment, best first."""
    c = case(kind, i)
    out = [
        (c.score(x), x)
        for x in rm.assignments(c.domains, c.evidence)
        if not constrained or rm.count_holds(c.count, x)
    ]
    out = [(s, x) for s, x in out if s != -math.inf]
    out.sort(key=lambda sx: -sx[0])
    return out


def constrained_model(c, family):
    if family is None:
        return c.pgm
    constraint = rm.count_constraint(c.count, family, hmm=c.pgm, T=c.get("T"))
    return rm.constrain(c.pgm, constraint)


def _assignment(c, states):
    """A MAP solution's states as a total assignment of ``c.domains``."""
    if c.kind == "hmm":
        return dict(enumerate(states))
    x = dict(c.evidence)
    x.update(states)
    return x


def _check_map(c, results, best, constrained):
    """The problems of a MAP result (an empty list if it agrees with ``best``)."""
    if not best:
        if results.solution is not None or results.solutions:
            return [f"expected no solution, got {results.solution.states}"]
        if results.termination_condition != "error: no feasible solutions":
            return [f"termination_condition {results.termination_condition!r}"]
        return []
    if results.solution is None:
        return [
            f"no solution (termination {results.termination_condition!r}), "
            f"expected {best[0][0]:.6f}"
        ]
    problems = []
    soln = results.solution
    if not math.isclose(soln.log_potential, best[0][0], abs_tol=TOL):
        problems.append(
            f"log_potential {soln.log_potential:.6f}, expected {best[0][0]:.6f}"
        )
    x = _assignment(c, soln.states)
    if set(x) != set(c.domains):
        problems.append(
            f"states {soln.states} do not assign {sorted(c.domains, key=str)}"
        )
        return problems
    if constrained and not rm.count_holds(c.count, x):
        problems.append(f"states {soln.states} violate the constraint")
    if not math.isclose(c.score(x), best[0][0], abs_tol=TOL):
        problems.append(f"states {soln.states} score {c.score(x):.6f}")
    return problems


def _ids(kinds):
    return [
        pytest.param(kind, i, id=f"{kind}-{i}")
        for kind in kinds
        for i in range(NUM_CASES)
    ]


@pytest.mark.parametrize("kind,i", _ids(KINDS))
def test_map_query_agrees_with_brute_force(kind, i):
    c = case(kind, i)
    methods = HMM_METHODS if c.kind == "hmm" else STATIC_METHODS
    problems = []
    for name, method, options, families in methods:
        if not _available(name):
            continue
        for family in families:
            best = scores(kind, i, family is not None)
            try:
                results = map_query(
                    constrained_model(c, family), method=method, **c.query, **options
                )
            except Exception as e:
                problems.append(f"{name}/{family}: raised {type(e).__name__}: {e}")
                continue
            problems += [
                f"{name}/{family}: {p}"
                for p in _check_map(c, results, best, family is not None)
            ]
    assert not problems, f"case {kind}-{i} ({c.count}):\n" + "\n".join(problems)


@pytest.mark.parametrize("kind,i", _ids(["hmm", "tiny-hmm"]))
@pytest.mark.parametrize("family", [None, "oracle", "recursive"])
def test_a_star_top_k_agrees_with_brute_force(kind, i, family):
    c = case(kind, i)
    k = int(_rng("top-k", i).integers(2, 6))
    best = scores(kind, i, family is not None)
    results = map_query(
        constrained_model(c, family), method="a_star", num_solutions=k, **c.query
    )
    got = [s.log_potential for s in results.solutions]
    expected = [s for s, _ in best[:k]]
    assert got == pytest.approx(expected, abs=TOL)
    for soln in results.solutions:
        x = _assignment(c, soln.states)
        assert family is None or rm.count_holds(c.count, x)
        assert c.score(x) == pytest.approx(soln.log_potential, abs=TOL)
    # Different assignments
    assert len({tuple(s.states) for s in results.solutions}) == len(got)


def _random_subset(rng, names, low, high):
    k = int(rng.integers(low, min(high, len(names)) + 1))
    return [names[j] for j in sorted(rng.choice(len(names), size=k, replace=False))]


@pytest.mark.parametrize(
    "kind,i", _ids(["mn", "bn", "dbn", "tiny-mn", "tiny-bn", "tiny-dbn"])
)
@pytest.mark.parametrize("family", [None, "factor_oracle", "smoek"])
def test_partial_map_variable_elimination_agrees_with_brute_force(kind, i, family):
    c = case(kind, i)
    rng = _rng("partial", i)
    free = [v for v in c.domains if v not in c.evidence]
    if len(free) < 2:
        pytest.skip("partial MAP needs two free variables")
    variables = _random_subset(rng, free, 1, len(free) - 1)
    constrained = family is not None
    # max over the query variables of the sum over the others
    totals = {}
    for s, x in scores(kind, i, constrained):
        key = tuple(x[v] for v in variables)
        totals.setdefault(key, []).append(s)
    totals = {key: rm.logsumexp(v) for key, v in totals.items()}
    results = map_query(
        constrained_model(c, family),
        method="variable_elimination",
        variables=variables,
        **c.query,
    )
    if not totals:
        assert results.solution is None
        return
    best = max(totals.values())
    states = results.solution.states
    assert set(states) == set(variables)
    assert totals.get(tuple(states[v] for v in variables), -math.inf) == pytest.approx(
        best, abs=TOL
    )


def _log_probability_case(c, rng):
    """``(domains, score, holds)`` of the joint distribution of a case's model."""
    if c.kind == "hmm":
        domains = {}
        for t in range(c.T):
            domains["H", t] = list(c.pgm.hidden_states)
            domains["E", t] = list(c.pgm.observable_states)

        def score(x):
            hidden = [x["H", t] for t in range(c.T)]
            observed = [x["E", t] for t in range(c.T)]
            return rm.score(c.pgm, dict(enumerate(hidden)), observed)

        def holds(x):
            return rm.count_holds(c.count, {t: x["H", t] for t in range(c.T)})

        return domains, score, holds
    return c.domains, c.score, lambda x: rm.count_holds(c.count, x)


@pytest.mark.parametrize("kind,i", _ids(KINDS))
@pytest.mark.parametrize("family", [None, "factor_oracle", "smoek"])
def test_log_probability_query_agrees_with_brute_force(kind, i, family):
    c = case(kind, i)
    rng = _rng("log_probability", i)
    domains, score, holds = _log_probability_case(c, rng)
    names = list(domains)
    query = _random_subset(rng, names, 1, 3)
    rest = [v for v in names if v not in query]
    given = _random_subset(rng, rest, 0, 2)

    def pick(v):
        return domains[v][int(rng.integers(len(domains[v])))]

    assignment = {v: pick(v) for v in query}
    evidence = {v: pick(v) for v in given}
    joint = [
        (score(x), x) for x in rm.assignments(domains) if family is None or holds(x)
    ]
    numerator = rm.logsumexp(
        s
        for s, x in joint
        if all(x[v] == assignment.get(v, evidence.get(v, x[v])) for v in x)
    )
    denominator = rm.logsumexp(
        s for s, x in joint if all(x[v] == value for v, value in evidence.items())
    )
    horizon = (
        {"stop": c.T - 1} if c.kind == "hmm" else {"stop": c.stop} if c.stop else {}
    )
    model = constrained_model(c, family)
    if denominator == -math.inf:
        with pytest.raises(InvalidInputError):
            log_probability_query(model, assignment, evidence=evidence, **horizon)
        return
    got = log_probability_query(model, assignment, evidence=evidence, **horizon)
    expected = numerator - denominator
    if expected == -math.inf:
        assert got == -math.inf
    else:
        assert got == pytest.approx(expected, abs=TOL)
