"""Load Toulbar2 cost function network (CFN) files as CONIN Markov networks.

The CFN format is Toulbar2's native JSON-like problem format.  A CFN describes
a set of discrete variables and a set of cost functions over them.  The total
cost of an assignment is the sum of the cost function values, and Toulbar2
minimizes (``"mustbe": "<UB"``) or maximizes (``"mustbe": ">LB"``) this total.

This module translates a CFN into a :class:`DiscreteMarkovNetwork` whose
unnormalized probability is ``exp(-total_cost / cost_scale)``.  Hence, a MAP
assignment of the Markov network is an optimal solution of the CFN.

See https://toulbar2.github.io/toulbar2/formats/cfnformat.html for a
description of the format.
"""

import itertools
import math
import os
import re
import warnings

from conin.constraints import Toulbar2Constraint
from conin.markov_network import (
    ConstrainedDiscreteMarkovNetwork,
    DiscreteFactor,
    DiscreteMarkovNetwork,
)

_number_re = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class _Name(str):
    """A quoted or bare token that is not a number."""


class _Punct(str):
    """A structural token: one of ``{``, ``}``, ``[`` or ``]``."""


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------


def _tokenize(text):
    """Tokenize CFN text.

    Supports both strict JSON and the relaxed CFN syntax accepted by Toulbar2,
    where quotes, commas and colons are optional.  Lines starting with ``#``
    are comments.
    """
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n,:":
            i += 1
        elif c == "#" and (i == 0 or text[i - 1] == "\n"):
            while i < n and text[i] != "\n":
                i += 1
        elif c in "{}[]":
            yield _Punct(c)
            i += 1
        elif c == '"':
            j = text.find('"', i + 1)
            if j < 0:
                raise ValueError("Unterminated string in CFN file")
            yield _Name(text[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n,:{}[]"':
                j += 1
            word = text[i:j]
            if _number_re.match(word):
                yield float(word) if any(ch in word for ch in ".eE") else int(word)
            else:
                yield _Name(word)
            i = j


def _is_punct(token, chars):
    return type(token) is _Punct and token in chars


def _parse_value(tokens, token):
    if _is_punct(token, "{"):
        # Objects are kept as a list of (key, value) pairs because CFN files
        # may legitimately contain duplicate function names.
        items = []
        while True:
            key = next(tokens)
            if _is_punct(key, "}"):
                return items
            if _is_punct(key, "{[]"):
                raise ValueError(f"Unexpected token {key!r} in CFN object")
            items.append((str(key), _parse_value(tokens, next(tokens))))
    if _is_punct(token, "["):
        items = []
        while True:
            t = next(tokens)
            if _is_punct(t, "]"):
                return items
            items.append(_parse_value(tokens, t))
    if _is_punct(token, "}]"):
        raise ValueError(f"Unexpected token {token!r} in CFN file")
    return token


def _parse(text):
    tokens = _tokenize(text)
    try:
        value = _parse_value(tokens, next(tokens))
    except StopIteration:
        raise ValueError("Unexpected end of CFN file") from None
    if not isinstance(value, list) or (value and not isinstance(value[0], tuple)):
        raise ValueError("A CFN file must contain a top-level object")
    return value


def _get(obj, key, default=None):
    for k, v in obj:
        if k == key:
            return v
    return default


def _is_object(value):
    return isinstance(value, list) and all(
        isinstance(v, tuple) and len(v) == 2 for v in value
    )


# -----------------------------------------------------------------------------
# Translation to a Markov network
# -----------------------------------------------------------------------------


def _parse_bound(mustbe):
    """Return (sign, bound) where sign=+1 for minimization, -1 for maximization.

    The upper bound is only returned for minimization problems.  For
    maximization problems the lower bound is ignored.
    """
    if mustbe is None:
        return 1, None
    mustbe = str(mustbe).strip()
    if mustbe.startswith("<"):
        sign = 1
    elif mustbe.startswith(">"):
        sign = -1
    else:
        raise ValueError(f"Unexpected CFN 'mustbe' value: {mustbe!r}")
    rest = mustbe[1:].strip()
    if not _number_re.match(rest):
        raise ValueError(f"Unexpected CFN 'mustbe' value: {mustbe!r}")
    return sign, (float(rest) if sign == 1 else None)


def _parse_variables(variables):
    """Return an ordered dict mapping variable names to lists of state names."""
    if not _is_object(variables):
        raise ValueError("The CFN 'variables' entry must be an object")
    states = {}
    for name, domain in variables:
        if name in states:
            raise ValueError(f"Duplicate CFN variable: {name}")
        if isinstance(domain, int):
            if domain <= 0:
                raise ValueError(f"Variable {name} must have a positive domain size")
            states[name] = list(range(domain))
        elif isinstance(domain, list) and len(domain) > 0:
            states[name] = [str(v) if isinstance(v, _Name) else v for v in domain]
            if len(set(states[name])) != len(states[name]):
                raise ValueError(f"Variable {name} has duplicate value names")
        else:
            raise ValueError(f"Unexpected domain for CFN variable {name}: {domain}")
    return states


def _value_index(states, var, value):
    """Map a value (index or value name) to its index in the variable domain."""
    if isinstance(value, _Name):
        try:
            return states[var].index(str(value))
        except ValueError:
            raise ValueError(
                f"Unknown value {value!r} for CFN variable {var}"
            ) from None
    if isinstance(value, int) and 0 <= value < len(states[var]):
        return value
    raise ValueError(f"Invalid value {value!r} for CFN variable {var}")


def _parse_scope(fname, fdef, states, varnames):
    scope = []
    for v in _get(fdef, "scope", []):
        if isinstance(v, int) and not isinstance(v, bool):
            if not 0 <= v < len(varnames):
                raise ValueError(f"Invalid variable index {v} in CFN function {fname}")
            v = varnames[v]
        v = str(v)
        if v not in states:
            raise ValueError(f"Unknown variable {v} in CFN function {fname}")
        scope.append(v)
    if len(set(scope)) != len(scope):
        raise ValueError(f"CFN function {fname} has a repeated variable in its scope")
    return scope


def _parse_function(fname, fdef, states, varnames, sign):
    """Return (scope, table) where table is a list of costs in lexicographic order.

    Costs are expressed for the equivalent minimization problem.
    """
    scope = _parse_scope(fname, fdef, states, varnames)

    costs = _get(fdef, "costs")
    if not isinstance(costs, list):
        raise NotImplementedError(
            f"CFN function {fname} does not define an explicit cost table"
        )

    cards = [len(states[v]) for v in scope]
    size = math.prod(cards)
    default = _get(fdef, "defaultcost")

    if default is None:
        # Full table in lexicographic order (the last variable changes fastest)
        if len(costs) != size:
            raise ValueError(
                f"CFN function {fname} has {len(costs)} costs but {size} were expected"
            )
        table = [float(c) for c in costs]
    else:
        # Sparse table: a list of tuples (value_1, ..., value_k, cost)
        width = len(scope) + 1
        if len(costs) % width != 0:
            raise ValueError(
                f"CFN function {fname} has a sparse cost list whose length is "
                f"not a multiple of {width}"
            )
        table = [float(default)] * size
        for i in range(0, len(costs), width):
            index = 0
            for var, value, card in zip(scope, costs[i : i + width - 1], cards):
                index = index * card + _value_index(states, var, value)
            table[index] = float(costs[i + width - 1])

    return scope, [sign * c for c in table]


# -----------------------------------------------------------------------------
# Global cost functions that CONIN can represent as Toulbar2 constraints
# -----------------------------------------------------------------------------

_SUPPORTED_GLOBAL_TYPES = ("knapsack", "knapsackv")


def _number(value, what, fname):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected a number for {what} in CFN function {fname}")
    return value


def _parse_knapsack(fname, fdef, states, varnames):
    """Parse a hard ``knapsack`` or ``knapsackv`` global cost function.

    Both encode a linear constraint over value indicators:

        sum_k weight_k * [var_k == value_k] >= capacity

    Returns a list of (node, state, weight) terms and the capacity.
    """
    ftype = str(_get(fdef, "type"))
    scope = _parse_scope(fname, fdef, states, varnames)
    params = _get(fdef, "params")
    if not _is_object(params):
        raise ValueError(f"CFN function {fname} must define 'params' as an object")

    expected = (
        {"capacity", "weights"}
        if ftype == "knapsack"
        else {
            "capacity",
            "weightedvalues",
        }
    )
    keys = {k for k, _ in params}
    if keys != expected:
        raise NotImplementedError(
            f"CFN function {fname} of type {ftype!r} has parameters "
            f"{sorted(keys)}, but only {sorted(expected)} are supported"
        )
    capacity = _number(_get(params, "capacity"), "capacity", fname)

    terms = []
    if ftype == "knapsack":
        # One weight per scope variable, which must be Boolean.  The weight
        # applies when the variable takes its second value (value index 1).
        weights = _get(params, "weights")
        if not isinstance(weights, list) or len(weights) != len(scope):
            raise ValueError(
                f"CFN function {fname} must have one weight per scope variable"
            )
        for var, weight in zip(scope, weights):
            if len(states[var]) != 2:
                raise NotImplementedError(
                    f"CFN function {fname} of type 'knapsack' is only supported "
                    f"for Boolean variables, but variable {var} has "
                    f"{len(states[var])} values"
                )
            terms.append((var, states[var][1], _number(weight, "weights", fname)))
    else:
        # A list of (variable, value index, weight) triples.  Variables are
        # given by name or by their index in the problem's variable list.
        weightedvalues = _get(params, "weightedvalues")
        if not isinstance(weightedvalues, list):
            raise ValueError(f"CFN function {fname} must define 'weightedvalues'")
        for entry in weightedvalues:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError(
                    f"CFN function {fname} has an invalid weighted value: {entry}"
                )
            var, value, weight = entry
            if isinstance(var, int) and not isinstance(var, bool):
                if not 0 <= var < len(varnames):
                    raise ValueError(
                        f"Invalid variable index {var} in CFN function {fname}"
                    )
                var = varnames[var]
            var = str(var)
            if var not in states:
                raise ValueError(f"Unknown variable {var} in CFN function {fname}")
            index = _value_index(states, var, value)
            terms.append((var, states[var][index], _number(weight, "weight", fname)))

    return scope, terms, capacity


def _create_knapsack_constraint(fname, ftype, scope, terms, capacity):
    """Create a Toulbar2Constraint that re-posts a knapsack constraint.

    TODO: Knapsack constraints are linear constraints over value indicators,
    so they could also be converted to algebraic constraints, e.g.

        @algebraic_constraint_fn()
        def constraint(model):
            return sum(w * model.V(var, state) for var, state, w in terms) >= capacity

    That would let loaded constraints be used with every inference method
    (e.g. ``integer_program``), not just ``toulbar2``.  This hasn't been done
    yet because conversion between CONIN constraint types is still an open
    design question.  The ``terms`` and ``rhs`` attributes stored below
    contain everything needed for this conversion.
    """

    def constraint(M):
        M.AddGeneralizedLinearConstraint(
            [M.V(var, state, coef=weight) for var, state, weight in terms],
            ">=",
            capacity,
        )

    con = Toulbar2Constraint(func=constraint, name=fname)
    # Keep a solver-independent description of the constraint so that it can
    # be converted to other constraint representations later.
    con.cfn_type = ftype
    con.scope = scope
    con.terms = terms
    con.operator = ">="
    con.rhs = capacity
    return con


def load_conin_model_from_cfn(filename=None, string=None, cost_scale=1.0):
    """Create a CONIN Markov network from a Toulbar2 CFN file.

    Each cost function in extension becomes a :class:`DiscreteFactor` whose
    values are ``exp(-(cost - min_cost) / cost_scale)``, where ``min_cost`` is
    the smallest cost in that function.  Subtracting the minimum cost only
    rescales each factor, so the most probable assignment of the resulting
    Markov network is an optimal solution of the CFN.  For maximization
    problems (``"mustbe": ">LB"``), costs are negated first.

    Hard linear constraints, which Toulbar2 stores as ``knapsack`` and
    ``knapsackv`` global cost functions, are loaded as
    :class:`~conin.constraints.Toulbar2Constraint` objects.  These are
    written by pytoulbar2's ``AddLinearConstraint``, ``AddSumConstraint`` and
    ``AddGeneralizedLinearConstraint`` methods.  If the file contains such
    constraints, a :class:`ConstrainedDiscreteMarkovNetwork` is returned and
    the unconstrained model is available as its ``pgm`` attribute.  Each
    constraint also records a solver-independent description of the linear
    constraint
    ``sum(weight * [node == state] for node, state, weight in terms) >= rhs``
    in the attributes ``cfn_type``, ``scope``, ``terms``, ``operator`` and
    ``rhs``.  Toulbar2 constraints can only be used with
    ``map_query(..., method="toulbar2")``.

    For minimization problems, the ``mustbe`` upper bound applies to the
    *total* cost, which cannot be represented exactly by independent factors.
    An entry is given a zero factor value if it violates the bound even when
    every other cost function takes its minimum cost.  When all costs are
    non-negative, entries with costs greater than or equal to the upper bound
    are treated as forbidden.  This matches how Toulbar2 encodes
    zero-probability entries.  The lower bound of a maximization problem is
    ignored.

    Variable names and value names are taken from the CFN file.  Variables
    with an integer domain size ``d`` have states ``0, ..., d-1``.  Variables
    that do not appear in any cost function receive a uniform unary factor.
    Zero-arity cost functions are constants and are ignored.

    Parameters
    ----------
    filename : str, optional
        Name of a CFN file.
    string : str, optional
        CFN content.  Used when ``filename`` is not specified.
    cost_scale : float, optional
        Costs are divided by this value before exponentiation.  Use the default
        of 1 when costs are negative natural log-probabilities.  CFN files
        written by Toulbar2 from probabilistic models (e.g. UAI files) store
        costs multiplied by ``10**precision``; Toulbar2's default precision is
        7, so use ``cost_scale=1e7`` for these files.

    Returns
    -------
    DiscreteMarkovNetwork or ConstrainedDiscreteMarkovNetwork
        A ``ConstrainedDiscreteMarkovNetwork`` is returned if the file
        contains supported global cost functions.

    Raises
    ------
    NotImplementedError
        If the CFN contains global cost functions other than hard ``knapsack``
        and ``knapsackv`` functions, ``knapsack`` functions over non-Boolean
        variables, or cost functions that are not explicit tables.
    """
    if cost_scale <= 0:
        raise ValueError(f"cost_scale must be positive: {cost_scale}")

    if filename:
        if not os.path.exists(filename):
            raise RuntimeError(f"Cannot read missing CFN file {filename}")
        with open(filename, "r") as INPUT:
            string = INPUT.read()
    if string is None:
        raise ValueError("Either a filename or a string must be specified")

    cfn = _parse(string)

    problem = _get(cfn, "problem", [])
    sign, bound = _parse_bound(_get(problem, "mustbe") if _is_object(problem) else None)

    states = _parse_variables(_get(cfn, "variables", []))
    varnames = list(states.keys())

    functions = _get(cfn, "functions", [])
    if _is_object(functions):
        functions = list(functions)
    elif isinstance(functions, list):
        # Toulbar2 also accepts a list of unnamed functions
        functions = [(f"F{i}", f) for i, f in enumerate(functions)]
    else:
        raise ValueError("The CFN 'functions' entry must be an object or list")

    parsed = []
    constraints = []
    for fname, fdef in functions:
        if not _is_object(fdef):
            raise ValueError(f"CFN function {fname} must be an object")
        ftype = _get(fdef, "type")
        if ftype is None:
            parsed.append(
                (fname,) + _parse_function(fname, fdef, states, varnames, sign)
            )
        elif str(ftype) in _SUPPORTED_GLOBAL_TYPES:
            scope, terms, capacity = _parse_knapsack(fname, fdef, states, varnames)
            constraints.append(
                _create_knapsack_constraint(fname, str(ftype), scope, terms, capacity)
            )
        else:
            raise NotImplementedError(
                f"CFN function {fname} uses the global cost function type "
                f"{str(ftype)!r}, which cannot be loaded into a CONIN model. "
                f"Supported global cost function types: "
                f"{', '.join(_SUPPORTED_GLOBAL_TYPES)}."
            )

    # The minimum total cost, used to identify entries that violate the bound
    total_min = sum(min(table) for _, _, table in parsed)

    factors = []
    used = set()
    underflow = []
    for fname, scope, table in parsed:
        if len(scope) == 0:
            continue
        used.update(scope)
        fmin = min(table)
        values = []
        for cost in table:
            if bound is not None and cost - fmin + total_min >= bound:
                values.append(0.0)
            else:
                value = math.exp(-(cost - fmin) / cost_scale)
                if value == 0.0:
                    underflow.append(fname)
                values.append(value)

        slist = [states[v] for v in scope]
        if len(scope) == 1:
            fvalues = dict(zip(slist[0], values))
        else:
            fvalues = dict(zip(itertools.product(*slist), values))
        factors.append(DiscreteFactor(nodes=scope, values=fvalues))

    if underflow:
        warnings.warn(
            "Some finite CFN costs were too large to represent as non-zero factor "
            f"values in functions {sorted(set(underflow))}. Consider specifying "
            "cost_scale (e.g. cost_scale=1e7 for CFN files written by Toulbar2 "
            "from probabilistic models)."
        )

    for v in varnames:
        if v not in used:
            factors.append(
                DiscreteFactor(nodes=[v], values={s: 1.0 for s in states[v]})
            )

    pgm = DiscreteMarkovNetwork()
    pgm.states = states
    pgm.factors = factors
    pgm.check_model()
    if constraints:
        return ConstrainedDiscreteMarkovNetwork(pgm, constraints=constraints)
    return pgm
