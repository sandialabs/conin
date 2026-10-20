"""Feasibility checks of total assignments against the constraints of a model.

Used by the ``is_feasible`` methods of the constrained models and by
:func:`conin.common.log_potential`.  Only the constraint families whose check
is a direct evaluation are supported:

* ``"oracle"``: the predicate is called with the assignment;
* ``"factor_oracle"``: the predicate is called with the assignment restricted
  to the constraint's node scope;
* ``"algebraic"``: the constraints are compiled into one predicate (as for
  variable elimination, with
  :func:`~conin.constraints.algebraic.create_oracle_constraints_from_algebraic`;
  needs smoek) and evaluated like a factor oracle;
* ``"mvr"``: each MVR is run along the hidden sequence (HMMs only).

``"pyomo"`` and ``"toulbar2"`` constraint functions can add their own
variables and constraints to the backend model, so checking them is a
feasibility problem of that model; they raise ``NotImplementedError``.
"""

import munch

from conin.constraints.mvr import BaseMVR, InhomMVR
from conin.exceptions import InvalidInputError

UNSUPPORTED_FAMILIES = ("pyomo", "toulbar2")


class StateSpace:
    """The ``nodes`` and ``states_of`` of a (possibly unrolled) model.

    This is the part of a graphical model the algebraic-constraint compiler
    reads.
    """

    def __init__(self, states):
        self._states = dict(states)

    @property
    def nodes(self):
        return list(self._states)

    def states_of(self, node):
        return self._states[node]


def check_total_assignment(states, assignment, *, what="assignment"):
    """Check that ``assignment`` gives a valid state to exactly the nodes in ``states``.

    Parameters
    ----------
    states : dict
        Mapping from each node to its list of states.
    assignment : dict
        Mapping from nodes to states.
    what : str, optional
        Name used in error messages.

    Raises
    ------
    InvalidInputError
        If ``assignment`` is not a dict, misses a node, has an unknown node, or
        gives a node a state it does not have.
    """
    if not isinstance(assignment, dict):
        raise InvalidInputError(
            f"The {what} must be a dict, got {type(assignment).__name__}"
        )
    missing = [node for node in states if node not in assignment]
    if missing:
        raise InvalidInputError(
            f"The {what} must give a value to every node; missing: {missing}"
        )
    unknown = [node for node in assignment if node not in states]
    if unknown:
        raise InvalidInputError(f"The {what} has unknown nodes: {unknown}")
    for node, value in assignment.items():
        if value not in states[node]:
            raise InvalidInputError(
                f"In the {what}, {value!r} is not a state of node {node!r}; "
                f"expected one of {list(states[node])}"
            )


def check_constraints(
    cmodel,
    *,
    assignment,
    factor_assignment,
    state_space,
    data,
    hidden=None,
):
    """Return whether an assignment satisfies all of ``cmodel``'s constraints.

    Parameters
    ----------
    cmodel : ConstrainedGraphicalModel
        The model whose constraints are checked.
    assignment : dict
        What ``"oracle"`` predicates see (node names to states; time indices
        to hidden states for an HMM).
    factor_assignment : dict
        What ``"factor_oracle"`` and ``"algebraic"`` constraints see (the
        nodes of the possibly unrolled model to states).
    state_space : StateSpace
        The nodes and states the algebraic constraints are compiled against.
    data : object
        Passed to constraint functions that take a second argument (and to
        callable node scopes).
    hidden : list, optional
        The hidden-state sequence (HMMs; used by ``"mvr"`` constraints).

    Returns
    -------
    bool

    Raises
    ------
    NotImplementedError
        For the ``"pyomo"`` and ``"toulbar2"`` constraint families.
    InvalidInputError
        If a constraint refers to a node that is not in the assignment.
    """
    family = cmodel.constraint_family
    constraints = cmodel.constraints
    if not constraints:
        return True

    if family in UNSUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"Feasibility checks are not supported for {family!r} constraints: "
            "their functions can add variables and constraints to the "
            f"{family} model, so checking them is an optimization feasibility "
            "problem. Use oracle, factor_oracle, algebraic or mvr constraints."
        )

    if family == "oracle":
        return all(_call(c, assignment, data, c.num_args) for c in constraints)

    if family == "factor_oracle":
        return all(
            _check_factor_oracle(c, factor_assignment, data) for c in constraints
        )

    if family == "algebraic":
        from conin.constraints.algebraic import (
            create_oracle_constraints_from_algebraic,
        )

        oracles = create_oracle_constraints_from_algebraic(
            pgm=state_space, constraints=list(constraints), data=data
        )
        return all(_check_factor_oracle(c, factor_assignment, data) for c in oracles)

    if family == "mvr":
        return all(mvr_accepts(mvr, hidden) for mvr in build_mvrs(cmodel))

    raise NotImplementedError(  # pragma: no cover
        f"Feasibility checks are not supported for {family!r} constraints"
    )


def _call(constraint, assignment, data, num_args):
    if num_args == 2:
        return bool(constraint(assignment, data))
    return bool(constraint(assignment))


def _check_factor_oracle(constraint, assignment, data):
    if type(constraint.nodes) is list:
        nodes = constraint.nodes
    else:
        nodes = list(constraint.nodes(data))
    missing = [node for node in nodes if node not in assignment]
    if missing:
        raise InvalidInputError(
            f"Constraint {constraint.name!r} refers to nodes that are not in "
            f"the assignment: {missing}"
        )
    return bool(constraint({node: assignment[node] for node in nodes}, data))


def build_mvrs(cmodel):
    """The MVRs of a constrained HMM, built as the ``"mvr"`` representation builds them.

    ``MVRConstraint`` factories are called with ``(hmm, cmodel.data)``;
    prebuilt MVRs are used as they are (they are not changed).
    """
    hmm = cmodel.model
    return [
        c if isinstance(c, BaseMVR) else c(hmm, cmodel.data) for c in cmodel.constraints
    ]


def mvr_accepts(mvr, hidden):
    """Return whether an MVR accepts a total hidden-state sequence.

    The MVR is initialised at the start ``a`` of its window, updated at each
    later time up to the end ``b`` and evaluated at ``b``.  The window is
    ``mvr.time_range``, or the whole sequence ``[0, len(hidden) - 1]``.  This is
    the semantics of the torch MVR algorithms.

    Parameters
    ----------
    mvr : HomMVR or InhomMVR
    hidden : list
        Hidden-state labels.

    Raises
    ------
    InvalidInputError
        If the window does not fit in the sequence, an inhomogeneous MVR is too
        short for it, or a hidden state is not one of the MVR's.
    """
    T = len(hidden)
    if mvr.time_range is None:
        a, b = 0, T - 1
    else:
        a, b = mvr.time_range
    if b >= T:
        raise InvalidInputError(
            f"MVR time_range [{a}, {b}] exceeds the sequence length {T}"
        )
    is_inhom = isinstance(mvr, InhomMVR)
    if is_inhom and b - a > mvr.time_horizon:
        raise InvalidInputError(
            f"InhomMVR time_horizon {mvr.time_horizon} is too short for the "
            f"window [{a}, {b}]"
        )
    try:
        m = mvr.ini[hidden[a]]
        for t in range(a + 1, b + 1):
            if is_inhom:
                m = mvr.upd[t - a - 1][(m, hidden[t])]
            else:
                m = mvr.upd[(m, hidden[t])]
    except KeyError as exc:
        raise InvalidInputError(
            f"Hidden state {exc.args[0]!r} is not one of the MVR's hidden states"
        ) from None
    return bool(mvr.evl[b - a][m] if is_inhom else mvr.evl[m])


def dbn_horizon(dbn, assignment, *, what="assignment"):
    """Return ``stop`` for a total assignment of ``dbn`` over times ``0..stop``.

    The dynamic nodes are the ``(name, t)`` keys whose ``name`` is a dynamic
    node of ``dbn``.

    Raises
    ------
    InvalidInputError
        If ``assignment`` is not a dict, has no dynamic nodes, or its times are
        not ``0..stop``.
    """
    if not isinstance(assignment, dict):
        raise InvalidInputError(
            f"The {what} must be a dict, got {type(assignment).__name__}"
        )
    dynamic = set(dbn.dynamic_states)
    times = {
        key[1]
        for key in assignment
        if type(key) is tuple and len(key) == 2 and key[0] in dynamic
    }
    if not times:
        raise InvalidInputError(
            f"The {what} has no dynamic nodes (keys (name, t) with name in "
            f"{sorted(dynamic, key=str)})"
        )
    bad = [t for t in times if type(t) is not int]
    if bad:
        raise InvalidInputError(f"The {what} has non-integer times: {bad}")
    stop = max(times)
    if min(times) != 0 or len(times) != stop + 1:
        raise InvalidInputError(
            f"The {what} must cover the times 0..stop; got times {sorted(times)}"
        )
    return stop


def dbn_states(dbn, stop):
    """Map each node of ``dbn`` unrolled over ``0..stop`` to its states."""
    states = dict(dbn.states)
    for t in range(stop + 1):
        for name, s in dbn.dynamic_states.items():
            states[(name, t)] = s
    return states


def dbn_constraint_data(stop):
    """``data`` for DBN constraints, as in the ``"factor"`` representation."""
    return munch.Munch(T=list(range(stop + 1)))


def hmm_sequences(hmm, hidden, observed=None):
    """Return ``(hidden, observed)`` as lists of labels, checked against ``hmm``.

    ``hidden`` and ``observed`` are lists, or dicts keyed by ``0..T-1``.
    ``observed`` may be ``None``.

    Raises
    ------
    InvalidInputError
        For an empty or badly keyed sequence, unknown labels, or different
        lengths.
    """
    hidden = _as_sequence(hidden, "hidden")
    for h in hidden:
        if h not in hmm.hidden_to_internal:
            raise InvalidInputError(
                f"{h!r} is not a hidden state; expected one of {hmm.hidden_states}"
            )
    if observed is None:
        return hidden, None
    observed = _as_sequence(observed, "observed")
    if len(observed) != len(hidden):
        raise InvalidInputError(
            f"The hidden ({len(hidden)}) and observed ({len(observed)}) "
            "sequences must have the same length"
        )
    for o in observed:
        if o not in hmm.observed_to_internal:
            raise InvalidInputError(
                f"{o!r} is not an observed state; expected one of "
                f"{hmm.observed_states}"
            )
    return hidden, observed


def _as_sequence(seq, what):
    if isinstance(seq, dict):
        if not all(type(t) is int for t in seq) or set(seq) != set(range(len(seq))):
            raise InvalidInputError(
                f"A {what} dict must be keyed by the times 0..T-1; got {list(seq)}"
            )
        seq = [seq[t] for t in range(len(seq))]
    elif isinstance(seq, (list, tuple)):
        seq = list(seq)
    else:
        raise InvalidInputError(
            f"The {what} sequence must be a list or a dict keyed by time, got "
            f"{type(seq).__name__}"
        )
    if not seq:
        raise InvalidInputError(f"The {what} sequence must not be empty")
    return seq


def hmm_constraint_data(T):
    """``data`` for HMM constraints, as in the ``"factor"`` representation."""
    return munch.Munch(hmm=munch.Munch(T=list(range(T))))


def hmm_state_space(hmm, T):
    """The nodes ``("H", t)`` and ``("E", t)`` of an HMM unrolled over ``0..T-1``."""
    states = {}
    for t in range(T):
        states[("H", t)] = list(hmm.hidden_states)
        states[("E", t)] = list(hmm.observed_states)
    return StateSpace(states)
