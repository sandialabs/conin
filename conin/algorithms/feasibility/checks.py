"""Feasibility checks of total assignments against the constraints of a model.

:func:`is_feasible` is dispatched on the constrained model, which decides what
the constraints see, and each constraint is evaluated by an overload of
``_accepts`` on its class.  Only the constraint families whose check is a
direct evaluation are supported:

* ``"oracle"``: the predicate is called with the assignment (HMMs only: the
  other models accept no oracle constraints);
* ``"factor_oracle"``: the predicate is called with the assignment restricted
  to the constraint's node scope;
* ``"smoek"``: the constraints are compiled into one predicate (as for
  variable elimination, with
  :func:`~conin.constraints.smoek.create_oracle_constraints_from_smoek`;
  needs smoek) and evaluated like a factor oracle;
* ``"mvr"``: each MVR is run along the hidden sequence (HMMs only);
* ``"recursive"``: each recursive constraint is run along the hidden sequence
  (HMMs only).

``"pyomo"`` and ``"toulbar2"`` constraint functions can add their own
variables and constraints to the backend model, so checking them is a
feasibility problem of that model; they raise ``NotImplementedError``.
"""

import munch
from ovld import ovld

from conin.constrained_models import (
    ConstrainedDiscreteBayesianNetwork,
    ConstrainedDiscreteMarkovNetwork,
    ConstrainedDynamicDiscreteBayesianNetwork,
    ConstrainedHiddenMarkovModel,
)
from conin.constraints import (
    Constraint,
    FactorOracleConstraint,
    MVRConstraint,
    OracleConstraint,
    PyomoConstraint,
    RecursiveConstraint,
    Toulbar2Constraint,
)
from conin.constraints.smoek import create_oracle_constraints_from_smoek
from conin.constraints.smoek.add_constraints_oracle import StateSpace
from conin.constraints.mvr import BaseMVR, InhomMVR
from conin.constraints.oracle import oracle_accepts
from conin.exceptions import InvalidInputError
from conin.graphical_models.assignments import (
    check_total_assignment,
    dbn_horizon,
    dbn_states,
    hmm_sequences,
)
from conin.util.constraint_data import (
    dbn_time_fields,
    hmm_time_fields,
    merge_constraint_data,
)


def is_feasible(cmodel, assignment, observed=None):
    """Return whether a total assignment satisfies the constraints of a model.

    Parameters
    ----------
    cmodel : ConstrainedGraphicalModel
        The constrained Markov network, Bayesian network, DBN or HMM.
    assignment : dict or list
        For a Markov or Bayesian network, a state for every node of the
        model.  For a DBN, a state for every static node and for every
        ``(name, t)`` with ``name`` a dynamic node and ``t`` in ``0..stop``
        (``stop`` is the largest time in the keys).  For an HMM, the sequence
        of external hidden-state labels (a list, or a dict keyed by the times
        ``0..T-1``).
    observed : list or dict, optional
        HMMs only: the observed labels, needed only by ``"factor_oracle"`` and
        ``"smoek"`` constraints that refer to the ``("E", t)`` nodes.

    Returns
    -------
    bool
        ``True`` if every constraint is satisfied (always, without
        constraints).

        * ``"oracle"`` predicates see the assignment (``{t: hidden}`` for an
          HMM);
        * ``"factor_oracle"`` predicates see the assignment restricted to
          their node scope (the nodes ``("H", t)`` and ``("E", t)`` of the
          unrolled model for an HMM, as for variable elimination);
        * ``"smoek"`` constraints are compiled into a predicate (this
          needs smoek) and see what factor oracles see;
        * ``"mvr"`` constraints are run along the sequence (see
          :func:`mvr_accepts`); ``MVRConstraint`` factories are called with
          ``(hmm, cmodel.data)``;
        * ``"recursive"`` constraints are run along the sequence.

        Constraint functions that take a second argument see the model's
        ``data``; for a DBN with ``T = list(range(stop + 1))`` set on a copy
        (``Munch(T=...)`` if it has none), and for an HMM with
        ``hmm = Munch(T=list(range(T)))`` set on a copy
        (``Munch(hmm=Munch(T=...))`` if it has none).

    Raises
    ------
    InvalidInputError
        If ``assignment`` is not a total assignment of the model (for an
        HMM: an empty or badly keyed sequence, or unknown labels), or a
        constraint needs observations that were not given, or ``observed`` is
        given for a model that is not an HMM.
    NotImplementedError
        For ``"pyomo"`` and ``"toulbar2"`` constraints, whose functions can
        add variables to the backend model.
    TypeError
        For a model that is not one of the constrained models above.

    Examples
    --------
    >>> from conin.algorithms.feasibility import is_feasible
    >>> from conin.constrained_models import ConstrainedHiddenMarkovModel
    >>> from conin.constraints import oracle_constraint_fn
    >>> from conin.graphical_models import HiddenMarkovModel
    >>> hmm = HiddenMarkovModel(
    ...     start_probs={"h0": 0.5, "h1": 0.5},
    ...     transition_probs={
    ...         ("h0", "h0"): 0.9, ("h0", "h1"): 0.1,
    ...         ("h1", "h0"): 0.1, ("h1", "h1"): 0.9,
    ...     },
    ...     emission_probs={("h0", "o"): 1.0, ("h1", "o"): 1.0},
    ... )
    >>> @oracle_constraint_fn()
    ... def ends_in_h1(seq):
    ...     return seq[len(seq) - 1] == "h1"
    >>> chmm = ConstrainedHiddenMarkovModel(hmm, ends_in_h1)
    >>> is_feasible(chmm, ["h0", "h1"]), is_feasible(chmm, ["h1", "h0"])
    (True, False)
    """
    return _is_feasible(cmodel, assignment, observed)


@ovld
def _is_feasible(cmodel: ConstrainedDiscreteMarkovNetwork, assignment, observed):
    return _check_factor_model(cmodel, assignment, observed)


@ovld
def _is_feasible(cmodel: ConstrainedDiscreteBayesianNetwork, assignment, observed):
    return _check_factor_model(cmodel, assignment, observed)


@ovld
def _is_feasible(
    cmodel: ConstrainedDynamicDiscreteBayesianNetwork, assignment, observed
):
    _no_observed(cmodel, observed)
    stop = dbn_horizon(cmodel.model, assignment)
    states = dbn_states(cmodel.model, stop)
    check_total_assignment(states, assignment)
    return _check_constraints(
        cmodel,
        assignment=assignment,
        factor_assignment=assignment,
        state_space=StateSpace(states),
        data=merge_constraint_data(cmodel.data, dbn_time_fields(0, stop)),
    )


@ovld
def _is_feasible(cmodel: ConstrainedHiddenMarkovModel, assignment, observed):
    hmm = cmodel.model
    hidden, observed = hmm_sequences(hmm, assignment, observed)
    T = len(hidden)
    factor_assignment = {("H", t): h for t, h in enumerate(hidden)}
    if observed is not None:
        factor_assignment.update({("E", t): o for t, o in enumerate(observed)})
    return _check_constraints(
        cmodel,
        assignment=dict(enumerate(hidden)),
        factor_assignment=factor_assignment,
        state_space=_hmm_state_space(hmm, T),
        data=merge_constraint_data(
            cmodel.data,
            hmm_time_fields(0, T - 1, hmm.representation("matvec").hidden_to_internal),
        ),
        hidden=hidden,
    )


@ovld
def _is_feasible(cmodel: object, assignment, observed):
    raise TypeError(
        "is_feasible needs a constrained model "
        f"(conin.constrained_models), got {type(cmodel).__name__}"
    )


def partial_is_feasible(cmodel, *, T, seq):
    """Check whether a partial hidden sequence can still be extended feasibly.

    Only constrained HMMs without constraints or with ``"oracle"``
    constraints are supported: the check is the constraints'
    ``partial_func(T, states)``, through ``cmodel.representation("oracle")``.

    Parameters
    ----------
    cmodel : ConstrainedHiddenMarkovModel
        The constrained HMM.
    T : int
        Target full sequence length.
    seq : list
        Partial sequence of external hidden-state labels.

    Returns
    -------
    bool
        ``True`` if the partial sequence admits a feasible completion and
        ``False`` otherwise.

    Raises
    ------
    TypeError
        For a model that is not a ``ConstrainedHiddenMarkovModel``.
    """
    return _partial_is_feasible(cmodel, T, seq)


@ovld
def _partial_is_feasible(cmodel: ConstrainedHiddenMarkovModel, T, seq):
    repn = cmodel.representation("oracle")
    return repn.partial_is_feasible(
        T=T, seq=[repn.hmm.hidden_to_internal[h] for h in seq]
    )


@ovld
def _partial_is_feasible(cmodel: object, T, seq):
    raise TypeError(
        "partial_is_feasible is only defined for a ConstrainedHiddenMarkovModel, "
        f"got {type(cmodel).__name__}"
    )


def _check_factor_model(cmodel, assignment, observed):
    """``is_feasible`` for a Markov or Bayesian network."""
    _no_observed(cmodel, observed)
    states = cmodel.model.states
    check_total_assignment(states, assignment)
    return _check_constraints(
        cmodel,
        assignment=assignment,
        factor_assignment=assignment,
        state_space=StateSpace(states),
        data=cmodel.data,
    )


def _no_observed(cmodel, observed):
    if observed is not None:
        raise InvalidInputError(
            f"observed is only used for HMMs, not for {type(cmodel).__name__}"
        )


def _check_constraints(
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
        What ``"factor_oracle"`` and ``"smoek"`` constraints see (the
        nodes of the possibly unrolled model to states).
    state_space : StateSpace
        The nodes and states the smoek constraints are compiled against.
    data : object
        Passed to constraint functions that take a second argument (and to
        callable node scopes).
    hidden : list, optional
        The hidden-state sequence (HMMs; used by ``"mvr"`` and
        ``"recursive"`` constraints).
    """
    constraints = cmodel.constraints
    if cmodel.constraint_family == "smoek":
        # The smoek constraints are compiled together, into one factor
        # oracle over the whole state space.
        constraints = create_oracle_constraints_from_smoek(
            pgm=state_space, constraints=list(constraints), data=data
        )
    ctx = munch.Munch(
        cmodel=cmodel,
        assignment=assignment,
        factor_assignment=factor_assignment,
        data=data,
        hidden=hidden,
    )
    return all(_accepts(c, ctx) for c in constraints)


@ovld
def _accepts(constraint: OracleConstraint, ctx):
    return oracle_accepts(constraint, ctx.assignment, ctx.data)


@ovld
def _accepts(constraint: FactorOracleConstraint, ctx):
    if type(constraint.nodes) is list:
        nodes = constraint.nodes
    else:
        nodes = list(constraint.nodes(ctx.data))
    missing = [node for node in nodes if node not in ctx.factor_assignment]
    if missing:
        raise InvalidInputError(
            f"Constraint {constraint.name!r} refers to nodes that are not in "
            f"the assignment: {missing}"
        )
    return bool(
        constraint({node: ctx.factor_assignment[node] for node in nodes}, ctx.data)
    )


@ovld
def _accepts(constraint: BaseMVR, ctx):
    # A prebuilt MVR is used as it is (it is not changed).
    return mvr_accepts(constraint, ctx.hidden)


@ovld
def _accepts(constraint: MVRConstraint, ctx):
    # The factory is called as the "mvr" representation calls it.
    return mvr_accepts(constraint(ctx.cmodel.model, ctx.cmodel.data), ctx.hidden)


@ovld
def _accepts(constraint: RecursiveConstraint, ctx):
    return recursive_accepts(constraint, ctx.hidden)


@ovld
def _accepts(constraint: PyomoConstraint | Toulbar2Constraint, ctx):
    family = type(constraint).family
    raise NotImplementedError(
        f"Feasibility checks are not supported for {family!r} constraints: "
        "their functions can add variables and constraints to the "
        f"{family} model, so checking them is an optimization feasibility "
        "problem. Use oracle, factor_oracle, smoek or mvr constraints."
    )


@ovld
def _accepts(constraint: Constraint, ctx):  # pragma: no cover
    raise NotImplementedError(
        f"Feasibility checks are not supported for {type(constraint).family!r} "
        "constraints"
    )


def recursive_accepts(constraint, hidden):
    """Return whether a ``RecursiveConstraint`` accepts a total hidden sequence.

    The state is ``initial(hidden[0])``, then ``update(state, h)`` for each
    later hidden state, and the sequence is accepted iff ``feasible(state)``.
    Like recursive A*, this checks only ``feasible`` (``feasible_partial`` is a
    pruning test).
    """
    state = constraint.initial(hidden[0])
    for h in hidden[1:]:
        state = constraint.update(state, h)
    return bool(constraint.feasible(state))


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


def _hmm_state_space(hmm, T):
    """The nodes ``("H", t)`` and ``("E", t)`` of an HMM unrolled over ``0..T-1``."""
    states = {}
    for t in range(T):
        states[("H", t)] = list(hmm.hidden_states)
        states[("E", t)] = list(hmm.observed_states)
    return StateSpace(states)
