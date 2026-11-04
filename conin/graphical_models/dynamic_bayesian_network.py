import copy

from conin.exceptions import InvalidInputError
from conin.graphical_models.base import GraphicalModel
from conin.graphical_models.bayesian_network import (
    DiscreteBayesianNetwork,
    DiscreteCPD,
)
from conin.graphical_models.time_index import TimeIndex
from conin.util.frozen import freeze


class DynamicDiscreteBayesianNetwork(GraphicalModel):
    """Dynamic Bayesian network with static and time-indexed state spaces.

    Stores ordinary nodes, dynamic nodes, their corresponding state spaces,
    and conditional probability distributions that may reference time-shifted
    variables.
    """

    def __init__(self, *, cpds, states=None, dynamic_states=None, t=None):
        """Initialize a dynamic discrete Bayesian network.

        Parameters
        ----------
        cpds : list
            Non-empty list of conditional probability distributions associated
            with the model.
        states : list or dict, optional
            Mapping from static node identifiers to their allowed states, or
            a list of cardinalities for anonymous static nodes.
        dynamic_states : dict, optional
            Mapping from dynamic node identifiers to their allowed states across
            time steps. At least one of ``states`` and ``dynamic_states`` must
            be non-empty.
        t : TimeIndex, optional
            Symbolic time-index placeholder used in time-shifted CPD templates
            (``t``, ``t - 1``). CPDs that refer to the time index must be built
            with the same ``TimeIndex`` instance that is passed here;
            it is exposed afterwards as ``dbn.t``. Defaults to a fresh
            ``TimeIndex()``, which is only appropriate when no CPD
            refers to the time index.

        Raises
        ------
        InvalidInputError
            If ``cpds`` is empty, or if both ``states`` and ``dynamic_states``
            are empty.
        """
        cpds = list(cpds)
        if not cpds:
            raise InvalidInputError("cpds must be a non-empty list")
        if states is None:
            states = {}
        if dynamic_states is None:
            dynamic_states = {}
        if not states and not dynamic_states:
            raise InvalidInputError(
                "At least one of states and dynamic_states must be non-empty"
            )

        self._edges = freeze([])
        self._dynamic_edges = freeze([])

        if type(states) is list:
            self._nodes = list(range(len(states)))
            self._states = {i: list(range(v)) for i, v in enumerate(states)}
        elif type(states) is dict:
            self._nodes = sorted(states.keys())
            self._states = dict(states)
        else:
            raise TypeError(f"Unexpected type for states: {type(states)}")

        if type(dynamic_states) is dict:
            self._dynamic_nodes = sorted(dynamic_states.keys())
            self._dynamic_states = dict(dynamic_states)
        else:
            raise TypeError(
                f"Unexpected type for dynamic states: {type(dynamic_states)}"
            )

        self._cpds = freeze([cpd.normalize(self) for cpd in cpds])
        self._nodes = freeze(self._nodes)
        self._states = freeze(self._states)
        self._dynamic_nodes = freeze(self._dynamic_nodes)
        self._dynamic_states = freeze(self._dynamic_states)

        self._t = t if t is not None else TimeIndex()

    @property
    def t(self):
        """TimeIndex: The time index of the CPD templates.

        Unrolling (:func:`~conin.graphical_models.create_bn_from_dbn`) and
        sampling set its value while they evaluate the templates; the
        attribute itself cannot be replaced.
        """
        return self._t

    def check_model(self):
        """Validate the model by unrolling it over the first time steps.

        The DBN is unrolled (:func:`create_bn_from_dbn`) over ``0..k + 1``,
        where ``k`` is the first time step at which every CPD template
        applies, and the unrolled Bayesian network is checked
        (:meth:`DiscreteBayesianNetwork.check_model`).  So every dynamic node
        must have exactly one CPD at every time step (an initial CPD for the
        times before its template applies, and no initial CPD where the
        template applies too), the distributions must sum to 1, and the CPDs
        must have no cycle.  Later time steps repeat the templates, so they
        need no further checks.

        Raises
        ------
        InvalidInputError
            If the unrolled network is not valid.
        """
        stop = self._check_horizon()
        try:
            create_bn_from_dbn(dbn=self, start=0, stop=stop)
        except InvalidInputError as e:
            raise InvalidInputError(
                f"Invalid dynamic Bayesian network (unrolled over 0..{stop}): {e}"
            ) from e

    def _check_horizon(self):
        """The last time step :meth:`check_model` unrolls."""
        probe = 1000
        t_value = self._t.value()
        latest = 0
        try:
            self._t.set_value(probe)
            for cpd in self._cpds:
                for v in [cpd.node, *(cpd.parents or [])]:
                    if type(v) is not tuple or v[0] not in self._dynamic_states:
                        continue
                    if type(v[1]) is int:
                        latest = max(latest, v[1])
                    else:
                        # The template applies from the time at which v is at 0.
                        latest = max(latest, probe - v[1].value())
        finally:
            self._t.set_value(t_value)
        return latest + 1

    #
    # Nodes
    #

    @property
    def nodes(self):
        """Return the static nodes in the model.

        Returns
        -------
        list
            Static node identifiers (read-only).
        """
        return self._nodes

    @property
    def dynamic_nodes(self):
        """Return the dynamic nodes in the model.

        Returns
        -------
        list
            Dynamic node identifiers (read-only).
        """
        return self._dynamic_nodes

    #
    # Edges
    #

    @property
    def edges(self):
        """Return the static edges in the model.

        Returns
        -------
        list
            Static edge definitions (read-only).
        """
        return self._edges

    @property
    def dynamic_edges(self):
        """Return the dynamic edges in the model.

        Returns
        -------
        list
            Dynamic edge definitions (read-only).
        """
        return self._dynamic_edges

    #
    # States
    #

    @property
    def states(self):
        """Return the state space for static nodes.

        Returns
        -------
        dict
            Mapping from static node identifiers to ordered state lists
            (a new dict; the state lists are read-only).

        Examples
        --------
        >>> from conin.graphical_models import DynamicDiscreteBayesianNetwork, DiscreteCPD
        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     states=[4, 3], cpds=[DiscreteCPD(node=0, values=[0.25] * 4)]
        ... )
        >>> ddbn.nodes
        [0, 1]
        >>> ddbn.states
        {0: [0, 1, 2, 3], 1: [0, 1, 2]}

        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     states={"A": ["T", "F"], "B": [-1, 1]},
        ...     cpds=[DiscreteCPD(node="A", values=[0.5, 0.5])],
        ... )
        >>> ddbn.nodes
        ['A', 'B']
        >>> ddbn.states
        {'A': ['T', 'F'], 'B': [-1, 1]}
        """
        return dict(self._states)

    @property
    def dynamic_states(self):
        """Return the state space for dynamic nodes.

        Returns
        -------
        dict
            Mapping from dynamic node identifiers to ordered state lists
            (a new dict; the state lists are read-only).

        Examples
        --------
        >>> from conin.graphical_models import DynamicDiscreteBayesianNetwork, DiscreteCPD
        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     dynamic_states={"A": ["T", "F"], "B": [-1, 1]},
        ...     cpds=[DiscreteCPD(node=("A", 0), values=[0.5, 0.5])],
        ... )
        >>> ddbn.dynamic_nodes
        ['A', 'B']
        >>> ddbn.dynamic_states
        {'A': ['T', 'F'], 'B': [-1, 1]}
        """
        return dict(self._dynamic_states)

    def states_of(self, node):
        """Return the states associated with a node.

        Static nodes, dynamic nodes, and time-indexed dynamic nodes represented
        as tuples are all supported.

        Parameters
        ----------
        node : Hashable or tuple
            Node identifier or time-indexed dynamic node.

        Returns
        -------
        list
            Ordered list of states for ``node`` (read-only).

        Raises
        ------
        ValueError
            If ``node`` does not correspond to a known static or dynamic node.
        """
        if node in self._states:
            return self._states[node]
        elif node in self._dynamic_states:
            return self._dynamic_states[node]
        elif type(node) is tuple and node[0] in self._dynamic_states:
            return self._dynamic_states[node[0]]
        raise ValueError(f"Unexpected node value: {node}")

    def card(self, node):
        """Return the cardinality of a node.

        Parameters
        ----------
        node : Hashable or tuple
            Node identifier or time-indexed dynamic node.

        Returns
        -------
        int
            Number of allowed states for ``node``.

        Raises
        ------
        ValueError
            If ``node`` does not correspond to a known static or dynamic node.
        """
        if node in self._states:
            return len(self._states[node])
        elif node in self._dynamic_states:
            return len(self._dynamic_states[node])
        elif type(node) is tuple and node[0] in self._dynamic_states:
            return len(self._dynamic_states[node[0]])
        raise ValueError(f"Unexpected node value: {node}")

    #
    # CPDs
    #

    @property
    def cpds(self):
        """Return the conditional probability distributions.

        Returns
        -------
        list
            Conditional probability distributions stored on the model (a new
            list; the CPDs themselves are immutable).

        Examples
        --------
        >>> from conin.graphical_models import DynamicDiscreteBayesianNetwork, DiscreteCPD
        >>> ddbn = DynamicDiscreteBayesianNetwork(
        ...     states={"X": ["T", "F"], "Y": [-1, 1]},
        ...     dynamic_states={"A": ["t", "f"], "B": [2, 3]},
        ...     cpds=[
        ...         DiscreteCPD(
        ...             node=("A", None),
        ...             parents=["X"],
        ...             values={"T": {"t": 0.3, "f": 0.7}, "F": {"t": 0.4, "f": 0.6}},
        ...         ),
        ...         DiscreteCPD(
        ...             node=("B", 0),
        ...             parents=["Y"],
        ...             values={-1: {2: 0.3, 3: 0.7}, 1: {2: 0.4, 3: 0.6}},
        ...         ),
        ...     ],
        ... )
        """
        return list(self._cpds)

    def with_cpd(self, cpd):
        """Return a new network with ``cpd`` added to its collection of CPDs.

        Parameters
        ----------
        cpd : DiscreteCPD
            CPD to add to the network.

        Returns
        -------
        DynamicDiscreteBayesianNetwork
            A new network containing the existing CPDs plus ``cpd``.
        """
        return DynamicDiscreteBayesianNetwork(
            states=self.states,
            dynamic_states=self.dynamic_states,
            cpds=self.cpds + [cpd],
            t=self.t,
        )


# Unrolling into a Bayesian network


def _all_cpds(*args):
    for arg in args:
        if arg:
            for v in arg:
                yield v


def check_unroll_start(start):
    """Raise unless ``start`` is ``0``, the only time a DBN or HMM can be unrolled from.

    The CPDs of the first unrolled time step are the DBN's initial CPDs, which
    describe time 0; starting later would need the marginal distribution at
    ``start``.

    Raises
    ------
    InvalidInputError
        If ``start != 0``.
    """
    if start != 0:
        raise InvalidInputError(
            f"A DBN or HMM can only be unrolled from start=0; " f"got start={start}"
        )


def create_bn_from_dbn(*, dbn, start, stop):
    """Unroll a dynamic Bayesian network into a Bayesian network.

    Parameters
    ----------
    dbn : DynamicDiscreteBayesianNetwork
        The model to unroll. It is not changed.
    start : int
        The first time step. Only ``0`` is supported: the CPDs of the first
        unrolled time step are the DBN's initial CPDs, which describe time 0.
    stop : int
        The last time step (inclusive), ``stop >= start``. ``stop == start``
        unrolls one time step.

    Returns
    -------
    DiscreteBayesianNetwork
        The network over the nodes ``(name, t)`` for ``t`` in ``start..stop``
        (plus the DBN's static nodes).

    Raises
    ------
    InvalidInputError
        If ``start != 0`` or ``stop < start``.
    """
    # The transition CPDs at time `start` have parents at `start - 1`, which
    # are not unrolled, so the time-`start` nodes would keep the initial CPDs
    # of time 0 (this used to fail later with an AttributeError).
    check_unroll_start(start)
    if stop < start:
        raise InvalidInputError(
            f"A dynamic Bayesian network is unrolled over start..stop with "
            f"stop >= start; got start={start}, stop={stop}"
        )

    #
    # Copy non-dynamic states
    #
    states = copy.copy(dbn.states)

    #
    # Copy dynamic states to all time steps
    #
    for t in range(start, stop + 1):
        for v, s in dbn.dynamic_states.items():
            states[(v, t)] = s

    #
    # Copy cpds
    #
    # If the node or parents are dynamic tuples, then we treat the
    # cpd as a dynamic map and add it if the time steps for all dynamic variaables
    # are feasible.
    #
    # The loop below evaluates the CPD time expressions by setting the DBN's
    # shared time variable. Restore its value afterwards so that the input DBN
    # is left unchanged.
    t_value = dbn.t.value()
    try:
        cpds = _unroll_cpds(dbn, start, stop)
    finally:
        dbn.t.set_value(t_value)

    pgm = DiscreteBayesianNetwork(states=states, cpds=cpds)
    pgm.check_model()

    return pgm


def _unroll_cpds(dbn, start, stop):
    cpds = []
    for cpd in dbn.cpds:
        dynamic = False
        for v in _all_cpds([cpd.node], cpd.parents):
            if (
                type(v) is tuple
                and v[0] in dbn.dynamic_states
                and not (type(v[1]) is int)
            ):
                dynamic = True
                break

        if not dynamic:
            cpds.append(cpd)
            continue

        for t in range(start, stop + 1):
            dbn.t.set_value(t)

            if type(cpd.node) is tuple:
                curr = cpd.node[1].value()
                if curr < start:
                    continue
                node = (cpd.node[0], curr)
            else:
                node = cpd.node

            skip = False
            parents = []
            for v in _all_cpds(cpd.parents):
                if type(v) is tuple:
                    curr = v[1].value()
                    if curr < start:
                        skip = True
                        break
                    parents.append((v[0], curr))
                else:
                    parents.append(v)

            if skip:
                continue
            cpds.append(DiscreteCPD(node=node, parents=parents, values=cpd.values))

    return cpds
