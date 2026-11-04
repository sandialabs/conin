import itertools
import math
from math import prod
from dataclasses import dataclass

from conin.exceptions import InvalidInputError
from conin.util import batched
from conin.util.frozen import freeze
from conin.graphical_models.base import GraphicalModel, node_sort_key
from conin.graphical_models.markov_network import DiscreteFactor


#
# NOTE: We leave the states dictionary off of the CPD object for now.  This is not
# currently used for CPD operations.
#
def _inferred_states(cpd):
    """The sorted states of ``cpd.node``, from the keys of its distributions."""
    values = cpd.values
    if isinstance(values, dict):
        row = next(iter(values.values()))
        if not cpd.parents and not isinstance(row, (dict, list)):
            return sorted(values)
        if isinstance(row, dict):
            return sorted(row)
    raise InvalidInputError(
        f"The states of node {cpd.node!r} are not given, and its CPD does not "
        "name them"
    )


@dataclass(frozen=True, slots=True)
class DiscreteCPD:
    """Define a discrete conditional probability distribution (CPD).

    The CPD describes the conditional distribution of ``node`` given its
    ``parents``. Values can be specified either as a mapping from parent
    assignments to probability tables or as a flat list interpreted in
    row-major order.

    Parameters
    ----------
    node : str or int
        Node whose conditional distribution is defined.
    values : list or dict
        Conditional probability values for the node.
    parents : list, optional
        Ordered collection of parent nodes.
    default_value : float, optional
        Default probability returned when a configuration is not specified
        explicitly.

    Notes
    -----
    ``values`` and ``parents`` are read-only copies of the arguments (a
    :class:`~conin.util.frozen.FrozenList` or
    :class:`~conin.util.frozen.FrozenDict`, also for the nested tables), so a
    CPD cannot be changed after construction, also not through a model that
    holds it.

    Examples
    --------
    For a distribution of ``P(grade|diff, intel)``::

        +---------+-------------------------+------------------------+
        |diff     |          easy           |         hard           |
        +---------+------+--------+---------+------+--------+--------+
        |intel    | low  | medium |  high   | low  | medium |  high  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeA   | 0.2  | 0.3    |   0.4   |  0.1 |  0.2   |   0.3  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeB   | 0.2  | 0.3    |   0.4   |  0.1 |  0.2   |   0.3  |
        +---------+------+--------+---------+------+--------+--------+
        |gradeC   | 0.6  | 0.4    |   0.2   |  0.8 |  0.6   |   0.4  |
        +---------+------+--------+---------+------+--------+--------+

    the ``values`` dictionary should be::

       {('easy','low'): dict(A=0.2, B=0.2, C=0.6),
        ('easy','medium'): dict(A=0.3, B=0.3, C=0.4),
        ('easy','high'): dict(A=0.4, B=0.4, C=0.2),
        ('hard','low'): dict(A=0.1, B=0.1, C=0.8),
        ('hard','medium'): dict(A=0.2, B=0.2, C=0.6),
        ('hard','high'): dict(A=0.3, B=0.3, C=0.4)}

    >>> from conin.graphical_models import DiscreteBayesianNetwork, DiscreteCPD
    >>> cpd = DiscreteCPD(
    ...     node='grade',
    ...     parents=['diff', 'intel'],
    ...     values={
    ...         ('easy','low'): dict(A=0.2, B=0.2, C=0.6),
    ...         ('easy','mid'): dict(A=0.3, B=0.3, C=0.4),
    ...         ('easy','high'): dict(A=0.4, B=0.4, C=0.2),
    ...         ('hard','low'): dict(A=0.1, B=0.1, C=0.8),
    ...         ('hard','mid'): dict(A=0.2, B=0.2, C=0.6),
    ...         ('hard','high'): dict(A=0.3, B=0.3, C=0.4),
    ...     },
    ... )
    >>> cpd.node
    'grade'
    >>> cpd.parents
    ['diff', 'intel']
    >>> cpd.values[('easy', 'low')]
    {'A': 0.2, 'B': 0.2, 'C': 0.6}

    List-valued CPDs are normalized against a Bayesian network's state order:

    >>> bn = DiscreteBayesianNetwork(
    ...     states={'A': [0, 1], 'B': [0, 1]},
    ...     cpds=[
    ...         DiscreteCPD(node='A', values=[0.5, 0.5]),
    ...         DiscreteCPD(node='B', values=[0.5, 0.5]),
    ...     ],
    ... )
    >>> cpd = DiscreteCPD(node='B', parents=['A'], values={0: [0.2, 0.8], 1: [0.9, 0.1]})
    >>> cpd.normalize(bn).values
    {0: {0: 0.2, 1: 0.8}, 1: {0: 0.9, 1: 0.1}}
    """

    node: str | int
    values: list | dict
    parents: list = None
    default_value: float = 0  # NOTE: Note used yet

    def __post_init__(self):
        values = self.values if isinstance(self.values, dict) else list(self.values)
        object.__setattr__(self, "values", freeze(values))
        if self.parents is not None:
            object.__setattr__(self, "parents", freeze(list(self.parents)))

    def __str__(self):
        return "\n".join(
            [
                "Dataclass(",
                f"\t{self.node=}",
                f"\t{self.values=}",
                f"\t{self.parents=}",
                f"\t{self.default_value=}",
                "\t)",
            ]
        )

    def normalize(self, pgm):
        """Return a CPD whose values are normalized to dictionaries.

        The method converts CPDs expressed as lists or dictionaries of lists
        into a canonical mapping from parent assignments to dictionaries keyed
        by node states. The state information is retrieved from ``pgm``.

        Parameters
        ----------
        pgm : DiscreteBayesianNetwork
            Bayesian network supplying state metadata.

        Returns
        -------
        DiscreteCPD
            CPD with dictionary-valued entries.

        Raises
        ------
        InvalidInputError
            If a list of values does not have one value per state of
            ``node`` (and combination of parent states).
        """
        if isinstance(self.values, dict):
            tmp = next(iter(self.values.values()))
            if isinstance(tmp, list):
                #
                # Convert a CPD specified with list values into a CPD with dictionary values.
                #
                states = pgm.states_of(self.node)
                for key, val in self.values.items():
                    if len(val) != len(states):
                        raise InvalidInputError(
                            f"The CPD of {self.node!r} has {len(val)} values for "
                            f"the parent states {key!r}, expected {len(states)}"
                        )
                return DiscreteCPD(
                    node=self.node,
                    default_value=self.default_value,
                    parents=self.parents,
                    values={
                        key: {states[i]: value for i, value in enumerate(val)}
                        for key, val in self.values.items()
                    },
                )
            else:
                return self

        else:
            #
            # Convert a CPD specified with a dict of list values into a CPD with dictionary values.
            #
            var_states = pgm.states_of(self.node)
            slist = [pgm.states_of(node) for node in self.parents or []]
            # The list of values must cover all combinations of node states
            expected = len(var_states) * prod(len(states) for states in slist)
            if len(self.values) != expected:
                raise InvalidInputError(
                    f"The CPD of {self.node!r} has {len(self.values)} values, "
                    f"expected {expected}"
                )
            if self.parents:
                node_values = [
                    dict(zip(var_states, vals))
                    for vals in batched(self.values, len(var_states))
                ]
                if len(slist) == 1:
                    values = dict(zip(*slist, node_values))
                else:
                    indices = [index for index in itertools.product(*slist)]
                    values = dict(zip(indices, node_values))
            else:
                values = dict(zip(var_states, self.values))

            return DiscreteCPD(
                node=self.node,
                default_value=self.default_value,
                parents=self.parents,
                values=values,
            )

    def to_factor(self):
        """Convert the CPD into a ``DiscreteFactor``.

        The factor mirrors the semantics of the CPD while providing a uniform
        representation that can be combined with other factors during
        inference.

        Returns
        -------
        DiscreteFactor
            Factor representation of the CPD.
        """
        if isinstance(self.values, list):
            values = self.values
        else:
            tmp = next(iter(self.values.values()))
            if isinstance(tmp, dict):
                values = {
                    (key if type(key) is tuple else (key,)) + (node_value,): value
                    for key, values in self.values.items()
                    for node_value, value in values.items()
                }
            else:
                values = self.values

        return DiscreteFactor(
            nodes=([self.node] if self.parents is None else [*self.parents, self.node]),
            values=values,
            default_value=self.default_value,
        )


class DiscreteBayesianNetwork(GraphicalModel):
    """Discrete Bayesian network composed of nodes, edges, and CPDs."""

    def __init__(self, *, cpds, states=None):
        """Create a new Bayesian network.

        Parameters
        ----------
        cpds : list of DiscreteCPD
            Non-empty collection of conditional probability distributions for
            the network.
        states : list or dict, optional
            Mapping from each node to its possible states, or a list of
            cardinalities for anonymous nodes. States of nodes that are not
            listed are inferred from their CPDs, which must then have
            dictionary values.

        Raises
        ------
        InvalidInputError
            If ``cpds`` is empty, the states of a node can't be inferred, or
            the network is not valid (:meth:`check_model`).
        """
        cpds = list(cpds)
        if not cpds:
            raise InvalidInputError("cpds must be a non-empty list")
        if states is None:
            states = {}
        if isinstance(states, list):
            self._nodes = list(range(len(states)))
            self._states = {i: list(range(v)) for i, v in enumerate(states)}
        elif isinstance(states, dict):
            tmp = list(states.keys())
            tmp.sort(key=node_sort_key)
            self._nodes = tmp
            self._states = dict(states)
        else:
            raise TypeError(f"Unexpected type for states: {type(states)}")
        for cpd in cpds:
            if cpd.node not in self._states:
                self._states[cpd.node] = _inferred_states(cpd)
                if cpd.node not in self._nodes:
                    self._nodes.append(cpd.node)
        self._cpds = freeze([cpd.normalize(self) for cpd in cpds])
        self._nodes = freeze(self._nodes)
        self._states = freeze(self._states)
        # Sorted as the nodes are, so that names and (name, t) tuples mix.
        self._edges = freeze(
            sorted(
                {
                    (e, cpd.node)
                    for cpd in self._cpds
                    for e in (cpd.parents if cpd.parents else [])
                },
                key=lambda edge: tuple(node_sort_key(node) for node in edge),
            )
        )
        self.check_model()

    def check_model(self):
        """Validate that the network structure and CPDs are well formed.

        The constructor calls this, so a model that exists is valid.

        The method checks that every CPD references known nodes, that every
        node has exactly one CPD, that the probabilities are non-negative and
        cover every possible configuration implied by the node states, that
        each distribution sums to 1, and that the CPDs have no cycle.

        Raises
        ------
        InvalidInputError
            If the network contains inconsistent definitions.
        """
        model_nodes = set(self._states.keys())

        cnodes = set()
        for f in self._cpds:
            if f.node not in self._states:
                raise InvalidInputError(f"Unexpected node {f.node} in cpd")
            vnodes = set(self._states[f.node])
            cnodes.add(f.node)
            if f.parents is not None:
                for node in f.parents:
                    if node not in self._states:
                        raise InvalidInputError(
                            f"Unexpected parent {node!r} in the cpd of {f.node!r}"
                        )
                    cnodes.add(node)

            if isinstance(f.values, dict):
                for k, v in f.values.items():
                    vkey = False
                    if isinstance(v, list):
                        for val in v:
                            if not (0 <= val <= 1):
                                raise InvalidInputError(f"Unexpected cpd value {val}")
                    elif isinstance(v, dict):
                        for key, val in v.items():
                            if key not in vnodes:
                                raise InvalidInputError(f"Unexpected cpd state {key}")
                            if not (0 <= val <= 1):
                                raise InvalidInputError(f"Unexpected cpd value {val}")
                    else:
                        if not (0 <= v <= 1):
                            raise InvalidInputError(
                                f"Unexpected cpd value {v} for key {k}"
                            )
                        vkey = True

                    if vkey:
                        # The key is for the node
                        if f.parents or k not in vnodes:
                            raise InvalidInputError(
                                f"Unexpected node value {k} (no parents). "
                                f"Factor values {f.values}"
                            )
                    else:
                        # The key is for the parents
                        if not f.parents:
                            raise InvalidInputError(
                                f"The cpd of {f.node!r} has no parents, but its "
                                f"values are keyed by parent states: {f.values}"
                            )
                        if type(k) is tuple:
                            for i, iv in enumerate(k):
                                if iv not in self._states[f.parents[i]]:
                                    raise InvalidInputError(
                                        f"Unexpected value {iv} in the {i}-th node "
                                        f"value of {k}"
                                    )
                        elif k not in self._states[f.parents[0]]:
                            raise InvalidInputError(
                                f"Unexpected node value {k} for parents "
                                f"{f.parents}. Factor values {f.values}"
                            )
            else:
                for v in f.values:
                    if not (0 <= v <= 1):
                        raise InvalidInputError(f"Unexpected cpd value {v}")
                # The list of values must cover all combinations of node states
                expected = len(self._states[f.node])
                if f.parents:
                    expected *= prod(len(self._states[v]) for v in f.parents)
                if len(f.values) != expected:
                    raise InvalidInputError(
                        f"The cpd of {f.node!r} has {len(f.values)} values, "
                        f"expected {expected}"
                    )

        # Every node must be used in the model
        if model_nodes != cnodes:
            raise InvalidInputError(
                "There are differences between the model nodes and the nodes used "
                f"in the model\n{sorted(model_nodes)=}\n{sorted(cnodes)=}"
            )

        self._check_one_cpd_per_node()
        for cpd in self._cpds:
            self._check_distributions(cpd)
        self._check_acyclic()

    def _check_one_cpd_per_node(self):
        seen = set()
        for cpd in self._cpds:
            if cpd.node in seen:
                raise InvalidInputError(
                    f"There is more than one CPD for node {cpd.node!r}"
                )
            seen.add(cpd.node)
        missing = [n for n in self._nodes if n not in seen]
        if missing:
            raise InvalidInputError(f"There is no CPD for the nodes {missing!r}")

    def _check_distributions(self, cpd):
        """Raise unless ``cpd`` has a distribution summing to 1 for every parent state."""
        cpd = cpd.normalize(self)
        if not cpd.parents:
            rows = {None: cpd.values}
        else:
            rows = dict(cpd.values)
            keys = itertools.product(*(self._states[p] for p in cpd.parents))
            for key in keys:
                key = key[0] if len(cpd.parents) == 1 else key
                if key not in rows:
                    raise InvalidInputError(
                        f"The CPD of {cpd.node!r} has no distribution for the "
                        f"parent states {key!r}"
                    )
        for key, row in rows.items():
            total = sum(row.values())
            if not math.isclose(total, 1, rel_tol=1e-5, abs_tol=1e-8):
                where = "" if key is None else f" for the parent states {key!r}"
                raise InvalidInputError(
                    f"The CPD of {cpd.node!r}{where} must sum to 1, got {total}"
                )

    def _check_acyclic(self):
        children = {n: [] for n in self._nodes}
        indegree = {n: 0 for n in self._nodes}
        for parent, child in self._edges:
            children[parent].append(child)
            indegree[child] += 1
        ready = [n for n, d in indegree.items() if d == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(indegree):
            cycle = sorted((n for n, d in indegree.items() if d > 0), key=node_sort_key)
            raise InvalidInputError(
                f"The CPDs have a cycle through the nodes {cycle!r}"
            )

    #
    # Nodes
    #

    @property
    def nodes(self):
        """List the nodes in the Bayesian network.

        Returns
        -------
        list
            Ordered collection of node identifiers (read-only).
        """
        return self._nodes

    #
    # Edges
    #

    @property
    def edges(self):
        """Return the edges implied by the CPDs.

        The edges are the parent relationships of the CPDs.

        Returns
        -------
        list of tuple
            Sorted list of ``(parent, child)`` pairs (read-only).
        """
        return self._edges

    #
    # States
    #

    @property
    def states(self):
        """Mapping from nodes to their discrete states.

        Returns
        -------
        dict
            Dictionary mapping each node to an ordered list of states
            (a new dict; the state lists are read-only).
        """
        return dict(self._states)

    def states_of(self, node):
        """Return the states associated with a node.

        Parameters
        ----------
        node : str or int
            Node whose state list is requested.

        Returns
        -------
        list
            Ordered list of states for ``node`` (read-only).
        """
        return self._states[node]

    def card(self, node):
        """Return the number of states for a node.

        Parameters
        ----------
        node : str or int
            Node whose cardinality is requested.

        Returns
        -------
        int
            Number of possible states for ``node``.
        """
        return len(self._states[node])

    #
    # CPDs
    #

    @property
    def cpds(self):
        """Return the conditional probability distributions of the network.

        Returns
        -------
        list of DiscreteCPD
            List of CPDs defining the network (a new list; the CPDs
            themselves are immutable).
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
        DiscreteBayesianNetwork
            A new network containing the existing CPDs plus ``cpd``.
        """
        return DiscreteBayesianNetwork(
            states=self.states,
            cpds=self.cpds + [cpd],
        )

    def num_cpd_parameters(self):
        return sum(len(cpd.values) for cpd in self.cpds)
