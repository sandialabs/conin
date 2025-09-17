import itertools
import munch
from collections import defaultdict, deque
from math import prod
from dataclasses import dataclass

from conin.util import batched
from conin.bayesian_network.inference import create_BN_map_query_model
from conin.markov_network import DiscreteFactor


#
# NOTE: We leave the states dictionary off of the CPD object for now.  This is not
# currently used for CPD operations.
#
@dataclass(slots=True)
class DiscreteCPD:
    node: str | int
    values: list | dict
    parents: list = None
    default_value: float = 0  # NOTE: Note used yet

    """Represent a discrete conditional probability distribution (CPD).

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

    >>> cpd = DiscreteCPD(node='grade',
    ...              parents=['diff', 'intel'],
    ...              values={('easy','low'): dict(A=0.2, B=0.2, C=0.6),
    ...                      ('easy','mid'): dict(A=0.3, B=0.3, C=0.4),
    ...                      ('easy','high'): dict(A=0.4, B=0.4, C=0.2),
    ...                      ('hard','low'): dict(A=0.1, B=0.1, C=0.8),
    ...                      ('hard','mid'): dict(A=0.2, B=0.2, C=0.6),
    ...                      ('hard','high'): dict(A=0.3, B=0.3, C=0.4)})
    >>> import os
    >>> os.environ['COLUMNS'] = "100"   # Make sure we print all columns in the table
    >>>
    >>> print(cpd)
    +----------+------------+------------+-------------+------------+------------+-------------+
    | diff     | diff(easy) | diff(easy) | diff(easy)  | diff(hard) | diff(hard) | diff(hard)  |
    +----------+------------+------------+-------------+------------+------------+-------------+
    | intel    | intel(low) | intel(mid) | intel(high) | intel(low) | intel(mid) | intel(high) |
    +----------+------------+------------+-------------+------------+------------+-------------+
    | grade(A) | 0.2        | 0.3        | 0.4         | 0.1        | 0.2        | 0.3         |
    +----------+------------+------------+-------------+------------+------------+-------------+
    | grade(B) | 0.2        | 0.3        | 0.4         | 0.1        | 0.2        | 0.3         |
    +----------+------------+------------+-------------+------------+------------+-------------+
    | grade(C) | 0.6        | 0.4        | 0.2         | 0.8        | 0.6        | 0.4         |
    +----------+------------+------------+-------------+------------+------------+-------------+
    >>> cpd.values
    array([[[0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3]],

           [[0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3]],

           [[0.6, 0.4, 0.2],
            [0.8, 0.6, 0.4]]])
    >>> cpd.variables
    ['grade', 'diff', 'intel']
    >>> cpd.cardinality
    array([3, 2, 3])
    >>> cpd.node
    'grade'
    >>> cpd.cardinality
    array([2])
    >>> cpd.node
    'A'
    >>> cpd.variable_card
    2

    >>> cpd = DiscreteCPD(node='B', parents=['A'],
    ...              values={0:[0.2, 0.8], 1:[0.9, 0.1]})
    >>> print(cpd)
    +------+------+------+
    | A    | A(0) | A(1) |
    +------+------+------+
    | B(0) | 0.2  | 0.9  |
    +------+------+------+
    | B(1) | 0.8  | 0.1  |
    +------+------+------+
    >>> cpd.values
    array([[0.2, 0.9],
           [0.8, 0.1]])
    >>> cpd.variables
    ['B', 'A']
    >>> cpd.cardinality
    array([2, 2])
    >>> cpd.node
    'B'
    >>> cpd.variable_card
    2
    """

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
        """
        if type(self.values) is dict:
            tmp = next(iter(self.values.values()))
            if type(tmp) is list:
                #
                # Convert a CPD specified with list values into a CPD with dictionary values.
                #
                states = pgm.states_of(self.node)
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
            if self.parents:
                slist = [pgm.states_of(node) for node in self.parents]
                node_values = [
                    dict(zip(var_states, vals))
                    for vals in batched(self.values, len(var_states))
                ]
                values = dict(zip(itertools.product(*slist), node_values))
            else:
                values = dict(zip(var_states, self.values))

            return DiscreteCPD(
                node=self.node,
                default_value=self.default_value,
                parents=self.parents,
                values=values,
            )

    def to_factor(self):
        """Convert the CPD into a :class:`~conin.markov_network.DiscreteFactor`.

        The factor mirrors the semantics of the CPD while providing a uniform
        representation that can be combined with other factors during
        inference.

        Returns
        -------
        DiscreteFactor
            Factor representation of the CPD.
        """
        if type(self.values) is list:
            values = self.values
        else:
            tmp = next(iter(self.values.values()))
            if type(tmp) is dict:
                values = {
                    (key if type(key) is tuple else (key,)) + (node_value,): value
                    for key, values in self.values.items()
                    for node_value, value in values.items()
                }
            else:
                values = self.values

        return DiscreteFactor(
            nodes=([self.node] if self.parents is None else self.parents + [self.node]),
            values=values,
            default_value=self.default_value,
        )


class DiscreteBayesianNetwork:
    """Discrete Bayesian network composed of nodes, edges, and CPDs."""

    def __init__(self, *, states={}, cpds=[]):
        """Create a new Bayesian network.

        Parameters
        ----------
        states : dict, optional
            Mapping from each node to its possible states. If omitted, the
            mapping can be populated later.
        cpds : list of DiscreteCPD, optional
            Collection of conditional probability distributions for the
            network.
        """
        self._nodes = []
        self._edges = None
        self._states = states
        self._cpds = cpds

    def check_model(self):
        """Validate that the network structure and CPDs are well formed.

        The method asserts that every CPD references known nodes and that the
        probabilities are non-negative, normalized, and cover every possible
        configuration implied by the node states.

        Raises
        ------
        AssertionError
            If the network contains inconsistent definitions.
        """
        model_nodes = set(self._states.keys())

        cnodes = set()
        for f in self._cpds:
            assert f.node in self._states, f"Unexpected node {f.node} in cpd"
            vnodes = set(self._states[f.node])
            cnodes.add(f.node)
            if f.parents is not None:
                for node in f.parents:
                    cnodes.add(node)

            if type(f.values) is dict:
                for k, v in f.values.items():
                    vkey = False
                    if type(v) is list:
                        for val in v:
                            assert val >= 0 and val <= 1, f"Unexpected cpd value {val}"
                    elif type(v) is dict:
                        for key, val in v.items():
                            assert key in vnodes, f"Unexpected cpd state {key}"
                            assert val >= 0 and val <= 1, f"Unexpected cpd value {val}"
                    else:
                        assert v >= 0 and v <= 1, f"Unexpected cpd value {v}"
                        vkey = True

                    if vkey:
                        # The key is for the node
                        assert (
                            not f.parents and k in vnodes
                        ), f"Unexpected cpd state {key}"
                    else:
                        # The key is for the parents
                        if type(k) is tuple:
                            for i, iv in enumerate(k):
                                assert (
                                    iv in self._states[f.parents[i]]
                                ), f"Unexpected value {iv} in the {i}-th node value of {k}"
                        else:
                            assert (
                                k in self._states[f.parents[0]]
                            ), f"Unexpected node value {k} for parents {self._nodes[0]}. Factor values {f.values}"
            else:
                for v in f.values:
                    assert v >= 0 and v <= 1
                # We assert equality to ensure the list of values covers all combinations of
                # node states
                if f.parents:
                    assert len(f.values) == prod(
                        len(self._states[v]) for v in f.parents
                    ) * len(self._states[f.node])
                else:
                    assert len(f.values) == len(self._states[f.node])

        # Note: We assert equality to ensure that all nodes are used in the model
        assert model_nodes == cnodes

    #
    # Nodes
    #

    @property
    def nodes(self):
        """List the nodes in the Bayesian network.

        Returns
        -------
        list
            Ordered collection of node identifiers.
        """
        return self._nodes

    #
    # Edges
    #

    @property
    def edges(self):
        """Return the edges implied by the CPDs.

        The edges are computed lazily by examining the parent relationships in
        the CPDs and cached for subsequent calls.

        Returns
        -------
        list of tuple
            Sorted list of ``(parent, child)`` pairs.
        """
        if not self._edges:
            self._edges = sorted(
                {
                    (e, cpd.node)
                    for cpd in self._cpds
                    for e in (cpd.parents if cpd.parents else [])
                }
            )
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
            Dictionary mapping each node to an ordered list of states.
        """
        return self._states

    @states.setter
    def states(self, values):
        """Define the state space for each node in the network.

        The setter accepts either a list describing the cardinality for
        anonymous nodes or a dictionary mapping node identifiers to their
        explicit state lists.

        Parameters
        ----------
        values : list or dict
            Cardinalities or explicit state definitions for nodes.

        Raises
        ------
        TypeError
            If ``values`` is neither a list nor a dictionary.
        """
        if type(values) is list:
            self._nodes = list(range(len(values)))
            self._states = {i: list(range(v)) for i, v in enumerate(values)}

        elif type(values) is dict:
            self._nodes = sorted(values.keys())
            self._states = values

        else:
            raise TypeError(f"Unexpected type for states: {type(values)}")

    def states_of(self, node):
        """Return the states associated with a node.

        Parameters
        ----------
        node : str or int
            Node whose state list is requested.

        Returns
        -------
        list
            Ordered list of states for ``node``.
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
            List of CPDs defining the network.
        """
        return self._cpds

    @cpds.setter
    def cpds(self, cpd_list):
        """Store CPDs after normalizing their internal representation.

        Parameters
        ----------
        cpd_list : list of DiscreteCPD
            Iterable of CPDs to attach to the network.
        """
        self._cpds = [cpd.normalize(self) for cpd in cpd_list]

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        """Create a MAP query model for the Bayesian network.

        The model can be used to compute maximum a posteriori assignments for
        ``variables`` given optional ``evidence``.

        Parameters
        ----------
        variables : list, optional
            Nodes for which to compute the MAP estimate.
        evidence : dict, optional
            Observed node assignments.
        timing : bool, optional
            Whether to collect timing information.
        **options : dict, optional
            Additional options forwarded to :func:`create_BN_map_query_model`.

        Returns
        -------
        conin.bayesian_network.inference.BNMapQueryModel
            Query model configured for MAP inference.
        """
        return create_BN_map_query_model(
            pgm=self,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )


class ConstrainedDiscreteBayesianNetwork:
    """Wrap a Bayesian network with optional constraint enforcement."""

    def __init__(self, pgm, constraints=None):
        """Initialise the constrained Bayesian network.

        Parameters
        ----------
        pgm : DiscreteBayesianNetwork
            Underlying Bayesian network to be constrained.
        constraints : callable, optional
            Callable that augments a query model with constraints.
        """
        self.pgm = pgm
        self.constraint_functor = constraints

    def check_model(self):
        """Validate the underlying Bayesian network."""
        self.pgm.check_model()

    def nodes(self):
        """Return the nodes of the wrapped Bayesian network.

        Returns
        -------
        list
            Nodes maintained by the underlying Bayesian network.
        """
        return self.pgm.nodes()

    @property
    def constraints(self, constraint_functor):
        """Assign the constraint functor used to modify MAP query models.

        Parameters
        ----------
        constraint_functor : callable
            Function that receives the query model and associated data,
            returning the constrained model.
        """
        self.constraint_functor = constraint_functor

    def create_constraints(self, model, data):
        """Apply the constraint functor to a MAP query model.

        Parameters
        ----------
        model : conin.bayesian_network.inference.BNMapQueryModel
            Query model produced for the Bayesian network.
        data : munch.Munch
            Data bundle describing variables and evidence.

        Returns
        -------
        conin.bayesian_network.inference.BNMapQueryModel
            Constrained model ready for inference.
        """
        if self.constraint_functor is not None:
            model = self.constraint_functor(model, data)
        return model

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        """Create a MAP query model and apply registered constraints.

        Parameters
        ----------
        variables : list, optional
            Nodes for which to compute the MAP estimate.
        evidence : dict, optional
            Observed node assignments.
        timing : bool, optional
            Whether to collect timing information.
        **options : dict, optional
            Additional keyword arguments forwarded to
            :func:`create_BN_map_query_model`.

        Returns
        -------
        conin.bayesian_network.inference.BNMapQueryModel
            Constrained MAP query model.
        """
        model = create_BN_map_query_model(
            pgm=self.pgm,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )
        self.data = munch.Munch(
            variables=variables,
            evidence=evidence,
        )
        return self.create_constraints(model, self.data)
