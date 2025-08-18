import munch
from dataclasses import dataclass

from conin.util import try_import
from conin.bayesian_network.inference import create_BN_map_query_model
from conin.markov_network import DiscreteFactor

with try_import() as pgmpy_available:
    import pgmpy.models


#
# NOTE: We leave the states dictionary off of the CPD object for now.  This is not
# currently used for CPD operations.
#
@dataclass(slots=True)
class DiscreteCPD:
    variable: str | int
    evidence: list
    values: list | dict
    default_value = 0  # NOTE: Note used yet

    """
    Defines a conditional probability distribution table (CPD table)

    Parameters
    ----------
    variable: int, string (any hashable python object)
        The variable whose CPD is defined.

    evidence: array-like
        List of variables in evidences (if any) w.r.t. which CPD is defined.

    values: dict, list
        Values for the CPD table. If a list is specified and no evidence is
        supplied, then the CPD is indexed by integers from 0 to N-1.
        See the example for the format needed for a dictionary.

    Examples
    --------
    For a distribution of P(grade|diff, intel)

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

    the values dictionary should be

       {('easy','low'): dict(A=0.2, B=0.2, C=0.6),
        ('easy','medium'): dict(A=0.3, B=0.3, C=0.4),
        ('easy','high'): dict(A=0.4, B=0.4, C=0.2),
        ('hard','low'): dict(A=0.1, B=0.1, C=0.8),
        ('hard','medium'): dict(A=0.2, B=0.2, C=0.6),
        ('hard','high'): dict(A=0.3, B=0.3, C=0.4)}

    >>> cpd = DiscreteCPD(variable='grade',
    ...              evidence=['diff', 'intel'],
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
    <BLANKLINE>
           [[0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3]],
    <BLANKLINE>
           [[0.6, 0.4, 0.2],
            [0.8, 0.6, 0.4]]])
    >>> cpd.variables
    ['grade', 'diff', 'intel']
    >>> cpd.cardinality
    array([3, 2, 3])
    >>> cpd.variable
    'grade'
    >>> cpd.cardinality
    array([2])
    >>> cpd.variable
    'A'
    >>> cpd.variable_card
    2

    >>> cpd = DiscreteCPD(variable='B', evidence=['A'],
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
    >>> cpd.variable
    'B'
    >>> cpd.variable_card
    2
    """

    def to_factor(self):
        if type(self.values) is dict:
            values = {
                key + (variable_value,): value
                for key, values in self.values.items()
                for variable_value, value in values.items()
            }
        else:
            values = self.values
        return DiscreteFactor(
            nodes=self.evidence + [self.variable],
            values=values,
            default_value=self.default_value,
        )


class DiscreteBayesianNetwork:

    def __init__(self, *, states={}, edges=None, cpds=[]):
        self._nodes = []
        self._states = states
        self._edges = edges if edges else []
        self._cpds = cpds

    #
    # Nodes
    #

    @property
    def nodes(self):
        return self._nodes

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        """
        DBN = DiscreteBayesianNetwork()
        DBN.states = [4, 3]  # Cardinality of nodes
        assert DBN.nodes == [0,1]
        assert DBN.states == {0: [0,1,2,3], 1:[0,1,2]}

        DBN = DiscreteBayesianNetwork()
        DBN.states = {"A": ["T", "F"], "B": [-1, 1]}
        assert DBN.nodes == ["A", "B"]
        assert DBN.states == {"A": ["T", "F"], "B": [-1, 1]}
        """
        if type(values) is list:
            self._nodes = list(range(len(values)))
            self._states = {i: list(range(v)) for i, v in enumerate(values)}

        elif type(values) is dict:
            self._nodes = sorted(values.keys())
            self._states = values

        else:
            raise TypeError(f"Unexpected type for states: {type(values)}")

    def card(self, node):
        return len(self._states[node])

    #
    # Edges
    #

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edges):
        """
        DBN = DiscreteBayesianNetwork()
        DBN.states = [4, 3]  # Cardinality of nodes
        DBN.edges = [ (0,1), (1,2) ]
        """
        self._edges = list(edges)

    #
    # CPDs
    #

    @property
    def cpds(self):
        return self._cpds

    @cpds.setter
    def cpds(self, cpd_list):
        """
        DBN = DiscreteBayesianNetwork()
        DBN.states = [4, 3]  # Cardinality of nodes
        c1 = TabularCPD(nodes=[0,1], values={(0,0):1, (0,1):2, (0,2):3})
        c2 = TabularCPD(nodes=[0], values={0:0, 1:1, 2:2, 3:3})
        DMN.cpds = [c1, c2]
        """
        self._cpds = cpd_list

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        return create_BN_map_query_model(
            pgm=self,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )


class ConstrainedDiscreteBayesianNetwork:

    def __init__(self, pgm, constraints=None):
        self.pgm = convert_to_DiscreteBayesianNetwork(pgm)
        self.constraint_functor = constraints

    def check_model(self):
        self.pgm.check_model()

    def nodes(self):
        return self.pgm.nodes()

    @property
    def constraints(self, constraint_functor):
        self.constraint_functor = constraint_functor

    def create_constraints(self, model, data):
        if self.constraint_functor is not None:
            model = self.constraint_functor(model, data)
        return model

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
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


class PgmpyWrapperDiscreteBayesianNetwork(DiscreteBayesianNetwork):

    def __init__(self, pgmpy_pgm):
        super().__init__()
        self._pgmpy_pgm = pgmpy_pgm
        self.states = pgmpy_pgm.states

        cpds = []
        for cpd in pgmpy_pgm.get_cpds():

            if len(cpd.variables) == 1:  # evidence == []
                values = [v[0] for v in cpd.get_values()]

            else:
                values = cpd.get_values()
                values = [
                    values[i][j]
                    for j in range(len(values[0]))
                    for i in range(len(values))
                ]

            cpds.append(
                DiscreteCPD(
                    variable=cpd.variable,
                    evidence=[] if len(cpd.variables) == 1 else cpd.variables[1:],
                    values=values,
                )
            )
        self.cpds = cpds


def convert_to_DiscreteBayesianNetwork(pgm):
    if (
        type(pgm) is DiscreteBayesianNetwork
        or type(pgm) is PgmpyWrapperDiscreteBayesianNetwork
    ):
        return pgm

    elif type(pgm) is pgmpy.models.DiscreteBayesianNetwork:
        return PgmpyWrapperDiscreteBayesianNetwork(pgm)

    else:
        raise TypeError(f"Unexpected Bayesian network type: {type(pgm)}")
