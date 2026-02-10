import itertools
from typing import Hashable, Optional, Dict, List, Tuple

from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.factors.discrete


def MapCPD(
    *,
    variable: Hashable,
    values: Dict,
    evidence: [List | Tuple] = None,
    state_names: Optional[dict] = None,
):
    """
    Defines the conditional probability distribution table (CPD table)

    Parameters
    ----------
    variable: int, string (any hashable python object)
        The variable whose CPD is defined.

    evidence: array-like
        List of variables in evidences(if any) w.r.t. which CPD is defined.

    values: dict, list
        Values for the CPD table. If a list is specified, then no evidence is
        supplied and the CPD is indexed by integers from 0 to N-1.
        See the example for the format needed for a dictionary.

    state_names: dict
        Optional dictionary that specifies the names of states for the
        variable and evidence names.  These names are inferred from
        the values in sorted order.  This optional dictionary specifies
        a user-defined order.

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

    >>> cpd = MapCPD(variable='grade',
    ...              evidence=['diff', 'intel'],
    ...              values={('easy','low'): dict(A=0.2, B=0.2, C=0.6),
    ...                      ('easy','mid'): dict(A=0.3, B=0.3, C=0.4),
    ...                      ('easy','high'): dict(A=0.4, B=0.4, C=0.2),
    ...                      ('hard','low'): dict(A=0.1, B=0.1, C=0.8),
    ...                      ('hard','mid'): dict(A=0.2, B=0.2, C=0.6),
    ...                      ('hard','high'): dict(A=0.3, B=0.3, C=0.4)},
    ...              state_names={  'grade':['A','B','C'],
    ...                             'diff': ['easy','hard'],
    ...                             'intel': ['low','mid','high']})

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
    >>> cpd.variable_card
    3


    >>> cpd = MapCPD(variable='A', values=[0.9, 0.1])
    >>> print(cpd)
    +------+-----+
    | A(0) | 0.9 |
    +------+-----+
    | A(1) | 0.1 |
    +------+-----+
    >>> cpd.values
    array([0.9, 0.1])
    >>> cpd.variables
    ['A']
    >>> cpd.cardinality
    array([2])
    >>> cpd.variable
    'A'
    >>> cpd.variable_card
    2

    >>> cpd = MapCPD(variable='B', evidence=['A'],
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
    if evidence is None:
        if type(values) is list:
            assert (
                state_names is None
            ), "Cannot specify state names when 'values' is a list"
            variable_card = len(values)
            state_names = {variable: list(range(len(values)))}
            evidence_card = []
            vlist = [[val] for val in values]
        else:
            variable_card = len(values)
            if state_names is None:
                state_names = {variable: list(sorted(values.keys()))}
            evidence_card = []
            vlist = [[values[s]] for s in state_names[variable]]
    else:
        assert (
            type(values) is dict
        ), "The 'evidence' is specified, but 'values' is not a dict"
        first = next(iter(values.values()))
        variable_card = len(first)

        if state_names is None:
            if type(first) is list:
                state_names_ = {variable: list(range(len(first)))}
            else:
                state_names_ = {variable: list(sorted(first.keys()))}
        if type(evidence) not in (list, tuple):
            evidence = list(evidence)
        if len(evidence) == 1:
            evidence_card = [len(values)]
            if state_names is None:
                state_names_[evidence[0]] = list(sorted(values.keys()))
        else:
            evidence_states = []
            for i in range(len(evidence)):
                evidence_states.append(set())
            for k in values:
                for i, s in enumerate(k):
                    evidence_states[i].add(s)
            evidence_card = [len(s) for s in evidence_states]
            if state_names is None:
                for i, s in enumerate(evidence_states):
                    state_names_[evidence[i]] = list(sorted(s))

        if state_names is None:
            state_names = state_names_

        vlist = []
        if len(evidence) == 1:
            for v in state_names[variable]:
                vlist.append([values[prod][v] for prod in state_names[evidence[0]]])
        else:
            for v in state_names[variable]:
                snames = [state_names[e] for e in evidence]
                vlist.append([values[prod][v] for prod in itertools.product(*snames)])

    return pgmpy.factors.discrete.TabularCPD(
        variable=variable,
        evidence=evidence,
        variable_card=variable_card,
        evidence_card=evidence_card,
        state_names=state_names,
        values=vlist,
    )
