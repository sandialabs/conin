import os
import re
import itertools
import pprint
from collections import defaultdict

from conin.exceptions import InvalidInputError
from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
from conin.graphical_models import DiscreteCPD, DiscreteBayesianNetwork
from conin.graphical_models import DynamicDiscreteBayesianNetwork

lineno = 0


def get_lines(filename, string):
    if filename:
        if not os.path.exists(filename):
            raise RuntimeError(f"Cannot read missing UAI file {filename}")
        with open(filename, "r") as INPUT:
            for line in INPUT:
                yield line
    else:
        for line in string.split("\n"):
            yield line


def tokenize(filename, string):
    global lineno
    lineno = -1

    for line in get_lines(filename, string):
        lineno += 1
        tmp = line.strip()
        if len(tmp) == 0 or tmp[0] == "#":
            continue
        for token in re.split("[ \t]+", tmp):
            yield token


class _Tokens:
    """The UAI tokens; running out raises ``ValueError`` (a truncated file)."""

    def __init__(self, tokens):
        self._tokens = tokens

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._tokens)
        except StopIteration:
            raise ValueError(
                f"Unexpected end of the UAI data after line {lineno + 1}"
            ) from None


def load_uai(filename=None, string=None, verbose=False):
    """Load a CONIN model from a UAI file or string.

    Parameters
    ----------
    filename : str, optional
        Path of a ``.uai`` file.
    string : str, optional
        The UAI text (used when ``filename`` is not given).
    verbose : bool, optional
        Print the parsed sections.

    Returns
    -------
    DiscreteBayesianNetwork or DiscreteMarkovNetwork
        A Bayesian network for a ``BAYES`` file, else a Markov network. The
        variables are named ``var0``, ``var1``, ... and their states are
        ``0..k-1``.

    Raises
    ------
    RuntimeError
        If ``filename`` does not exist.
    ValueError
        If the data ends early or a function table has the wrong length.
    """
    global lineno
    lineno = -1

    tokens = _Tokens(tokenize(filename, string))

    pgmtype = next(tokens)
    if verbose:  # pragma:nocover
        print(f"TYPE {pgmtype}")
    is_bayes = pgmtype == "BAYES"

    # Parse vars
    nvars = int(next(tokens))
    if verbose:  # pragma:nocover
        print(f"NVars {nvars}")
    vcard = [int(next(tokens)) for i in range(nvars)]
    vname = [f"var{i}" for i in range(nvars)]

    # Add states
    states = {vname[i]: list(range(vcard[i])) for i in range(nvars)}

    # Add nodes to PGM
    # if pgmtype == "BAYES":
    #    pgm.add_nodes_from(vname)
    # else:
    #    pgm.add_nodes_from(vname)

    # Parse factor/cpd definitions
    nfun = int(next(tokens))
    if verbose:  # pragma:nocover
        print(f"NFun {nfun}")
    fun = []
    for i in range(nfun):
        n = int(next(tokens))
        fun.append([int(next(tokens)) for i in range(n)])

    # Parse factor/cpd values
    factors = []
    for f in fun:
        n = int(next(tokens))
        N = 1
        for v in f:
            N *= vcard[v]
        if n != N:
            raise ValueError(
                f"Inconsistent function definition at line {lineno}: read {n} "
                f"but expected {N} from function definition"
            )

        if is_bayes:
            if verbose:  # pragma:nocover
                print("")
                print(f[-1])
            if len(f) == 1:
                map_values = {i: float(next(tokens)) for i in range(vcard[f[0]])}

                parents = None
            elif len(f) == 2:
                map_values = defaultdict(list)
                for index in range(vcard[f[0]]):
                    for i in range(vcard[f[-1]]):
                        map_values[index].append((i, float(next(tokens))))
                map_values = {k: dict(v) for k, v in map_values.items()}

                parents = [vname[v] for v in f[:-1]]
            else:
                vlist = [list(range(vcard[v])) for v in f[:-1]]

                indices = list(itertools.product(*vlist))
                map_values = {index: dict() for index in indices}
                for index in indices:
                    for i in range(vcard[f[-1]]):
                        map_values[index][i] = float(next(tokens))

                parents = [vname[v] for v in f[:-1]]

            if verbose:  # pragma:nocover
                pprint.pprint(map_values)
                print("")
            factors.append(
                DiscreteCPD(node=vname[f[-1]], parents=parents, values=map_values)
            )
        else:
            if len(f) == 1:
                values = [float(next(tokens)) for i in range(vcard[f[0]])]
            else:
                vlist = [list(range(vcard[v])) for v in f]

                values = {
                    index: float(next(tokens)) for index in itertools.product(*vlist)
                }

            if verbose:  # pragma:nocover
                pprint.pprint(values)
                print("")
            factors.append(
                DiscreteFactor(
                    nodes=[vname[v] for v in f],
                    values=values,
                )
            )

    # Add factors/cpds
    if is_bayes:
        pgm = DiscreteBayesianNetwork(states=states, cpds=factors)
    else:
        # The edges are inferred from the factor scopes (every pair of nodes in
        # a factor).
        pgm = DiscreteMarkovNetwork(states=states, factors=factors)

    pgm.check_model()

    return pgm


def save_uai(pgm, name, quiet=True):
    """Serialize a CONIN model in UAI format.

    Parameters
    ----------
    pgm : DiscreteBayesianNetwork or DynamicDiscreteBayesianNetwork or DiscreteMarkovNetwork
        Model to write.  A dynamic Bayesian network can only be written if it
        has no dynamic (time-indexed) nodes; unroll it with
        :func:`~conin.graphical_models.create_bn_from_dbn` first.
    name : str
        Output filename. Its extension (if any) is replaced by ``.uai``.
    quiet : bool, optional
        Accepted for API compatibility and currently unused.

    Raises
    ------
    InvalidInputError
        For a dynamic Bayesian network with dynamic nodes.
    """
    text = _uai_text(pgm)
    with open(os.path.splitext(name)[0] + ".uai", "w") as f:
        f.write(text)


def _uai_text(pgm):
    """Return the UAI text that :func:`save_uai` writes for ``pgm``.

    Variables are numbered in ``pgm.nodes`` order and states in
    ``pgm.states[node]`` order.
    """
    if isinstance(pgm, DiscreteBayesianNetwork):
        case = "BAYES"
    elif isinstance(pgm, DynamicDiscreteBayesianNetwork):
        case = "BAYES"
        if pgm.dynamic_nodes:
            raise InvalidInputError(
                "A UAI file cannot hold the time-indexed CPD templates of a "
                "dynamic Bayesian network; unroll it first with "
                "conin.graphical_models.create_bn_from_dbn(dbn=..., start=0, "
                "stop=...) and save the Bayesian network"
            )
    elif isinstance(pgm, DiscreteMarkovNetwork):
        case = "MARKOV"
    else:
        raise ValueError(
            f"pgm must be an instance of a <Dynamic>DiscreteBayesianNetwork or DiscreteMarkovNetwork: {type(pgm)=}"
        )

    # get node2id mapping
    node2id = {n: i for i, n in enumerate(pgm.nodes)}

    # get domain sizes
    sizes = [str(len(pgm.states[n])) for n in pgm.nodes]

    # get factors
    if case == "BAYES":
        factors = [cpd.to_factor() for cpd in pgm.cpds]
    elif case == "MARKOV":
        factors = pgm.factors

    # get function table preamble and values for each vactor
    tables = []
    for factor in factors:
        # table_preamble
        # len(factor.nodes), factor.node[0], factor.node[1], ..., factor.node[n-1]
        factor_node_ids = [str(node2id[n]) for n in factor.nodes]
        table_preamble = [str(len(factor_node_ids))] + factor_node_ids

        # table values
        assignments = list(factor.assignments(pgm.states))
        table_values = []
        for assignment in assignments:
            if len(assignment) == 1:
                state = assignment[0][1]
            else:
                state = tuple([t[1] for t in assignment])
            table_values.append(str(factor.values[state]))

        tables += [(table_preamble, table_values)]

    lines = [
        case,  # network type
        str(len(node2id)),  # num. nodes
        " ".join(sizes),  # node domain sizes
        str(len(tables)),  # num. function tables
    ]
    lines += [" ".join(table_preamble) for table_preamble, _ in tables]

    # Function Tables
    lines.append("")
    for _, table_values in tables:
        lines.append(str(len(table_values)))  # len table
        lines.append(" ".join(table_values))  # table values
    return "\n".join(lines) + "\n"
