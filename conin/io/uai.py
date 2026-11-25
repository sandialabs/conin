import os
import re
import itertools
import pprint
from collections import defaultdict

from conin.exceptions import InvalidInputError, ModelFileError
from conin.graphical_models import DiscreteFactor, DiscreteMarkovNetwork
from conin.graphical_models import DiscreteCPD, DiscreteBayesianNetwork
from conin.graphical_models import DynamicDiscreteBayesianNetwork

from .text import read_text

lineno = 0


def get_lines(filename, string):
    if filename:
        string = read_text(filename, missing="Cannot read missing UAI file")
    yield from string.split("\n")


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
    """The UAI tokens; running out raises ``ModelFileError`` (a truncated file)."""

    def __init__(self, tokens):
        self._tokens = tokens

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._tokens)
        except StopIteration:
            raise ModelFileError(
                f"Unexpected end of the UAI data after line {lineno + 1}"
            ) from None


def load_uai(filename=None, string=None, verbose=False):
    """Load a CONIN model from a UAI file or string.

    Parameters
    ----------
    filename : str, optional
        Path of a ``.uai`` file (gzipped if it ends with ``.gz``).
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
    FileNotFoundError
        If ``filename`` does not exist.
    ModelFileError
        If the file cannot be read (not UTF-8, or a bad ``.gz`` file), or the
        data is malformed (e.g. it ends early, a function table has
        the wrong length or a number cannot be read) or does not describe a
        valid model.
    """
    try:
        return _load_uai(filename, string, verbose)
    except ModelFileError:
        raise
    except (ValueError, IndexError, InvalidInputError) as e:
        raise ModelFileError(f"Invalid UAI data: {e}") from e


def _load_uai(filename, string, verbose):
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
            raise ModelFileError(
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

    # Add factors/cpds; the constructors validate the model
    try:
        if is_bayes:
            return DiscreteBayesianNetwork(states=states, cpds=factors)
        # The edges are inferred from the factor scopes (every pair of nodes in
        # a factor).
        return DiscreteMarkovNetwork(states=states, factors=factors)
    except InvalidInputError as e:
        raise ModelFileError(f"Invalid UAI model: {e}") from e


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
    TypeError
        For another kind of model.

    Notes
    -----
    A UAI file has no variable names.  The variables are numbered in
    ``pgm.nodes`` order, except that if the nodes are ``var0`` ...
    ``var{n-1}`` (the names :func:`load_uai` gives), ``var{i}`` is variable
    ``i``, so saving a loaded file keeps its variable numbers.
    """
    text = _uai_text(pgm, keep_loaded_numbers=True)
    with open(os.path.splitext(name)[0] + ".uai", "w") as f:
        f.write(text)


def _uai_variable_order(nodes):
    """The order in which :func:`save_uai` numbers the variables.

    If the nodes are ``var0`` ... ``var{n-1}``, the names :func:`load_uai`
    gives, ``var{i}`` is variable ``i``, so saving a loaded file keeps its
    variable numbers.  (``nodes`` are sorted as strings, so for more than 10
    variables their order puts ``var10`` before ``var2``.)  Otherwise the
    variables are numbered in ``nodes`` order.
    """
    names = [f"var{i}" for i in range(len(nodes))]
    if set(nodes) == set(names):
        return names
    return list(nodes)


def _uai_text(pgm, keep_loaded_numbers=False):
    """Return the UAI text of ``pgm``.

    Variables are numbered in ``pgm.nodes`` order (which the toulbar2
    representation relies on), or in :func:`_uai_variable_order` if
    ``keep_loaded_numbers`` (what :func:`save_uai` writes), and states in
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
        raise TypeError(
            "A UAI file holds a DiscreteBayesianNetwork, a "
            "DynamicDiscreteBayesianNetwork without time-indexed CPDs or a "
            f"DiscreteMarkovNetwork, got {type(pgm).__name__}"
        )

    nodes = _uai_variable_order(pgm.nodes) if keep_loaded_numbers else pgm.nodes

    # get node2id mapping
    node2id = {n: i for i, n in enumerate(nodes)}

    # get domain sizes
    sizes = [str(len(pgm.states[n])) for n in nodes]

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
            table_values.append(str(factor.values.get(state, factor.default_value)))

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
