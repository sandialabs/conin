import os
import re
import itertools
import numpy as np
import pprint


from conin.markov_network import DiscreteFactor, DiscreteMarkovNetwork
from conin.bayesian_network import DiscreteCPD, DiscreteBayesianNetwork


lineno = 0


def get_lines(filename, string):
    if filename:
        assert os.path.exists(filename), f"Cannot read missing UAI file {filename}"
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


def load_conin_model_from_uai(filename=None, string=None, verbose=False):
    global lineno
    lineno = -1

    tokens = tokenize(filename, string)

    pgmtype = next(tokens)
    if verbose:
        print(f"TYPE {pgmtype}")
    is_bayes = pgmtype == "BAYES"

    if is_bayes:
        pgm = DiscreteBayesianNetwork()
    else:
        pgm = DiscreteMarkovNetwork()

    # Parse vars
    nvars = int(next(tokens))
    if verbose:
        print(f"NVars {nvars}")
    vcard = [int(next(tokens)) for i in range(nvars)]
    vname = [f"var{i}" for i in range(nvars)]

    # Add states
    pgm.states = {vname[i]: list(range(vcard[i])) for i in range(nvars)}
    import pprint

    pprint.pprint(pgm.states)

    # Add nodes to PGM
    # if pgmtype == "BAYES":
    #    pgm.add_nodes_from(vname)
    # else:
    #    pgm.add_nodes_from(vname)

    # Parse factor/cpd definitions
    nfun = int(next(tokens))
    if verbose:
        print(f"NFun {nfun}")
    fun = []
    for i in range(nfun):
        n = int(next(tokens))
        fun.append([int(next(tokens)) for i in range(n)])

    # Add edges to MNs
    if not is_bayes:
        edges = set()
        for f in fun:
            if len(f) == 1:
                continue
            child_name = vname[f[-1]]
            for parent in f[:-1]:
                edges.add((vname[parent], child_name))
        pgm.edges = edges

    # Parse factor/cpd values
    factors = []
    for f in fun:
        n = int(next(tokens))
        N = 1
        for v in f:
            N *= vcard[v]
        assert (
            n == N
        ), f"Inconsistent function definition at line {lineno}: read {n} but expected {N} from function definition"

        if is_bayes:
            if verbose:
                print("")
                print(f[-1])
            if len(f) == 1:
                # map_values = [float(next(tokens)) for i in range(vcard[f[0]])]
                map_values = {i: float(next(tokens)) for i in range(vcard[f[0]])}
                evidence = None
            elif len(f) == 2:
                map_values = {
                    index: {i: float(next(tokens)) for i in range(vcard[f[-1]])}
                    for index in range(vcard[f[0]])
                }
                evidence = [vname[v] for v in f[:-1]]
            else:
                vlist = [list(range(vcard[v])) for v in f[:-1]]
                map_values = {
                    tuple(reversed(index)): {
                        i: float(next(tokens)) for i in range(vcard[f[-1]])
                    }
                    for index in itertools.product(*list(reversed(vlist)))
                }
                evidence = [vname[v] for v in f[:-1]]

            if verbose:
                pprint.pprint(map_values)
                print("")
            factors.append(
                DiscreteCPD(variable=vname[f[-1]], evidence=evidence, values=map_values)
            )
        else:
            if len(f) == 1:
                values = [float(next(tokens)) for i in range(vcard[f[0]])]
            else:
                vlist = [list(range(vcard[v])) for v in f]
                map_values = {
                    index: float(next(tokens)) for index in itertools.product(*vlist)
                }
                values = [
                    map_values[tuple(reversed(index))]
                    for index in itertools.product(*list(reversed(vlist)))
                ]

            factors.append(
                DiscreteFactor(
                    nodes=[vname[v] for v in f],
                    # cardinality=[vcard[v] for v in f],
                    values=values,
                )
            )

    # Add factors/cpds
    if is_bayes:
        pgm.cpds = factors
    else:
        pgm.factors = factors
    pgm.check_model()

    return pgm
