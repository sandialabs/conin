import os
import re
import itertools
import numpy as np
import pprint

try:
    import pgmpy.models
    from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
    pgmpy_available=True
except:
    pgmpy_available=False

from conin.bayesian_network import MapCPD


lineno = 0

def get_lines(filename, string):
    if filename:
        assert os.path.exists(filename), f"Cannot read missing UAI file {filename}"
        with open(filename, 'r') as INPUT:
            for line in INPUT:
                yield line
    else:
        for line in string.split('\n'):
            yield line

def tokenize(filename, string):
    global lineno

    lineno = -1
    for line in get_lines(filename, string):
        lineno += 1
        tmp = line.strip()
        if len(tmp) == 0 or tmp[0] == '#':
            continue
        for token in re.split("[ \t]+", tmp):
            yield token
        

def load_pgmpy_model_from_uai(filename=None, string=None, verbose=False):
    global lineno

    tokens = tokenize(filename, string)

    pgmtype = next(tokens)
    if verbose:
        print(f"TYPE {pgmtype}")
    is_bayes = pgmtype == "BAYES"

    if is_bayes:
        pgm = pgmpy.models.DiscreteBayesianNetwork()
    else:
        pgm = pgmpy.models.MarkovNetwork()
    
    # Parse vars
    nvars = int(next(tokens))
    if verbose:
        print(f"NVars {nvars}")
    vcard = [int(next(tokens)) for i in range(nvars)]
    vname = [f"var{i}" for i in range(nvars)]

    # Add nodes to PGM
    if pgmtype == "BAYES":
        pgm.add_nodes_from(vname)
    else:
        pgm.add_nodes_from(vname)
    
    # Parse factor/cpd definitions
    nfun = int(next(tokens))
    if verbose:
        print(f"NFun {nfun}")
    fun = []
    for i in range(nfun):
        n = int(next(tokens))
        fun.append( [int(next(tokens)) for i in range(n)] )

    # Add edges to PGM
    edges = set()
    for f in fun:
        if len(f) == 1:
            continue
        child_name = vname[f[-1]]
        for parent in f[:-1]:
            edges.add( (vname[parent],child_name) )
    pgm.add_edges_from(edges)

    # Parse factor/cpd values
    factors = []
    for f in fun:
        n = int(next(tokens))
        N = 1
        for v in f:
            N *= vcard[v]
        assert n == N, f"Inconsistent function definition at line {lineno}: read {n} but expected {N} from function definition"

        if is_bayes:
            if verbose:
                print("")
                print(f[-1])
            if len(f) == 1:
                #map_values = [float(next(tokens)) for i in range(vcard[f[0]])]
                map_values = {i:float(next(tokens)) for i in range(vcard[f[0]])}
                evidence = None
            elif len(f) == 2:
                map_values = {index:{i:float(next(tokens)) for i in range(vcard[f[-1]])} for index in range(vcard[f[0]]) }
                evidence = [vname[v] for v in f[:-1]]
            else:
                vlist = [list(range(vcard[v])) for v in f[:-1]]
                map_values = {tuple(reversed(index)):{i:float(next(tokens)) for i in range(vcard[f[-1]])} for index in itertools.product( *list(reversed(vlist)) )}
                evidence = [vname[v] for v in f[:-1]]

            if verbose:
                pprint.pprint(map_values)
                print("")
            factors.append(MapCPD(
                variable=vname[f[-1]],
                evidence=evidence,
                values=map_values))
        else:
            if len(f) == 1:
                values = [float(next(tokens)) for i in range(vcard[f[0]])]
            else:
                vlist = [list(range(vcard[v])) for v in f]
                map_values = {index:float(next(tokens)) for index in itertools.product( *vlist )}
                values = [ map_values[tuple(reversed(index))] for index in itertools.product( *list(reversed(vlist)) ) ]

            factors.append(DiscreteFactor(
                variables=[vname[v] for v in f],
                cardinality=[vcard[v] for v in f],
                values=values))

    # Add factors/cpds
    if is_bayes:
        pgm.add_cpds(*factors)
        pgm.check_model()
    else:
        pgm.add_factors(*factors)
        pgm.check_model()

    return pgm


                

