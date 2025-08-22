from conin.util import try_import

with try_import() as pgmpy_available:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DiscreteBayesianNetwork


"""
The MIT License (MIT)

Copyright (c) 2013-2024 pgmpy

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def get_constant_bn(dbn, t_slice=0):
    """
    Returns a normal Bayesian Network object which has nodes from the first two
    time slices and all the edges in the first time slice and edges going from
    first to second time slice. The returned Bayesian Network basically represents
    the part of the DBN which remains constant.

    This is adapted from the pgmpy DynamicBayesianNetwork.get_constant_bn() method.
    The node namers are *not* changed to strings here.
    """
    edges = [
        (
            (u[0], u[1] + t_slice),
            (v[0], v[1] + t_slice),
        )
        for u, v in dbn.edges
    ]

    new_cpds = []
    new_states = {}

    for cpd in dbn.cpds:
        new_vars = [(var, time + t_slice) for var, time in cpd.variables]

        new_states.update( dict(
            zip(new_vars, [dbn.state_names[var] for var in cpd.variables])
        ) )

        new_cpds.append(
            DiscreteCPD(
                variable=new_vars[0],
                evidence=new_vars[1:],
                values=cpd.values,
            )
        )

    bn = DiscreteBayesianNetwork()
    bn.edges = edges
    bn.states = new_states
    bn.cpds = new_cpds

    return bn


def create_bn_from_dbn(*, dbn, start, stop):
    assert start < stop
    # Initialize the DBN to copy relationships from step 0 to subsequent steps
    #dbn.initialize_initial_state()

    bn = get_constant_bn(dbn, start)
    states = bn.states
    edges = bn.edges
    cpds = {cpd.variable : cpd for cpd in bn.cpds}
    _pyomo_index_names = {(name, t): f"{name}_{t}" for name,t in bn.nodes}

    for i in range(start + 1, stop):
        bni = get_constant_bn(dbn, i)

        states.update( bni.states )
        edges.extend( bni.edges )
        _pyomo_index_names = {(name, t): f"{name}_{t}" for name,t in bn.nodes}

        for name, t in bni.nodes:
            node = (name, t)
            if t == i + 1:
                cpd = cpds.get(node, None)
                if cpd is not None:
                    cpds[cpd.variable] = cpd

    bn = DiscreteBayesianNetwork()
    bn.states = states
    bn.edges = edges
    bn.cpds = cpds
    bn._pyomo_index_names = _pyomo_index_names

    return bn
