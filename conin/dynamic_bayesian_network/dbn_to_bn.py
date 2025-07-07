try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.models import DiscreteBayesianNetwork
except:
    pass


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
        for u, v in dbn.edges()
    ]
    new_cpds = []
    for cpd in dbn.cpds:
        new_vars = [(var, time + t_slice) for var, time in cpd.variables]
        new_state_names = dict(
            zip(new_vars, [cpd.state_names[var] for var in cpd.variables])
        )
        new_cpds.append(
            TabularCPD(
                variable=new_vars[0],
                variable_card=cpd.cardinality[0],
                values=cpd.get_values(),
                evidence=new_vars[1:],
                evidence_card=cpd.cardinality[1:],
                state_names=new_state_names,
            )
        )

    bn = DiscreteBayesianNetwork(edges)
    bn.add_cpds(*new_cpds)
    return bn


def create_bn_from_dbn(*, dbn, start, stop):
    assert start < stop
    # Initialize the DBN to copy relationships from step 0 to subsequent steps
    dbn.initialize_initial_state()

    bn = get_constant_bn(dbn, start)
    for i in range(start + 1, stop):
        bni = get_constant_bn(dbn, i)

        bn.add_nodes_from([(name, t) for name, t in bni.nodes() if t == i + 1])
        bn.add_edges_from(bni.edges())

        for name, t in bni.nodes():
            node = (name, t)
            if t == i + 1:
                cpd = bni.get_cpds(node)
                if cpd is not None:
                    bn.add_cpds(bni.get_cpds(node))

    bn._pyomo_index_names = {
        (name, t): f"{name}_{t}"
        for t_slice in range(start, stop + 1)
        for name, t in dbn.get_slice_nodes(t_slice)
    }

    return bn
