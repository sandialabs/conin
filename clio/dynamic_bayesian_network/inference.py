from clio.bayesian_network import pyomo_BN_map_query


def create_bn_from_dbn(*, dbn, start, stop):
    assert start < stop
    bn = dbn.get_constant_bn(start)
    for i in range(start + 1, stop):
        bni = dbn.get_constant_bn(i)

        bn.add_nodes_from(bni.nodes())
        bn.add_edges_from(bni.edges())

        for node in bni.nodes():
            bn.add_cpds(bni.get_cpds(node))

    bn._pyomo_node_index = {
        (name, t): f"{name}_{t}"
        for t_slice in range(start, stop + 1)
        for name, t in dbn.get_slice_nodes(t_slice)
    }
    return bn


def pyomo_DBN_map_query(*, pgm, start=0, stop=1, variables=None, evidence=None):
    bn = create_bn_from_dbn(dbn=pgm, start=start, stop=stop)
    return pyomo_BN_map_query(pgm=bn, variables=variables, evidence=evidence)
