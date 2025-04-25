from conin.bayesian_network import create_BN_map_query_model


def create_bn_from_dbn(*, dbn, start, stop):
    assert start < stop
    bn = dbn.get_constant_bn(start)
    for i in range(start + 1, stop):
        bni = dbn.get_constant_bn(i)

        bn.add_nodes_from([node for node in bni.nodes() if node.endswith(f"_{i+1}")])
        bn.add_edges_from(bni.edges())

        for node in bni.nodes():
            if node.endswith(f"_{i+1}"):
                bn.add_cpds(bni.get_cpds(node))

    bn._pyomo_node_index = {
        (name, t): f"{name}_{t}"
        for t_slice in range(start, stop + 1)
        for name, t in dbn.get_slice_nodes(t_slice)
    }
    return bn


def create_DBN_map_query_model(*, pgm, start=0, stop=1, variables=None, evidence=None):
    bn = create_bn_from_dbn(dbn=pgm, start=start, stop=stop)
    return create_BN_map_query_model(pgm=bn, variables=variables, evidence=evidence)
