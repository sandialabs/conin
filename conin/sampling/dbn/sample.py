import numpy as np
import itertools
from collections import defaultdict, deque
from conin.dynamic_bayesian_network.expr import ExpressionNode, MinusNode


def _get_cpd_tensor(G, cpd):
    """
    Convert the CPD from graph G into a tensor representation
    Format:  cpd_tensor[conditioning_state_index(ices), conditioned_state_index] = p
    """
    if cpd.parents is not None:
        parents = [p[0] for p in cpd.parents]
    else:
        parents = []
    node = cpd.node[0]
    state_values = [
        G.states[n] if n in G.states else G.dynamic_states[n] for n in parents + [node]
    ]
    tensor_shape = [len(sv) for sv in state_values]
    if len(tensor_shape) == 1:
        tensor_shape = tensor_shape[0]
    cpd_tensor = np.zeros(tensor_shape)
    for combination in itertools.product(*state_values):
        idx = tuple((sv.index(c) for sv, c in zip(state_values, combination)))
        if len(idx) == 1:
            cpd_tensor[idx[0]] = cpd.values[combination[0]]
        else:
            cpd_tensor[idx] = cpd.values[combination[:-1]][combination[-1]]

    return cpd_tensor


def _get_representation(G):
    """
    Get the dictionary representation of the nodes of G:
    representation = {
        <node>: {
          'states': <states>,
          'parents': <parents>,
          'cpd': <cpd_tensor>
        }
    }
    Dynamic nodes with initialization, e.g. 'A.0' vs 'A.t', will  have separate entries
    """
    representation = {}
    for node in G.states:
        for cpd in G.cpds:
            if cpd.node == node:
                representation[node] = {
                    "states": G.states[node],
                    "parents": [],
                    "cpd": _get_cpd_tensor(G, cpd),
                }
                break  # there should be exactly one entry
    for node in G.dynamic_states:
        for cpd in G.cpds:
            # check to see if we have an initial version
            if (
                isinstance(cpd.node, tuple)
                and cpd.node[0] == node
                and isinstance(cpd.node[1], int)
            ):
                representation[node + ".0"] = {
                    "states": G.dynamic_states[node],
                    "parents": [],
                    "cpd": _get_cpd_tensor(G, cpd),
                }
            elif (
                isinstance(cpd.node, tuple)
                and cpd.node[0] == node
                and isinstance(cpd.node[1], ExpressionNode)
            ):
                # for each dynamic node parent, we need to check whether it's a current or previous-time node
                parents = []
                if cpd.parents is not None:
                    for parent in cpd.parents:
                        if isinstance(parent, tuple) and isinstance(
                            parent[1], MinusNode
                        ):
                            parents.append(parent[0] + ".t-1")
                        elif isinstance(parent, tuple):
                            parents.append(parent[0] + ".t")
                        else:
                            parents.append(parent)
                representation[node + ".t"] = {
                    "states": G.dynamic_states[node],
                    "parents": parents,
                    "cpd": _get_cpd_tensor(G, cpd),
                }
    if len(representation) != len(G.cpds):
        raise ValueError(f"We have an CPD count mismatch - are there multiple entries?")

    return representation


def _get_topological_order(representation, init=False):
    """
    Given G and a dictionary representation of G, find the topological sort order
    """
    nodes = list(
        dict.fromkeys(  # preserve order
            [k[: k.rindex(".")] if "." in k else k for k in representation.keys()]
        )
    )
    static_nodes = [
        k for k in representation.keys() if not (k.endswith(".0") or k.endswith(".t"))
    ]
    indeg = {n: 0 for n in nodes}
    children = defaultdict(list)
    for node in indeg.keys():
        if init and node + ".0" in representation:  # initial dynamic node
            pass  # no parents
        elif node + ".t" in representation:  # dynamic node
            for parent in representation[node + ".t"]["parents"]:
                if not parent.endswith(".t-1"):  # pass over edges from previous time
                    indeg[node] += 1
                    if parent.endswith(".t"):
                        children[parent[: parent.rindex(".")]].append(node)
                    else:
                        children[parent].append(node)

    q = deque([n for n, d in indeg.items() if d == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # remove static nodes from order if not initializing
    if not init:
        order = [n for n in order if not n in static_nodes]

    return [nodes.index(n) for n in order]


def _get_arrays(representation, init=False):
    """
    Helper function to get cpd and index arrays for fancy indexing
    """
    nodes = list(
        dict.fromkeys(
            [k[: k.rindex(".")] if "." in k else k for k in representation.keys()]
        )
    )

    cpd_array = []
    index_array = []
    for node in nodes:
        if init and node + ".0" in representation:
            rep = representation[node + ".0"]
        elif node + ".t" in representation:
            rep = representation[node + ".t"]
        else:
            rep = representation[node]

        cpd = rep["cpd"]
        if len(cpd.shape) == 1:  # nodes with no parents
            cpd = np.reshape(cpd, (1, len(cpd)))
        cpd_array.append(cpd)
        parents = [
            p[: p.rindex(".")] if (p.endswith(".t") or p.endswith(".t-1")) else p
            for p in rep["parents"]
        ]
        index_array.append([nodes.index(p) for p in parents])

    return cpd_array, index_array


def _sample_step(order, states, cpd_arrays, index_arrays):
    for n in order:
        cpd_array = cpd_arrays[
            n
        ]  # d_conditioning_0, ... d_conditioning_k-1, d_conditioned
        index_array = index_arrays[n]  # k
        indices = states[:, index_array]  # N, k
        cpds = cpd_array[
            tuple(indices[:, i] for i in range(indices.shape[1]))
        ]  # N, d_conditioned
        if indices.shape[1] == 0:
            cpds = np.tile(cpds, (indices.shape[0], 1))

        vals = np.random.uniform(size=cpds.shape[0])
        temp = np.zeros_like(vals)
        idx = np.zeros_like(vals).astype(int)
        for i in range(cpds.shape[1]):
            temp += cpds[:, i]
            idx += (temp >= vals) * (idx == 0) * (i + 1)
        states[:, n] = idx - 1
    return states


def _sample(G, N=10, T=10, return_indices=False):
    """
    Sample N traces of length T from G
    """
    representation = _get_representation(G)
    nodes = list(
        dict.fromkeys(
            [k[: k.rindex(".")] if "." in k else k for k in representation.keys()]
        )
    )
    traces = np.zeros((N, T, len(nodes))).astype(int)

    # t == 0
    order_0 = _get_topological_order(representation, init=True)
    cpd_arrays_0, index_arrays_0 = _get_arrays(representation, init=True)
    states = np.zeros((N, len(nodes))).astype(int)  # blank slate
    traces[:, 0, :] = _sample_step(order_0, states, cpd_arrays_0, index_arrays_0)

    # t > 0
    order_t = _get_topological_order(representation)
    cpd_arrays_t, index_arrays_t = _get_arrays(representation)
    for t in range(1, T):
        traces[:, t, :] = _sample_step(
            order_t, traces[:, t - 1, :].copy(), cpd_arrays_t, index_arrays_t
        )

    if return_indices:
        return traces
    else:
        states = G.states | G.dynamic_states
        traces = np.stack(
            [np.array(states[n])[traces[:, :, i]] for i, n in enumerate(nodes)], -1
        )
        return traces
