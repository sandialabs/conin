import conin.markov_network


def create_mn_from_bn(bn):
    pgm = conin.markov_network.DiscreteMarkovNetwork(
        states=bn.states, factors=[cpd.to_factor() for cpd in bn.cpds]
    )
    pgm.check_model()

    return pgm
