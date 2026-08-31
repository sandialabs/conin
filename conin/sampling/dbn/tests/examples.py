from conin.dynamic_bayesian_network import DynamicDiscreteBayesianNetwork
from conin.bayesian_network.model import DiscreteCPD


def create_ddbn0():
    G = DynamicDiscreteBayesianNetwork()
    
    states = {
      'X':['T', 'F']
    }
    G.states = states
    
    dynamic_states = {
      'A':['a', 'b', 'c'],
      'B':[0, 1, 2, 3],
      'C':['t', 'f']
    }
    G.dynamic_states = dynamic_states
    
    cpd_start_A = DiscreteCPD(
      node = ('A', 0), 
      values = [.7, .1, .2]
    )
    cpd_trans_A = DiscreteCPD(
      node = ('A', G.t),
      parents = [('A', G.t-1), ('B', G.t-1), 'X'],
      values = {
          ('a', 0, 'T'):[.7, .2, .1], ('a', 0, 'F'):[.2, .1, .7],
          ('a', 1, 'T'):[.1, .1, .8], ('a', 1, 'F'):[.9, 0, .1],
          ('a', 2, 'T'):[.3, .3, .4], ('a', 2, 'F'):[.4, .4, .2],
          ('a', 3, 'T'):[.6, .1, .3], ('a', 3, 'F'):[.5, .4, .1],
          ('b', 0, 'T'):[.2, .2, .6], ('b', 0, 'F'):[.6, 0, .4],
          ('b', 1, 'T'):[.3, .4, .3], ('b', 1, 'F'):[0, .9, .1],
          ('b', 2, 'T'):[.5, 0, .5], ('b', 2, 'F'):[.7, .3, 0],
          ('b', 3, 'T'):[.8, .1, .1], ('b', 3, 'F'):[0, 0, 1.],
          ('c', 0, 'T'):[.3, .5, .2], ('c', 0, 'F'):[.2, .3, .5],
          ('c', 1, 'T'):[.1, .7, .2], ('c', 1, 'F'):[.2, .6, .2],
          ('c', 2, 'T'):[0, .5, .5], ('c', 2, 'F'):[.6, .2, .2],
          ('c', 3, 'T'):[.1, .8, .1], ('c', 3, 'F'):[.8, .1, .1]
      }
    )
    cpd_B = DiscreteCPD(
        node = ('B', G.t),
        parents = [('A', G.t), ('C', G.t)],
        values = {
            ('a', 't'):[.2, .3, .3, .2], ('a', 'f'):[.2, .4, 0, .4],
            ('b', 't'):[.6, .2, .1, .1], ('b', 'f'):[.8, 0, .1, .1],
            ('c', 't'):[.3, .3, .1, .3], ('c', 'f'):[0, .1, .2, .7]
        }
    )
    cpd_C = DiscreteCPD(
        node = ('C', G.t),
        values = [.5, .5]
    )
    cpd_X = DiscreteCPD(
        node = 'X',
        values = [.9, .1]
    )
    G.cpds = [cpd_start_A, cpd_trans_A, cpd_B, cpd_C, cpd_X]
    
    #print(G.check_model())
    return G