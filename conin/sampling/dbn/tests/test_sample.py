import pytest
import unittest
import numpy as np

from conin.sampling.dbn.sample import (
    _get_cpd_tensor, _get_representation,
    _get_topological_order, _get_arrays, 
    _sample_step, _sample
)
import conin.sampling.dbn.tests.examples as tc


class TestSample(unittest.TestCase):
    def setUp(self):
        self.G = tc.create_ddbn0()

    def test_get_cpd_tensor(self):
        target = np.array(
            [[[0.2, 0.3, 0.3, 0.2],
              [0.2, 0.4, 0. , 0.4]],
             
             [[0.6, 0.2, 0.1, 0.1],
              [0.8, 0. , 0.1, 0.1]],
             
             [[0.3, 0.3, 0.1, 0.3],
              [0. , 0.1, 0.2, 0.7]]]
        )
        assert np.array_equal(_get_cpd_tensor(self.G, self.G.cpds[2]), target)

    def test_get_representation(self):
        targets = {
            'X':{
                'states':self.G.states['X'],
                'parents':[],
                'cpd':_get_cpd_tensor(self.G, self.G.cpds[4]),
            },
            'A.0':{
                'states':self.G.dynamic_states['A'],
                'parents':[],
                'cpd':_get_cpd_tensor(self.G, self.G.cpds[0])
            },
            'A.t':{
                'states':self.G.dynamic_states['A'],
                'parents':['A.t-1', 'B.t-1', 'X'],
                'cpd':_get_cpd_tensor(self.G, self.G.cpds[1])
            },
            'B.t':{
                'states':self.G.dynamic_states['B'],
                'parents':['A.t', 'C.t'],
                'cpd':_get_cpd_tensor(self.G, self.G.cpds[2])
            },
            'C.t':{
                'states':self.G.dynamic_states['C'],
                'parents':[],
                'cpd':_get_cpd_tensor(self.G, self.G.cpds[3])
            }
        }
        for ((k,v), (_k,_v)) in zip(_get_representation(self.G).items(), targets.items()):
            assert k==_k
            assert v['states']==_v['states']
            assert v['parents']==_v['parents']
            assert np.array_equal(v['cpd'], _v['cpd'])

    def test_get_topological_order_init(self):
        rep = _get_representation(self.G)
        nodes = list(dict.fromkeys(
            [k[:k.rindex('.')] if '.' in k else k for k in rep.keys()]
        ))        
        order = _get_topological_order(rep, init=True)
        assert set(order)=={0, 1, 2, 3}
        assert nodes[order[-1]]=='B'

    def test_get_topological_order_noinit(self):
        rep = _get_representation(self.G)
        nodes = list(dict.fromkeys(
            [k[:k.rindex('.')] if '.' in k else k for k in rep.keys()]
        ))        
        order = _get_topological_order(rep)
        assert [nodes[k] for k in order]==['C', 'A', 'B']

    def test_get_arrays_init(self):
        rep = _get_representation(self.G)
        nodes = list(dict.fromkeys(
            [k[:k.rindex('.')] if '.' in k else k for k in rep.keys()]
        ))         
        cpd_arrays, index_arrays = _get_arrays(rep, init=True)
        assert(index_arrays[nodes.index('X')]==[])
        assert(index_arrays[nodes.index('A')]==[])
        assert(set(index_arrays[nodes.index('B')])=={nodes.index('A'), nodes.index('C')})
        assert(index_arrays[nodes.index('C')]==[])
        assert(np.array_equal(np.squeeze(cpd_arrays[nodes.index('X')]), rep['X']['cpd']))  # squeeze if no parents
        assert(np.array_equal(np.squeeze(cpd_arrays[nodes.index('A')]), rep['A.0']['cpd']))
        assert(np.array_equal(cpd_arrays[nodes.index('B')], rep['B.t']['cpd']))
        assert(np.array_equal(np.squeeze(cpd_arrays[nodes.index('C')]), rep['C.t']['cpd']))
    
    def test_get_arrays_noinit(self):
        rep = _get_representation(self.G)
        nodes = list(dict.fromkeys(
            [k[:k.rindex('.')] if '.' in k else k for k in rep.keys()]
        ))
        cpd_arrays, index_arrays = _get_arrays(rep, init=False)
        assert(index_arrays[nodes.index('X')]==[])
        assert(set(index_arrays[nodes.index('A')])==
               {nodes.index('X'), nodes.index('A'), nodes.index('B')})
        assert(set(index_arrays[nodes.index('B')])==
               {nodes.index('A'), nodes.index('C')})
        assert(index_arrays[nodes.index('C')]==[])
        assert(np.array_equal(np.squeeze(cpd_arrays[nodes.index('X')]), rep['X']['cpd']))  # squeeze if no parents
        assert(np.array_equal(cpd_arrays[nodes.index('A')], rep['A.t']['cpd']))
        assert(np.array_equal(cpd_arrays[nodes.index('B')], rep['B.t']['cpd']))
        assert(np.array_equal(np.squeeze(cpd_arrays[nodes.index('C')]), rep['C.t']['cpd']))

    def test_sample_step(self):
        rep = _get_representation(self.G)
        nodes = list(dict.fromkeys(
            [k[:k.rindex('.')] if '.' in k else k for k in rep.keys()]
        ))
        order = _get_topological_order(rep, init=True)
        cpd_arrays, index_arrays = _get_arrays(rep, init=True)
        init_states = np.zeros((10, len(nodes))).astype(int)
        states = _sample_step(order, init_states, cpd_arrays, index_arrays)
        assert(states.shape==(10, 4))

    def test_sample(self):
        traces = _sample(self.G)
        assert(traces.shape==(10, 10, 4))

    def tearDown(self):
        self.G = None