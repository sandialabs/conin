import pytest
import unittest

from conin.dynamic_bayesian_network.sampling import Sampler
import conin.dynamic_bayesian_network.tests.examples as tc


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.G = tc.create_ddbn0()

    def test_sampler(self):
        sampler = Sampler(self.G)
        assert sampler is not None

    def test_set_time(self):
        # @t==0
        sampler = Sampler(self.G)
        assert(len(sampler.representation['A']['parents'])==0)

        # @t!=0
        sampler.set_time(3)
        assert(len(sampler.representation['A']['parents'])==3)
        sampler.set_time(4)
        assert(len(sampler.representation['A']['parents'])==3)        

        # @t==0 (reset)
        sampler.set_time(0)
        assert(len(sampler.representation['A']['parents'])==0)

    def test_get_cpd(self):
        # @t==0
        sampler = Sampler(self.G)
        cpd = sampler.get_cpd('A')
        assert(cpd.node==('A', 0))
        assert(cpd.values=={'a':0.7, 'b':0.1, 'c':0.2})

        # @t!=0
        sampler.set_time(1)
        cpd = sampler.get_cpd('A')
        assert(cpd.node[0]=='A')
        assert(cpd.node[1].value()==sampler.G.t.value())
        assert(cpd.values[('a', 0, 'T')]=={'a':0.7, 'b':0.2, 'c':0.1})

    def test_get_cpd_map(self):
        sampler = Sampler(self.G)
        cpd_map = sampler.get_cpd_map()
        assert(cpd_map['X'].node=='X')
        assert(cpd_map['A'].node[0]=='A')
        assert(cpd_map['B'].node[0]=='B')
        assert(cpd_map['C'].node[0]=='C')

    def test_extract_parents(self):
        sampler = Sampler(self.G)
        parents = sampler.extract_parents(sampler.G.cpds[2])
        assert(parents==[('A', False), ('C', False)])

    def test_get_children(self):
        sampler = Sampler(self.G)
        children = sampler.get_children('A')
        assert(children==['B'])

    def test_build_representation(self):
        sampler = Sampler(self.G)
        representation = sampler.build_representation()
        assert(representation['X']=={
            'parents': [], 
            'values': {'T': 0.9, 'F': 0.1}, 
            'children': []
        })
        assert(representation['A']=={
            'parents': [],
            'values': {'a': 0.7, 'b': 0.1, 'c': 0.2},
            'children': ['B']
        })
        assert(representation['B']=={
            'parents': [('A', False), ('C', False)],
            'values': {('a', 't'): {0: 0.2, 1: 0.3, 2: 0.3, 3: 0.2},
                       ('a', 'f'): {0: 0.2, 1: 0.4, 2: 0, 3: 0.4},
                       ('b', 't'): {0: 0.6, 1: 0.2, 2: 0.1, 3: 0.1},
                       ('b', 'f'): {0: 0.8, 1: 0, 2: 0.1, 3: 0.1},
                       ('c', 't'): {0: 0.3, 1: 0.3, 2: 0.1, 3: 0.3},
                       ('c', 'f'): {0: 0, 1: 0.1, 2: 0.2, 3: 0.7}},
            'children': []
        })
        assert(representation['C']=={
            'parents': [], 
            'values': {'t': 0.5, 'f': 0.5}, 
            'children': ['B']            
        })

    def test_sample_next_states_bad_init(self):
        sampler = Sampler(self.G)
        with pytest.raises(
            ValueError, 
            match="G.t==0 but previous_states is non-empty"
        ):
            sampler.sample_next_states(previous_states={'A':.7})

    def test_sample_next_states_values(self):
        sampler = Sampler(self.G)
        next_states = sampler.sample_next_states()
        assert(next_states['A'] in sampler.G.dynamic_states['A'])
        assert(next_states['B'] in sampler.G.dynamic_states['B'])
        assert(next_states['C'] in sampler.G.dynamic_states['C'])
        assert(next_states['X'] in sampler.G.states['X'])        

    def test_sample_next_states_transition(self):
        sampler = Sampler(self.G)
        sampler.set_time(1)
        test_values = []
        for i in range(100):
            next_states = sampler.sample_next_states(previous_states={
                'A':'b', 'B':3, 'C':'t', 'X':'F'
            })
            test_values.append(next_states['A'])
        assert(set(test_values)=={'c'})

    def test_sample(self):
        sampler = Sampler(self.G)
        runs = sampler.sample(T=6, N=8)
        assert(len(runs)==8)
        for k, trace in runs.items():
            assert(len(trace)==6)
    
    def tearDown(self):
        self.G = None