import math
import pprint
import random
import munch
import copy
import pyomo.environ as pyo
import conin
from conin.inference import viterbi, a_star, lp_inference, ip_inference


class Knapsack(conin.HMMApplication):

    def __init__(self):
        super().__init__(self.__class__.__name__)

    def initialize(self, *, value, weight, capacity, hmm=None):
        assert set(weight.keys()) == set(value.keys())
        items = list(value.keys())
        self._data.items = items
        self._data.value = value
        self._data.weight = weight
        self._data.capacity = capacity
        #
        # Hidden states
        #   (i,flag):       see item i and pickup if flag==True
        #
        hidden_states = [(i, True) for i in items] + \
            [(i, False) for i in items]
        self._hidden_states = hidden_states
        #
        # Observable states
        #
        self._observable_states = items

        if hmm is not None:
            self.hmm = hmm

    def run_simulations(
        self, *, num=1, debug=False, with_observations=False, seed=None
    ):
        if seed is not None:
            random.seed(seed)

        ans = []
        for i in range(num):
            res = munch.Munch()

            # Shuffle the items
            items = copy.copy(self._data.items)
            random.shuffle(items)

            # Iterate through the items, and keep the ones that
            # greedily fit into the knapsack
            weight = 0.0
            hidden = []
            for item in items:
                # WEH - Should we consider modifying the probability of picking up an item?
                # prob = 1.0 / (1.0 + math.exp(-value[item] / weight[item]))
                prob = 1.0
                if (weight + self._data.weight[item] < self._data.capacity) and (
                    random.random() < prob
                ):
                    hidden.append((item, True))
                    weight += self._data.weight[item]
                else:
                    hidden.append((item, False))

            res = munch.Munch(hidden=hidden, index=i)
            if with_observations:
                res.observations = items
            ans.append(res)

        return ans

    def create_hmm(self):
        """
        This method creates the HMM using a bias towards items that have
        high value/weight.
        """
        hidden_states = self._hidden_states
        items = self._data.items
        weight = self._data.weight
        value = self._data.value

        N_ = len(items)

        # The probability of starting in state j is uniform,
        # but our probability of wanting to pickup item j is biased
        # by the ratio value[j]/weight[j].  We use a logistic function to
        # compute the probability of picking up item j.
        start_probs = {h: 0.0 for h in hidden_states}
        for j in items:
            logistic = 1.0 / (1.0 + math.exp(-value[j] / weight[j]))
            start_probs[(j, True)] = logistic / N_
            start_probs[(j, False)] = (1.0 - logistic) / N_
        # pprint.pprint(start_probs)

        # Transition probability from item i to item j is uniform,
        # but our probability of wanting to pickup item j is biased
        # by the ratio value[j]/weight[j].  We use a logistic function to
        # compute the probability of picking up item j.
        transition_probs = {}
        for i in hidden_states:
            for j in hidden_states:
                transition_probs[i, j] = 0.0
        for i in items:
            for j in items:
                logistic = 1.0 / (1.0 + math.exp(-value[j] / weight[j]))
                transition_probs[(i, True), (j, True)] = logistic / N_
                transition_probs[(i, True), (j, False)] = (1.0 - logistic) / N_
                transition_probs[(i, False), (j, True)] = logistic / N_
                transition_probs[(i, False), (j, False)] = (
                    1.0 - logistic) / N_
        # pprint.pprint(transition_probs)

        # We always observe the item that is picked up, but we do not
        # observe if it is stashed in the knapsack.
        emission_probs = {}
        for i in hidden_states:
            for o in items + [None]:
                emission_probs[i, o] = 1.0 if i[0] == o else 0.0
        # pprint.pprint(emission_probs)

        self.hmm = conin.HMM()
        self.hmm.load_model(
            start_probs=start_probs,
            transition_probs=transition_probs,
            emission_probs=emission_probs,
        )


class Knapsack_Oracle(Knapsack, conin.Constrained_HMM):

    def __init__(self):
        Knapsack.__init__(self)
        conin.Constrained_HMM.__init__(self)

    def initialize(self, *, value, weight, capacity):
        # Initialize the Knapsack base class
        super().initialize(value=value, weight=weight, capacity=capacity)

        # Initialize the ConstrainedHMM base class with the HMM from Knapsack
        if self.hmm is not None:
            self.load_model(hmm=self.hmm)

        # Add an oracle constraint
        self.add_constraint(
            conin.Constraint(
                func=lambda hidden: sum(
                    self._data.weight[h[0]] if h[1] else 0.0 for h in hidden
                )
                <= self._data.capacity
            )
        )


class Knapsack_Pyomo(Knapsack, conin.PyomoHMMApplication):

    def __init__(self):
        super().__init__()

    def generate_algebraic_constraints(self, *, observations, constrained=True):
        M = super().generate_algebraic_lp_constraints(observations=observations)
        if not constrained:
            return M

        ITEMS = self._data.items

        M.x = pyo.Var(ITEMS, within=pyo.Binary)

        # Objective defined by LP formulation generated by PyomoHMMApplication
        # M.o = pyo.Objective(expr=sum(self._data.value[i] * M.x[i] for i in ITEMS))

        M.c = pyo.Constraint(
            expr=sum(self._data.weight[i] * M.x[i] for i in ITEMS)
            <= self._data.capacity
        )

        def hmm_c_(m, i):
            i_ = self.hmm.hidden_to_internal[i, True]
            return M.x[i] == sum(M.hmm.x[t, i_] for t in range(len(observations)))

        M.hmm_c = pyo.Constraint(ITEMS, rule=hmm_c_)

        return M


app = Knapsack_Oracle()
app.initialize(
    value={"a": 5.0, "b": 3.0, "c": 2.0, "d": 7.0, "e": 4.0},
    weight={"a": 2.0, "b": 8.0, "c": 4.0, "d": 2.0, "e": 5.0},
    capacity=10.0,
)

simulations = app.run_simulations(seed=123456789, with_observations=True)
hidden = simulations[0].hidden
observations = simulations[0].observations
print()
if app.is_feasible(hidden):
    print(f"Feasible hidden states: {hidden}")
else:
    print(f"Infeasible hidden states: {hidden}")

app.initialize_hmm_from_simulations(
    seed=123456789, emission_tolerance=0.0, num=1000)
print()
print("-" * 60)
print("HMM Parameters")
print("-" * 60)
pprint.pprint(app.hmm.to_dict(), indent=4, sort_dicts=True)

results = viterbi(observed=observations, statistical_model=app)
print()
print("-" * 60)
print("Inference with Viterbi (unconstrained)")
print("-" * 60)
print(f"Observations:    {results.observations}")
print(f"Inferred Hidden: {results.solutions[0].hidden}")
print(f"Log Likelihood:  {results.solutions[0].log_likelihood}")

results = a_star(observed=observations, statistical_model=app)
print()
print("-" * 60)
print("Inference with A* (constrained oracle)")
print("-" * 60)
print(f"Observations:    {results.observations}")
print(f"Inferred Hidden: {results.solutions[0].hidden}")
print(f"Log Likelihood:  {results.solutions[0].log_likelihood}")

pyapp = Knapsack_Pyomo()
pyapp.initialize(
    value={"a": 5.0, "b": 3.0, "c": 2.0, "d": 7.0, "e": 4.0},
    weight={"a": 2.0, "b": 8.0, "c": 4.0, "d": 2.0, "e": 5.0},
    capacity=10.0,
    hmm=app.hmm,
)

results = lp_inference(observed=observations, statistical_model=pyapp)
print()
print("-" * 60)
print("Inference with LP (unconstrained optimization)")
print("-" * 60)
print(f"Observations:    {results.observations}")
print(f"Inferred Hidden: {results.solutions[0].hidden}")
print(f"Log Likelihood:  {results.solutions[0].log_likelihood}")

results = ip_inference(observed=observations, statistical_model=pyapp)
print()
print("-" * 60)
print("Inference with IP (constrained optimization)")
print("-" * 60)
print(f"Observations:    {results.observations}")
print(f"Inferred Hidden: {results.solutions[0].hidden}")
print(f"Log Likelihood:  {results.solutions[0].log_likelihood}")
print("Nonzero solution variables:")
pprint.pprint(results.solutions[0].variables, sort_dicts=True, indent=4)
