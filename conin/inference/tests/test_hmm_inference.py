import pytest
import pyomo.opt
import random

from conin.util import try_import

with try_import() as or_topas_available:
    import or_topas.aos as aos

from conin.inference.recursive import recursive_a_star
from conin.constrained_models import ConstrainedHiddenMarkovModel
from conin.inference.oracle import a_star
from conin.constraints.oracle_library import *

from conin.inference.pyomo import ip_inference

import conin.examples.hidden_markov_model as tc

mip_solver = pyomo.opt.check_available_solvers("gurobi", "highs", "glpk")
mip_solver = mip_solver[0] if mip_solver else None

# TODO IP Testing
# TODO Viterbi/LP Testing


@pytest.fixture
def hmm():
    return tc.create_hmm1()


@pytest.fixture
def chmm1_pyomo():
    return tc.create_chmm1_pyomo()


@pytest.fixture
def chmm1_pyomo_aos():
    return tc.create_chmm1_pyomo_aos()


@pytest.fixture
def chmm2_pyomo_aos():
    return tc.create_chmm2_pyomo_aos()


@pytest.fixture
def lb():
    return 10


@pytest.fixture
def ub():
    return 12


@pytest.fixture
def constraints(lb, ub):
    num_zeros_lb = has_minimum_number_of_occurences_constraint(val="h0", count=lb)
    num_zeros_ub = has_maximum_number_of_occurences_constraint(val="h0", count=ub)
    return [num_zeros_lb, num_zeros_ub]


@pytest.fixture
def chmm(hmm, constraints):

    return ConstrainedHiddenMarkovModel(hmm, constraints)


@pytest.fixture
def recursive_chmm(lb, ub):
    return tc.create_chmm1_recursive(lb, ub)


class Test_Inference_a_star:

    def test_a_star(self, chmm, recursive_chmm):
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]

        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states

        inferred2 = (
            recursive_a_star(hmm=recursive_chmm, observed=observed).solutions[0].states
        )

        # assert inferred1 == inferred2

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    def test_a_star_2(self, chmm, recursive_chmm):
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]

        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states

        inferred2 = (
            recursive_a_star(hmm=recursive_chmm, observed=observed).solutions[0].states
        )

        # assert inferred1 == inferred2

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    def test_a_star_mult(self, chmm, recursive_chmm):
        observed = [
            "o1",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
        ]

        # inference1 = Inference(statistical_model=chmm, num_solutions=2)
        # inferred1 = inference1(observed)

        inferred2 = recursive_a_star(
            hmm=recursive_chmm, observed=observed, num_solutions=2
        )

        # assert inferred1.termination_condition == "ok"
        assert inferred2.termination_condition == "ok"
        # assert [sol.states for sol in inferred1.solutions] == [
        #    ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        #    ["h1", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        # ]
        assert [sol.states for sol in inferred2.solutions] == [
            ["h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
            ["h1", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0", "h0"],
        ]

    def test_a_star_no_solution(self, chmm, recursive_chmm):
        observed = ["o0"]

        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed)

        inferred2 = recursive_a_star(hmm=recursive_chmm, observed=observed)

        # assert inferred1.termination_condition == "error: no feasible solutions"

        assert inferred2.termination_condition == "error: no feasible solutions"

    def test_a_star_not_enough_solutions(self, chmm, recursive_chmm):
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]

        # inference1 = Inference(statistical_model=chmm, num_solutions=2)
        # inferred1 = inference1(observed)

        inferred2 = recursive_a_star(
            hmm=recursive_chmm, observed=observed, num_solutions=2
        )
        # assert inferred1.termination_condition == "ok"
        assert inferred2.termination_condition == "ok"


class Test_Inference_ip:

    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip(self, chmm1_pyomo):
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]

        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states

        inferred2 = (
            ip_inference(
                hmm=chmm1_pyomo,
                observed=observed,
                solver=mip_solver,
            )
            .solutions[0]
            .states
        )

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    @pytest.mark.skipif(not or_topas_available, reason="or_topas is not installed")
    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip_aos_trivial(self, chmm1_pyomo):
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]
        solver_options = dict(
            num_solutions=1,
            rel_opt_gap=None,
            abs_opt_gap=None,
            solver_options={},
            pool_manager=None,
            solver=mip_solver,
            topas_method="balas",
        )
        solver = "or_topas"
        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states

        inferred2 = (
            ip_inference(
                hmm=chmm1_pyomo,
                observed=observed,
                solver=solver,
                solver_options=solver_options,
            )
            .solutions[0]
            .states
        )

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    @pytest.mark.skipif(not or_topas_available, reason="or_topas is not installed")
    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip_aos_1(self, chmm1_pyomo_aos):
        # this test does not run into the constraints that limit inference
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]
        solver_options = dict(
            num_solutions=2,
            rel_opt_gap=None,
            abs_opt_gap=None,
            solver_options={},
            pool_manager=None,
            solver=mip_solver,
            topas_method="balas",
        )
        solver = "or_topas"
        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states
        ip_results = ip_inference(
            hmm=chmm1_pyomo_aos,
            observed=observed,
            solver=solver,
            solver_options=solver_options,
        )
        inferred2 = ip_results.solutions[0].states
        inferred_second_best = ip_results.solutions[1].states

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

        assert inferred_second_best == [
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
        ]

    @pytest.mark.skipif(not or_topas_available, reason="or_topas is not installed")
    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip_aos_2(self, chmm1_pyomo_aos):
        # this test does run into the constraints that limit inference
        observed = [
            "o1",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
        ]
        solver_options = dict(
            num_solutions=1,
            rel_opt_gap=None,
            abs_opt_gap=None,
            solver_options={},
            pool_manager=None,
            solver=mip_solver,
            topas_method="balas",
        )
        solver = "or_topas"
        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states
        ip_results = ip_inference(
            hmm=chmm1_pyomo_aos,
            observed=observed,
            solver=solver,
            solver_options=solver_options,
        )
        inferred2 = ip_results.solutions[0].states

        assert inferred2 == [
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
        ]

    @pytest.mark.skipif(not or_topas_available, reason="or_topas is not installed")
    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip_aos_3(self, chmm2_pyomo_aos):
        # this test does not run into the constraints that limit inference

        # best sol is h0*length
        # second best is h1*length
        # third best is h2, h0*(length-1)
        # fourth best is h1,h2, h0*(length-2)
        # if length > 12, second best can't happen, move sols up one
        observed = ["o1", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0", "o0"]
        solver_options = dict(
            num_solutions=4,
            rel_opt_gap=None,
            abs_opt_gap=None,
            solver_options={},
            pool_manager=None,
            solver=mip_solver,
            topas_method="balas",
        )
        solver = "or_topas"
        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states
        ip_results = ip_inference(
            hmm=chmm2_pyomo_aos,
            observed=observed,
            solver=solver,
            solver_options=solver_options,
        )
        inferred2 = ip_results.solutions[0].states
        inferred_second_best = ip_results.solutions[1].states
        inferred_third_best = ip_results.solutions[2].states
        inferred_fourth_best = ip_results.solutions[3].states

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

        assert inferred_second_best == [
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
        ]

        assert inferred_third_best == [
            "h2",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]
        assert inferred_fourth_best == [
            "h1",
            "h2",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    @pytest.mark.skipif(not or_topas_available, reason="or_topas is not installed")
    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip_aos_4(self, chmm2_pyomo_aos):
        # this test does run into the constraints that limit inference
        # length = 14

        # best sol is h0*length
        # second best is h1*length
        # third best is h2, h0*(length-1)
        # fourth best is h1,h2, h0*(length-2)
        # if length > 12, second best can't happen, move sols up one
        observed = [
            "o1",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
            "o0",
        ]
        solver_options = dict(
            num_solutions=4,
            rel_opt_gap=None,
            abs_opt_gap=None,
            solver_options={},
            pool_manager=None,
            solver=mip_solver,
            topas_method="balas",
        )
        solver = "or_topas"
        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states
        ip_results = ip_inference(
            hmm=chmm2_pyomo_aos,
            observed=observed,
            solver=solver,
            solver_options=solver_options,
        )
        inferred2 = ip_results.solutions[0].states
        inferred_second_best = ip_results.solutions[1].states
        inferred_third_best = ip_results.solutions[2].states

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

        assert inferred_second_best == [
            "h2",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

        assert inferred_third_best == [
            "h1",
            "h2",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip2(self, chmm1_pyomo):
        observed = ["o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1", "o1"]

        # TODO - Compare LP solution?
        # inference1 = Inference(statistical_model=chmm)
        # inferred1 = inference1(observed).solutions[0].states

        inferred2 = (
            ip_inference(
                hmm=chmm1_pyomo,
                observed=observed,
                solver=mip_solver,
            )
            .solutions[0]
            .states
        )

        # assert inferred1 == inferred2

        assert inferred2 == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
        ]

    @pytest.mark.skipif(not mip_solver, reason="No mip solver installed")
    def test_ip_deterministic_hmm(self):
        hmm = tc.create_chmm1_pyomo()

        observed = ["o0"] + ["o1"] * 14
        hidden = (
            ip_inference(hmm=hmm, observed=observed, solver=mip_solver)
            .solutions[0]
            .states
        )
        assert hidden == [
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h0",
            "h1",
            "h1",
            "h1",
            "h1",
            "h1",
        ]
