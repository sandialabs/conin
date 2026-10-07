"""Tests for conin.util.solution."""

import math

from conin.util.solution import MPESolution


class TestMPESolution:
    def test_default_states_is_none(self):
        sol = MPESolution()
        assert sol.states is None

    def test_default_log_value_is_nan(self):
        sol = MPESolution()
        assert math.isnan(sol.log_value)

    def test_states_kwarg(self):
        states = {"A": 0, "B": 1}
        sol = MPESolution(states=states)
        assert sol.states == {"A": 0, "B": 1}

    def test_log_value_kwarg(self):
        sol = MPESolution(log_value=-3.5)
        assert sol.log_value == -3.5

    def test_positional_construction(self):
        states = {"X": 2}
        sol = MPESolution(states, -1.0)
        assert sol.states == {"X": 2}
        assert sol.log_value == -1.0

    def test_equality(self):
        sol1 = MPESolution(states={"A": 0}, log_value=-1.0)
        sol2 = MPESolution(states={"A": 0}, log_value=-1.0)
        assert sol1 == sol2

    def test_inequality_states(self):
        sol1 = MPESolution(states={"A": 0}, log_value=-1.0)
        sol2 = MPESolution(states={"A": 1}, log_value=-1.0)
        assert sol1 != sol2

    def test_inequality_log_value(self):
        sol1 = MPESolution(states={"A": 0}, log_value=-1.0)
        sol2 = MPESolution(states={"A": 0}, log_value=-2.0)
        assert sol1 != sol2

    def test_repr_contains_classname(self):
        sol = MPESolution(states={"A": 0}, log_value=-1.0)
        assert "MPESolution" in repr(sol)

    def test_states_is_mutable(self):
        sol = MPESolution(states={"A": 0})
        sol.states["B"] = 1
        assert sol.states == {"A": 0, "B": 1}
