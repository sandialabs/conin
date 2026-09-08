from abc import ABC, abstractmethod
from typing import Any

from conin.exceptions import InvalidInputError
from conin.hidden_markov_model.mvr import BaseMVR


class OperatorFunctor(ABC):
    """Abstract callable interface for constraint operators."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Apply the operator."""
        pass


class MVROperator(OperatorFunctor):

    def __init__(self, func=None, name=None, arity=None):
        self.func = func
        self.arity = arity

        if name is not None:
            self.name = name
        elif func is not None:
            self.name = func.__name__
        else:
            self.name = "Unnamed operator"

        if func is not None:
            self.__doc__ = func.__doc__

        if arity is not None:
            if not isinstance(arity, int) or arity <= 0:
                raise InvalidInputError("arity must be a positive integer")

    def __call__(self, mvrs, *op_args, hidden_markov_model=None, **kwargs):
        if self.func is None:
            raise InvalidInputError(
                f"In operator {self.name}, the actual operator function is not defined."
            )

        if mvrs is None:
            raise InvalidInputError("an MVR or list of MVRs must be provided")
        elif isinstance(mvrs, BaseMVR):
            input_arity = 1
            hidden_states = mvrs.hidden_states
            mvrs = [mvrs]
        elif isinstance(mvrs, list):
            if len(mvrs) == 0:
                raise InvalidInputError("empty list was passed in")
            for i, mvr in enumerate(mvrs):
                if not isinstance(mvr, BaseMVR):
                    raise InvalidInputError(f"Object at index {i} is not an MVR object")
            input_arity = len(mvrs)
            hidden_states = _validate_hidden_states(mvrs)
        else:
            raise InvalidInputError("input must be an MVR or list of MVRs")

        if self.arity is not None:
            if self.arity != input_arity:
                raise InvalidInputError(
                    f"operator arity is {self.arity} but provided list has length {input_arity}"
                )

        if hidden_markov_model is not None:
            if hidden_markov_model.hidden_states != hidden_states:
                raise InvalidInputError(
                    "The hidden states of the HMM and the (first) MVR do not exactly match"
                )

        result = self.func(mvrs, *op_args, **kwargs)

        if not isinstance(result, BaseMVR):
            raise InvalidInputError(
                f"The output of operator {self.name} is not an MVR object."
            )

        # An operator returning one of its inputs is explicitly in-place or
        # idempotent. Preserve that identity; there is no new state space to prune.
        if any(result is mvr for mvr in mvrs):
            return result

        # Minimize here rather than in each operator body: product and subset
        # constructions leave most states unreachable, and cost is K * prod(M_i).
        return result.prune()


def mvr_operator_fn(*, name=None, arity=None):
    """Decorate a function as an MVR operator with optional fixed arity."""

    def decorator(func):
        return MVROperator(func=func, name=name, arity=arity)

    return decorator


def _validate_hidden_states(mvrs: list[BaseMVR]) -> list[Any]:
    """Validate a common hidden space and return the first MVR's ordering."""
    hidden_states = mvrs[0].hidden_states
    hidden_space = set(hidden_states)

    for i, mvr in enumerate(mvrs[1:], start=1):
        if set(mvr.hidden_states) != hidden_space:
            raise InvalidInputError(
                f"mvrs[{i}] must have the same hidden_states as mvrs[0]."
            )

    return hidden_states
