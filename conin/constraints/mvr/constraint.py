import inspect

from conin.exceptions import InvalidInputError
from conin.constraints.base import Constraint
from .base import BaseMVR


class MVRConstraint(Constraint):

    family = "mvr"

    def __init__(self, func=None, name=None):
        if func is not None:
            num_args = len(inspect.signature(func).parameters)
            if num_args > 2:
                raise InvalidInputError(
                    "MVR constraint defined with more than 2 arguments"
                )
            if num_args == 0:
                raise InvalidInputError(
                    "An MVR constraint function takes (hidden_markov_model) or "
                    f"(hidden_markov_model, data); {func.__name__} takes none"
                )
        else:
            num_args = None

        # If no name is provided, use the function's name
        if name is None:
            name = func.__name__ if func is not None else "Unnamed constraint"
        self._set(func=func, num_args=num_args, name=name)

    def __call__(self, hidden_markov_model, data=None):
        if self.func is None:
            raise InvalidInputError(
                f"In constraint {self.name}, the actual constraint function is not defined."
            )

        if self.num_args == 1:
            mvr = self.func(hidden_markov_model)
        else:
            mvr = self.func(hidden_markov_model, data)

        if not isinstance(mvr, BaseMVR):
            raise InvalidInputError(
                f"In constraint {self.name}, the returned object is not an MVR object."
            )

        return mvr


def mvr_constraint_fn(*, name=None):
    """
    Decorator factory that takes the 'name' and returns a decorator function that creates an instance of MVRConstraint.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a MVRConstraint class.
        """
        return MVRConstraint(func=func, name=name)

    return decorator
