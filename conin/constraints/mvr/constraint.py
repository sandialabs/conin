import inspect

from conin.exceptions import InvalidInputError
from conin.constraints.base import Constraint
from .base import BaseMVR


class MVRConstraint(Constraint):

    family = "mvr"

    def __init__(self, func=None, name=None):
        self.func = func
        if self.func is not None:
            self.num_args = len(inspect.signature(self.func).parameters)
            if self.num_args > 2:
                raise ValueError("MVR constraint defined with more than 2 arguments")
            if self.num_args == 0:
                raise ValueError(
                    "An MVR constraint function takes (hidden_markov_model) or "
                    f"(hidden_markov_model, data); {self.func.__name__} takes none"
                )
        else:
            self.num_args = None

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name
        elif func is not None:
            self.name = func.__name__
        else:
            self.name = "Unnamed constraint"

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
