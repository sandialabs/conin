import inspect
from conin.exceptions import InvalidInputError

# One could also create an inherited class for additional functionality
# TODO think about partial_func semantics


class OracleConstraint:

    def __init__(
        self,
        *,
        func=None,
        name=None,
        partial_func=None,
        same_partial_as_func=None,
    ):
        """
        Initialize a OracleConstraint object.

        Parameters:
            func (callable, optional): The constraint function to be applied.
            name (str, optional): The name of the constraint. If not provided, it will default to the function's name.
            partial_func (callable, optional): This is a function which returns false on a partial sequence of
                hidden states only if there is no completion of the sequence which satisfies the constraints. E.g. if
                you go over budget halfway through, you can never recover from that. This has, as inputs T and seq instead
                of just seq like for func. This is useful for things like has minimum_number_of_occurences
            same_partial_as_func (bool, optional): If this is true, partial_func is set to func
        """
        self.func = func

        if same_partial_as_func is True:
            self.partial_func = lambda T, seq: func(seq)
        elif partial_func is not None:
            self.partial_func = partial_func
        else:
            self.partial_func = lambda T, seq: True

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name
        elif func is not None:
            self.name = func.__name__
        else:
            self.name = "Unnamed constraint"  # Could also be none

    def __call__(self, seq):
        """
        Apply the constraint function to a given sequence.

        Parameters:
            seq (iterable): The sequence to which the constraint function will be applied.

        Returns:
            The result of applying the constraint function to the sequence.

        Raises:
            InvalidInputError: If the constraint function is not defined.
        """
        if self.func is None:
            raise InvalidInputError(
                f"In constraint {self.name}, the actual constraint function is not defined."
            )
        return self.func(seq)


def oracle_constraint_fn(*, name=None, same_partial_as_func=None):
    """
    Decorator factory that takes the 'name' and returns a decorator function.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a OracleConstraint class.
        """
        return OracleConstraint(
            func=func, name=name, same_partial_as_func=same_partial_as_func
        )

    return decorator


class PyomoConstraint:

    def __init__(self, func, name=None):
        self.func = func
        self.num_args = len(inspect.signature(self.func).parameters)
        if self.num_args > 2:
            raise ValueError("Pyomo constraint defined with more than 2 arguments")

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name
        else:
            self.name = func.__name__

    def __call__(self, model, data):
        if self.num_args == 1:
            model_ = self.func(model)
        else:
            model_ = self.func(model, data)
        return model if model_ is None else model_


def pyomo_constraint_fn(*, name=None):
    """
    Decorator factory that takes the 'name' and returns a decorator function.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a PyomoConstraint class.
        """
        return PyomoConstraint(func=func, name=name)

    return decorator


class Toulbar2Constraint:

    def __init__(self, func, name=None):
        self.func = func
        self.num_args = len(inspect.signature(self.func).parameters)
        if self.num_args > 2:
            raise ValueError("Toulbar2 constraint defined with more than 2 arguments")

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name
        else:
            self.name = func.__name__

    def __call__(self, model, data):
        if self.num_args == 1:
            model_ = self.func(model)
        else:
            model_ = self.func(model, data)
        return model if model_ is None else model_


def toulbar2_constraint_fn(*, name=None):
    """
    Decorator factory that takes the 'name' and returns a decorator function.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a Toulbar2Constraint class.
        """
        return Toulbar2Constraint(func=func, name=name)

    return decorator
