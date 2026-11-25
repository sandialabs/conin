import inspect

from conin.exceptions import InvalidInputError

from .base import Constraint, decorate


class Toulbar2Constraint(Constraint):

    family = "toulbar2"

    def __init__(self, func, name=None):
        num_args = len(inspect.signature(func).parameters)
        if num_args > 2:
            raise InvalidInputError(
                "Toulbar2 constraint defined with more than 2 arguments"
            )

        # If no name is provided, use the function's name
        self._set(
            func=func,
            num_args=num_args,
            name=name if name is not None else func.__name__,
        )

    def __call__(self, model, data):
        if self.num_args == 1:
            model_ = self.func(model)
        else:
            model_ = self.func(model, data)
        return model if model_ is None else model_


def toulbar2_constraint_fn(func=None, /, *, name=None):
    """Decorator factory that turns a function into a :class:`Toulbar2Constraint`.

    The function is called as ``func(model)`` or ``func(model, data)`` while
    the ``"toulbar2"`` representation builds the cost function network of a
    query. ``model`` is the ``pytoulbar2.CFN``: ``model.V(node, state)`` (or
    ``model.V(name, t, state)``) is the ``(variable, value, coefficient)``
    triple of pytoulbar2's linear constraints, ``model.V[node]`` the index
    of a variable and ``model.states[i]`` the states of variable ``i``. The
    function adds the constraint with the CFN's methods and returns ``None``
    (or the model).

    Use it as ``@toulbar2_constraint_fn`` or, with arguments,
    ``@toulbar2_constraint_fn(name=...)``.

    Parameters
    ----------
    func : callable, optional
        The constraint function, when the decorator is used without
        arguments.
    name : str, optional
        The constraint's name; defaults to the function's name.

    Notes
    -----
    toulbar2's cost functions are soft: a cost function forbids an assignment
    only if the assignment costs at least toulbar2's upper bound. CONIN scales
    the model's costs up, so a cost that is merely large (``1e6``) can be
    cheaper than the model's costs, and the solution may violate the
    constraint. Give the forbidden assignments of a cost function
    (``AddFunction``) the cost ``float("inf")``. The linear constraints
    (``AddLinearConstraint``, ``AddGeneralizedLinearConstraint``) are hard. The
    violation cost of a global cost function (``AddGlobalFunction``) can't be
    made hard this way.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a Toulbar2Constraint class.
        """
        return Toulbar2Constraint(func=func, name=name)

    return decorate(func, decorator, "toulbar2_constraint_fn")
