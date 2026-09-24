import itertools
import inspect
from .constraint import ConstraintFunctor
from ..exceptions import InvalidInputError
from ..markov_network import DiscreteFactor, DiscreteMarkovNetwork
from ..bayesian_network import DiscreteCPD, DiscreteBayesianNetwork


class FactorConstraint(ConstraintFunctor):

    def __init__(
        self,
        *,
        nodes,
        func=None,
        name=None,
    ):
        """
        Initialize a FactorConstraint object.

        Parameters:
            func (callable, optional): The constraint function to be applied.
            name (str, optional): The name of the constraint. If not provided, it will default to the function's name.
            nodes (list, callable): The nodes that are arguments for the constraint function.  If this is not a list, then this is assumed to be a function that accepts an argument that provides data for the constraint.
        """
        self.nodes = nodes
        self.func = func
        self.num_args = len(inspect.signature(self.func).parameters)
        if self.num_args > 2:
            raise ValueError(
                "FactorConstraint constraint defined with more than 2 arguments"
            )

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name
        elif func is not None:
            self.name = func.__name__
        else:
            self.name = "Unnamed constraint"  # Could also be none

    def __call__(self, pgm, data=None):
        """
        Create a DiscreteFactor for the constraint function.

        Returns:
            The a DiscreteFactor object.

        Raises:
            InvalidInputError: If the constraint function is not defined.
        """
        if self.func is None:
            raise InvalidInputError(
                f"In constraint {self.name}, the actual constraint function is not defined."
            )

        if type(self.nodes) is list:
            nodes = self.nodes
        else:
            nodes = list(self.nodes(data))

        if type(pgm) == DiscreteMarkovNetwork:
            M = 10**6
            values = {}
            for states in itertools.product(*(pgm.states[name] for name in nodes)):
                args = {}
                for i, name in enumerate(nodes):
                    args[name] = states[i]
                if self.num_args == 1:
                    feasible = self.func(args)  # True is yes, False is no
                else:
                    feasible = self.func(args, data)  # True is yes, False is no
                values[states] = M if feasible else 0
            return DiscreteFactor(nodes=nodes, values=values)

        elif type(pgm) == DiscreteBayesianNetwork:
            node = self.name
            values = {}
            for states in itertools.product(*(pgm.states[name] for name in nodes)):
                args = {}
                for i, name in enumerate(nodes):
                    args[name] = states[i]
                if self.num_args == 1:
                    feasible = self.func(args)  # True is yes, False is no
                else:
                    feasible = self.func(args, data)  # True is yes, False is no
                feasible = int(feasible)
                values[states] = {1: feasible, 0: 1 - feasible}
            return DiscreteCPD(node=node, parents=nodes, values=values)

        else:
            raise ValueError(f"Unexpected pgm: {type(pgm)}")


def factor_constraint_fn(*, nodes=None, name=None):
    """
    Decorator factory that takes the 'name' and returns a decorator function that creates an instance of FactorConstraint.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a OracleConstraint class.
        """
        return FactorConstraint(nodes=nodes, func=func, name=name)

    return decorator
