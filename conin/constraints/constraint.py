import itertools
import inspect
from abc import ABC, abstractmethod
from ..exceptions import InvalidInputError

# One could also create an inherited class for additional functionality


class ConstraintFunctor(ABC):
    """
    Abstract base class for all constraint functors.

    A constraint functor is a callable object that encapsulates a constraint function
    and provides a consistent interface for applying constraints in different contexts.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Apply the constraint function.

        This method should be implemented by concrete constraint classes
        to define how the constraint is applied in their specific context.
        """
        pass


class OracleConstraint(ConstraintFunctor):

    def __init__(
        self,
        *,
        func=None,
        name=None,
        partial_func=None,
        same_partial_as_func=None,
        nodes=None,
    ):
        """
        Initialize an OracleConstraint object.

        This is the unified constraint class for all predicate-style constraints.
        The user function always receives a **dict** mapping keys to state values
        and returns ``True`` / ``False``.

        * For BN/MN the keys are node names, e.g. ``{"Dyspnoea": 1, "Xray": 0}``.
        * For DBN the keys are ``(name, t)`` tuples, e.g. ``{("A", 0): 0}``.
        * For HMM the keys are time indices, e.g. ``{0: "rainy", 1: "sunny"}``.

        Parameters
        ----------
        func : callable, optional
            The constraint predicate ``func(assignment) -> bool``, or
            ``func(assignment, data) -> bool`` for constraints that need
            inference-time data (e.g. DBN time indices).
        name : str, optional
            Human-readable name.  Defaults to ``func.__name__`` when *func*
            is provided.
        partial_func : callable, optional
            A function ``(T, states_dict) -> bool`` that may prune partial
            sequences during A* search.  Only relevant for HMM oracle
            constraints.
        same_partial_as_func : bool, optional
            If ``True``, ``partial_func`` is set to
            ``lambda T, states: func(states)``.
        nodes : list or callable, optional
            Node scope for factor materialisation (BN / MN / DBN).
            Can be a list of node names or a callable ``nodes(data)`` that
            yields node names.  When ``nodes`` is set, calling the constraint
            with a PGM materialises it into a ``DiscreteFactor`` or
            ``DiscreteCPD``.  When ``nodes`` is ``None`` and the constraint is
            called with a PGM, every node in the PGM is used as the scope
            (note: this can be expensive for large models — prefer specifying
            an explicit ``nodes`` scope when possible).
        """
        self.func = func
        self.nodes = nodes

        if same_partial_as_func is True:
            self.partial_func = lambda T, states: func(states)
        elif partial_func is not None:
            self.partial_func = partial_func
        else:
            self.partial_func = lambda T, states: True

        # Determine how many positional args `func` accepts (1 or 2).
        # This matters for factor-style constraints that may accept `data`.
        if func is not None:
            self.num_args = len(inspect.signature(func).parameters)
        else:
            self.num_args = None

        # If no name is provided, use the function's name
        if name is not None:
            self.name = name
        elif func is not None:
            self.name = func.__name__
        else:
            self.name = "Unnamed constraint"  # Could also be none

    def __call__(self, *args, **kwargs):
        """
        Apply the constraint.

        When called with a states dict (oracle / black-box mode):
            ``constraint(states_dict) -> bool``

        When called with a PGM as the first argument (factor mode):
            ``constraint(pgm, data=None)`` — materialises the predicate into
            a ``DiscreteFactor`` (for MN) or ``DiscreteCPD`` (for BN) by
            enumerating all assignments over the declared ``nodes``.  When
            ``nodes`` is ``None``, every node in the PGM is used as the scope
            (note: this can be expensive for large models).

        Raises
        ------
        InvalidInputError
            If the constraint function is not defined.
        """
        if self.func is None:
            raise InvalidInputError(
                f"In constraint {self.name}, the actual constraint function is not defined."
            )

        # Route to materialisation when called with a pgm as the first argument.
        # This covers both nodes-set (factor mode) and nodes=None (all-nodes mode).
        from ..markov_network import DiscreteMarkovNetwork
        from ..bayesian_network import DiscreteBayesianNetwork

        if args and isinstance(
            args[0], (DiscreteMarkovNetwork, DiscreteBayesianNetwork)
        ):
            return self._materialise(*args, **kwargs)

        if self.nodes is not None:
            return self._materialise(*args, **kwargs)

        return self.func(*args, **kwargs)

    def _materialise(self, pgm, data=None):
        """Materialise the predicate into a factor or CPD.

        Parameters
        ----------
        pgm : DiscreteMarkovNetwork or DiscreteBayesianNetwork
            The probabilistic graphical model whose state spaces are used
            for enumeration.
        data : optional
            Application-specific data forwarded to the predicate when it
            accepts two arguments, and to ``nodes`` when it is callable.

        Returns
        -------
        DiscreteFactor or DiscreteCPD
        """
        from ..markov_network import DiscreteFactor, DiscreteMarkovNetwork
        from ..bayesian_network import DiscreteCPD, DiscreteBayesianNetwork

        if self.nodes is None:
            nodes = list(pgm.nodes)
        elif type(self.nodes) is list:
            nodes = self.nodes
        else:
            nodes = list(self.nodes(data))

        if type(pgm) == DiscreteMarkovNetwork:
            M = 10**6
            values = {}
            for states in itertools.product(*(pgm.states[name] for name in nodes)):
                assignment = {name: states[i] for i, name in enumerate(nodes)}
                if self.num_args == 1:
                    feasible = self.func(assignment)
                else:
                    feasible = self.func(assignment, data)
                values[states] = M if feasible else 0
            return DiscreteFactor(nodes=nodes, values=values)

        elif type(pgm) == DiscreteBayesianNetwork:
            node = self.name
            values = {}
            for states in itertools.product(*(pgm.states[name] for name in nodes)):
                assignment = {name: states[i] for i, name in enumerate(nodes)}
                if self.num_args == 1:
                    feasible = self.func(assignment)
                else:
                    feasible = self.func(assignment, data)
                feasible = int(feasible)
                values[states] = {1: feasible, 0: 1 - feasible}
            return DiscreteCPD(node=node, parents=nodes, values=values)

        else:
            raise ValueError(f"Unexpected pgm: {type(pgm)}")


def oracle_constraint_fn(*, nodes=None, name=None, same_partial_as_func=None):
    """
    Decorator factory that creates an ``OracleConstraint``.

    Parameters
    ----------
    nodes : list or callable, optional
        Node scope for factor materialisation.  See
        :class:`OracleConstraint` for details.
    name : str, optional
        Human-readable constraint name.
    same_partial_as_func : bool, optional
        If ``True``, the partial-feasibility check reuses the main predicate.
    """

    def decorator(func):
        return OracleConstraint(
            func=func,
            name=name,
            same_partial_as_func=same_partial_as_func,
            nodes=nodes,
        )

    return decorator


class MVRConstraint(ConstraintFunctor):

    def __init__(self, func=None, name=None):
        self.func = func
        if self.func is not None:
            self.num_args = len(inspect.signature(self.func).parameters)
            if self.num_args > 2:
                raise ValueError("MVR constraint defined with more than 2 arguments")
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

        from ..hidden_markov_model.mvr import BaseMVR

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


class PyomoConstraint(ConstraintFunctor):

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
    Decorator factory that takes the 'name' and returns a decorator function that creates an instance of PyomoConstraint.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a PyomoConstraint class.
        """
        return PyomoConstraint(func=func, name=name)

    return decorator


class Toulbar2Constraint(ConstraintFunctor):

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
    Decorator factory that takes the 'name' and returns a decorator function that creates an instance of Toulbar2Constraint.
    """

    def decorator(func):
        """
        The actual decorator that wraps the user constraint function in a Toulbar2Constraint class.
        """
        return Toulbar2Constraint(func=func, name=name)

    return decorator
