"""Shared toulbar2 representation code.

- :class:`Toulbar2FactorRepresentation`: the base class of the ``"toulbar2"``
  representations (it holds the UAI text and builds a fresh
  ``pytoulbar2.CFN`` per query);
- :class:`Toulbar2VarWrapper`: ``model.V(...)`` on a CFN;
- :func:`apply_toulbar2_constraints`: the single constraint applier for the
  toulbar2-backend families (``"toulbar2"`` and ``"algebraic"``);
- :func:`add_algebraic_constraints_to_toulbar2_model` and
  :func:`add_toulbar2_constraints`: the smoek -> toulbar2 linear translator.
"""

import os.path
import tempfile
from fractions import Fraction
from math import gcd
from functools import reduce

import munch
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
from pyomo.common.collections import ComponentMap
from pyomo.common.timing import TicTocTimer

from conin.constraints.algebraic.decorators import build_smoek_model
from conin.io.uai import _uai_text
from conin.representations.base import Representation, apply_constraints
from conin.util import try_import, State
from conin.util.constraint_data import merge_constraint_data
from conin.util.var_wrapper import PyomoVarWrapper

with try_import() as smoek_available:
    import smoek

with try_import() as pytoulbar2_available:
    import pytoulbar2

__all__ = [
    "Toulbar2VarWrapper",
    "apply_toulbar2_constraints",
    "Toulbar2FactorRepresentation",
    "add_algebraic_constraints_to_toulbar2_model",
    "add_toulbar2_constraints",
]


class Toulbar2VarWrapper(object):
    """``model.V`` on a toulbar2 CFN.

    ``V(node, state, coef=1)`` (or ``V(name, t, state, coef=1)`` for the
    time-indexed node ``(name, t)``) returns the ``(variable index, value
    index, coef)`` triple used by pytoulbar2's linear-constraint API;
    ``V[node]`` returns the variable index.  Variables are numbered in
    ``pgm.nodes`` order and values in ``pgm.states_of(node)`` order, as in the
    UAI file.
    """

    def __init__(self, pgm):
        self._V = {name: i for i, name in enumerate(pgm.nodes)}
        self._V_state = {
            (name, state): i
            for name in pgm.nodes
            for i, state in enumerate(pgm.states_of(name))
        }

    def __len__(self):
        return len(self._V)

    def __call__(self, *args, coef=1):
        if len(args) == 2:
            r, s = args
        elif len(args) == 3:
            r, i, s = args
            r = (r, i)
        else:
            raise ValueError("There must be either 2 or 3 arguments")

        return (self._V[r], self._V_state[r, s], coef)

    def __getitem__(self, r):
        return self._V[r]

    def items(self):
        for k, v in self._V.items():
            yield k, v


def _apply_toulbar2_family(constraints, model, data):
    for func in constraints:
        model = func(model, data)
    return model


def apply_toulbar2_constraints(*, pgm, constraints, model, data):
    """Apply constraints of a toulbar2-backend family to a toulbar2 ``model``.

    This is the single constraint applier for the ``"toulbar2"``
    representations.

    Parameters
    ----------
    pgm : GraphicalModel
        The model the constraints are written against (the MN, the BN, the
        unrolled BN of a DBN, or the DBN of an HMM).  The algebraic translator
        reads its ``states_of``.
    constraints : sequence of Toulbar2Constraint or AlgebraicConstraint
        Constraints of one family (``"toulbar2"`` or ``"algebraic"``).
    model : pytoulbar2.CFN
        The model; constraint functions see ``model.V(...)``.
    data : munch.Munch
        Passed to each constraint function.

    Returns
    -------
    pytoulbar2.CFN
        The model with constraints added.

    Raises
    ------
    TypeError
        For constraints of several families or of another family.
    """
    result = apply_constraints(
        constraints,
        {
            "toulbar2": lambda cons: _apply_toulbar2_family(cons, model, data),
            "algebraic": lambda cons: add_algebraic_constraints_to_toulbar2_model(
                pgm=pgm, constraints=cons, model=model, data=data
            ),
        },
        backend="toulbar2",
    )
    return model if result is None else result


class Toulbar2FactorRepresentation(Representation):
    """The toulbar2 cost function network of a constrained MN/BN/DBN/HMM.

    A ``pytoulbar2.CFN`` can't be copied, so the representation holds the
    evidence-independent intermediate: the UAI text of ``uai_model`` and the
    constraints.  :meth:`create_model` reads the text into a fresh CFN on
    every call, assigns the evidence and then applies the constraints.

    Attributes
    ----------
    uai_text : str
        The UAI text of ``uai_model`` (what :func:`conin.io.save_model` writes).
    uai_model : DiscreteMarkovNetwork or DiscreteBayesianNetwork
        The model the CFN is read from (the MN, the BN or the unrolled BN).
        Its ``nodes`` and ``states_of`` order fix the CFN variable and value
        indices.
    pgm : GraphicalModel
        The model passed to the constraint applier (the MN, the BN, the
        unrolled BN of a DBN, or the DBN of an HMM).
    constraints : tuple
        The user constraints (family ``"toulbar2"`` or ``"algebraic"``).
    nodes : tuple
        The nodes of ``uai_model``.
    data : object or None
        The constrained model's application data.  The constraint functions
        see a copy of it with the fields of :meth:`constraint_data` set.

    Notes
    -----
    Nothing is mutated after construction, so one representation can serve
    several queries.
    """

    kind = "toulbar2"

    def __init__(self, *, uai_model, pgm, constraints, uai_text=None, data=None):
        self.uai_model = uai_model
        self.uai_text = _uai_text(uai_model) if uai_text is None else uai_text
        self.pgm = pgm
        self.constraints = tuple(constraints)
        self.nodes = tuple(uai_model.nodes)
        self.data = data

    def structure_fields(self):
        """The constraint ``data`` fields of the structure params (none here).

        The DBN and HMM subclasses return ``start``, ``stop`` and ``T`` (or
        ``hmm``); see :mod:`conin.representations.unrolled`.
        """
        return munch.Munch()

    def constraint_data(self, *, evidence=None, variables=None):
        """The fields this representation adds to the constraint ``data`` for a query.

        ``variables`` and ``evidence`` plus :meth:`structure_fields`.  The
        constraint functions see these fields set on a copy of the
        constrained model's ``data`` (or just these fields if it has none).
        """
        fields = munch.Munch(variables=variables, evidence=evidence)
        fields.update(self.structure_fields())
        return fields

    def create_model(
        self,
        *,
        evidence=None,
        variables=None,
        verbose=-1,
        write_uai_file=None,
        timing=False,
    ):
        """Build a fresh ``pytoulbar2.CFN`` for one query.

        Parameters
        ----------
        evidence : dict, optional
            ``{node: state}``; each node is assigned with ``CFN.Assign``.
        variables : list, optional
            Passed to the constraint functions as ``data.variables``.
        verbose : int, optional
            The toulbar2 verbosity (``-1``: no messages).
        write_uai_file : str, optional
            Write the UAI text to this file (and read the CFN from it).
        timing : bool, optional
            Print timing information.

        Returns
        -------
        pytoulbar2.CFN
            With ``V`` (a :class:`Toulbar2VarWrapper`), ``states``
            (variable index -> states) and ``V_evidence`` (the assigned nodes).
        """
        if timing:  # pragma:nocover
            timer = TicTocTimer()
            timer.tic("toulbar2 create_model - START")

        if write_uai_file:
            model = self._read(write_uai_file, verbose)
        else:
            with tempfile.TemporaryDirectory() as tempdir:
                model = self._read(os.path.join(tempdir, "model.uai"), verbose)

        pgm = self.uai_model
        model.V = Toulbar2VarWrapper(pgm)
        model.states = {i: pgm.states_of(name) for i, name in enumerate(pgm.nodes)}

        model.V_evidence = set()
        if evidence:
            for k, v in evidence.items():
                model.Assign(model.V[k], pgm.states_of(k).index(v))
                model.V_evidence.add(k)

        if self.constraints:
            apply_toulbar2_constraints(
                pgm=self.pgm,
                constraints=self.constraints,
                model=model,
                data=merge_constraint_data(
                    self.data,
                    self.constraint_data(evidence=evidence, variables=variables),
                ),
            )

        if timing:  # pragma:nocover
            timer.toc("toulbar2 create_model - STOP")
        return model

    def _read(self, filename, verbose):
        if not pytoulbar2_available:
            raise ImportError("pytoulbar2 is required to build a toulbar2 model")
        with open(filename, "w") as f:
            f.write(self.uai_text)
        _reset_toulbar2_input_format()
        model = pytoulbar2.CFN(verbose=verbose)
        model.Read(filename)
        return model


# pytoulbar2 keeps the format of the last file it read in global options and
# never clears them, so after a user reads e.g. a ``.cfn`` or ``.cfn.gz`` file
# with pytoulbar2, reading a plain UAI file fails ("wrong problem file format").
_TOULBAR2_INPUT_FORMAT_OPTIONS = (
    "bz2",
    "cfn",
    "gz",
    "lp",
    "opb",
    "qpbo",
    "wcnf",
    "xmlflag",
    "xz",
)


def _reset_toulbar2_input_format():
    option = pytoulbar2.pytb2.option
    for name in _TOULBAR2_INPUT_FORMAT_OPTIONS:
        if hasattr(option, name):
            setattr(option, name, False)


def add_algebraic_constraints_to_toulbar2_model(*, pgm, constraints, model, data):
    """Translate smoek (algebraic) constraints into linear toulbar2 constraints.

    The constraints are generated on a temporary Pyomo model with one variable
    per ``(node, state)`` of ``pgm`` and then added to ``model`` with
    :func:`add_toulbar2_constraints`.

    Raises
    ------
    ImportError
        If smoek is not installed.
    ValueError
        For a nonlinear constraint.
    """
    smoek_model = build_smoek_model(constraints, data)

    pyomo_model = pyo.ConcreteModel()
    N = sum(len(pgm.states_of(k)) for k, _ in model.V.items())
    pyomo_model.V_conin_temp = pyo.Var(pyo.RangeSet(0, N - 1))

    pyomo_model.V_to_tb2 = ComponentMap()
    tmp = {}
    ctr = 0
    for k, _ in model.V.items():
        for s in pgm.states_of(k):
            pyomo_model.V_to_tb2[pyomo_model.V_conin_temp[ctr]] = model.V(k, s)
            tmp[k, State(s)] = pyomo_model.V_conin_temp[ctr]
            ctr += 1

    pyomo_model.V = PyomoVarWrapper(tmp)

    pyomo_model = smoek.pymodel.pyomo.generate(
        model=smoek_model,
        pyomo_model=pyomo_model,
        data=data,
    )
    add_toulbar2_constraints(model=model, pyomo_model=pyomo_model)
    return model


def add_toulbar2_constraints(model, pyomo_model):
    """
    Add constraints from a Pyomo model to a Toulbar2 model.

    This function iterates through all constraints in the Pyomo model and adds
    them to the Toulbar2 model. Only linear constraints are supported.

    Parameters
    ----------
    model : pytoulbar2.CFN
        The Toulbar2 constraint satisfaction network model.
    pyomo_model : pyomo.environ.ConcreteModel
        The Pyomo optimization model containing constraints to be added.

    Returns
    -------
    pytoulbar2.CFN
        The Toulbar2 model with added constraints.

    Raises
    ------
    ValueError
        If a nonlinear constraint is encountered.

    Notes
    -----
    This function converts linear Pyomo constraints (equality and inequality) to
    Toulbar2 weighted constraints. The conversion handles:
    - Linear equality constraints (a*x + b*y + ... == c)
    - Linear inequality constraints (a*x + b*y + ... <= c or >= c)

    Nonlinear constraints will raise an exception.
    """
    # Iterate through all constraint components in the Pyomo model
    for constraint_component in pyomo_model.component_objects(
        pyo.Constraint, active=True
    ):
        # Iterate through each constraint data object (handles indexed constraints)
        for index in constraint_component:
            constraint_data = constraint_component[index]

            # Skip inactive constraints
            if not constraint_data.active:
                continue

            # Get the constraint body expression
            body = constraint_data.body
            lower = constraint_data.lower
            upper = constraint_data.upper

            # Generate standard representation to check linearity and extract coefficients
            repn = generate_standard_repn(body)

            # Check if the constraint is nonlinear
            if repn.nonlinear_expr is not None:
                raise ValueError(
                    f"Nonlinear constraint detected: {constraint_component.name}[{index}]. "
                    "Only linear constraints are supported when adding Pyomo constraints to Toulbar2."
                )

            # Check for quadratic terms
            if repn.quadratic_vars:
                raise ValueError(
                    f"Quadratic constraint detected: {constraint_component.name}[{index}]. "
                    "Only linear constraints are supported when adding Pyomo constraints to Toulbar2."
                )

            # Extract linear terms: coefficients and variables
            # repn.linear_vars is a tuple of variables
            # repn.linear_coefs is a tuple of coefficients
            # repn.constant is the constant term

            variables = []
            coefficients = []

            for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                variables.append(pyomo_model.V_to_tb2[var])
                coefficients.append(float(coef))

            constant = float(repn.constant) if repn.constant is not None else 0.0

            # Convert the constraint to Toulbar2 format
            # Toulbar2 uses weighted constraints where violations have costs

            if lower is not None and upper is not None:
                # Equality constraint or range constraint
                if lower == upper:
                    # Equality: sum(coef[i] * var[i]) == rhs
                    # Convert to: sum(coef[i] * var[i]) - rhs == 0
                    rhs = float(lower) - constant
                    _add_toulbar2_linear_constraint(
                        model,
                        variables,
                        coefficients,
                        rhs,
                        operator="==",
                        name=f"{constraint_component.name}[{index}]",
                    )
                else:
                    # Range constraint: lower <= sum(coef[i] * var[i]) <= upper
                    # Split into two inequality constraints
                    rhs_lower = float(lower) - constant
                    rhs_upper = float(upper) - constant
                    _add_toulbar2_linear_constraint(
                        model,
                        variables,
                        coefficients,
                        rhs_lower,
                        operator=">=",
                        name=f"{constraint_component.name}[{index}]_lower",
                    )
                    _add_toulbar2_linear_constraint(
                        model,
                        variables,
                        coefficients,
                        rhs_upper,
                        operator="<=",
                        name=f"{constraint_component.name}[{index}]_upper",
                    )
            elif lower is not None:
                # Lower bound only: sum(coef[i] * var[i]) >= lower
                rhs = float(lower) - constant
                _add_toulbar2_linear_constraint(
                    model,
                    variables,
                    coefficients,
                    rhs,
                    operator=">=",
                    name=f"{constraint_component.name}[{index}]",
                )
            elif upper is not None:
                # Upper bound only: sum(coef[i] * var[i]) <= upper
                rhs = float(upper) - constant
                _add_toulbar2_linear_constraint(
                    model,
                    variables,
                    coefficients,
                    rhs,
                    operator="<=",
                    name=f"{constraint_component.name}[{index}]",
                )

    return model


def _scale_to_integers(coefficients, rhs, max_denominator=10**6):
    """
    Scale floating-point coefficients and RHS to integers while preserving precision.

    This function finds an appropriate scaling factor to convert all floating-point
    values to integers without losing significant precision.

    Parameters
    ----------
    coefficients : list of float
        The coefficients to scale.
    rhs : float
        The right-hand side value to scale.
    max_denominator : int, optional
        Maximum denominator to use when converting to fractions (default: 10^6).

    Returns
    -------
    tuple
        (scaled_coefficients, scaled_rhs, scale_factor)
        - scaled_coefficients: list of int
        - scaled_rhs: int
        - scale_factor: float (the factor used for scaling)

    Notes
    -----
    The function uses the Fraction class to find the least common multiple of
    denominators, then scales all values to integers.
    """
    # Convert all values to Fractions to get exact rational representations
    # Limit the denominator to avoid extremely large integers
    all_values = coefficients + [rhs]
    fractions = [
        Fraction(float(v)).limit_denominator(max_denominator) for v in all_values
    ]

    # Find the LCM of all denominators
    denominators = [f.denominator for f in fractions]

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)

    lcm_denom = reduce(lcm, denominators)

    # Scale all values by the LCM to get integers
    scaled_coefficients = [
        int(fractions[i] * lcm_denom) for i in range(len(coefficients))
    ]
    scaled_rhs = int(fractions[-1] * lcm_denom)

    # Simplify by finding GCD of all scaled values to reduce magnitude
    all_scaled = scaled_coefficients + [scaled_rhs]
    # Filter out zeros before computing GCD
    non_zero_values = [v for v in all_scaled if v != 0]

    if non_zero_values:
        common_gcd = reduce(gcd, non_zero_values)
        if common_gcd > 1:
            scaled_coefficients = [c // common_gcd for c in scaled_coefficients]
            scaled_rhs = scaled_rhs // common_gcd
            # The values are now scaled by lcm_denom / common_gcd, which is
            # less than 1 if they were integers with a common factor.
            return scaled_coefficients, scaled_rhs, lcm_denom / common_gcd

    return scaled_coefficients, scaled_rhs, float(lcm_denom)


def _add_toulbar2_linear_constraint(
    model, variables, coefficients, rhs, operator, name=None
):
    """
    Helper function to add a linear constraint to a Toulbar2 model.

    Parameters
    ----------
    model : pytoulbar2.CFN
        The Toulbar2 model.
    variables : list
        List of Pyomo variables involved in the constraint.
    coefficients : list
        List of coefficients for each variable.
    rhs : float
        Right-hand side value of the constraint.
    operator : str
        Type of constraint: '==', '<=' or '>='
    name : str, optional
        Name of the constraint for debugging.
    """
    # Convert coefficients and rhs to integers with proper scaling
    # Toulbar2 works with integer costs, so we need to scale floating point values
    int_coefficients, int_rhs, scale_factor = _scale_to_integers(coefficients, rhs)

    assert len(variables) == len(int_coefficients)
    variables_ = []
    for i, v in enumerate(variables):
        index, value, _ = v
        variables_.append((index, value, int_coefficients[i]))

    try:
        model.AddGeneralizedLinearConstraint(variables_, operator, int_rhs)
    except AttributeError:
        raise NotImplementedError(
            f"Toulbar2 model does not have AddGeneralizedLinearConstraint method. "
            f"Please ensure you are using a compatible version of pytoulbar2. "
            f"Constraint: {name}, vars={variables}, coefs={int_coefficients}, "
            f"op={operator}, rhs={int_rhs}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add constraint '{name}' to Toulbar2 model: {e}")
