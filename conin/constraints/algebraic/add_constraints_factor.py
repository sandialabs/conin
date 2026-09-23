import pprint
import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables

from conin.constraints.factor import factor_constraint_fn
from .bridge import ConinVarNode
from .decorators import AlgebraicConstraint
from conin.util import try_import, State

with try_import() as smoek_available:
    import smoek


class PyomoVarWrapper(dict):
    def __init__(self, *arg, **kw):
        super(PyomoVarWrapper, self).__init__(*arg, **kw)

    def pprint(self):  # pragma:nocover
        pprint.pprint(self)

    def __call__(self, *args):
        if len(args) == 2:
            r, s = args
        elif len(args) == 3:
            r, i, s = args
            r = (r, i)
        else:
            raise ValueError("There must be either 2 or 3 arguments")

        if type(s) is not State:
            s = State(s)
        return dict.__getitem__(self, (r, s))


def create_factor_constraints_from_algebraic(*, pgm, constraints, data):
    """
    Convert a list of AlgebraicConstraints into a list with a single FactorConstraint.

    This function:
    1. Takes a list of AlgebraicConstraints
    2. Generates a string representation of each constraint
    3. Composes these string representations into a function that returns True
       if all constraints are satisfied

    Parameters:
        algebraic_constraints: A single AlgebraicConstraint or list of AlgebraicConstraints
        name: Optional name for the resulting FactorConstraint

    Returns:
        A FactorConstraint instance that evaluates all the algebraic constraints
    """
    if not smoek_available:
        raise ImportError(
            "smoek is required for create_factor_constraints_from_algebraic. Install with: pip install smoek"
        )

    # Normalize input to list
    if len(constraints) == 0:
        return []

    # Validate all inputs are AlgebraicConstraints
    for i, constraint in enumerate(constraints):
        if not isinstance(constraint, AlgebraicConstraint):
            raise TypeError(
                f"Element {i} is not an AlgebraicConstraint, got {type(constraint)}"
            )

    # Determine function signature (1 arg or 2 args)
    # Check the first constraint to determine if we need data argument
    first_constraint = constraints[0]
    num_args = first_constraint.num_args

    # Collect all nodes referenced in the constraints
    nodes_set = set()

    smoek_model = smoek.model()
    for func in constraints:
        func(smoek_model, data)
    smoek_model._update_smoek_components()

    node_map = pyo.ComponentMap()
    pyomo_model = pyo.ConcreteModel()
    tmp = {}
    for k in pgm.nodes:
        for s in pgm.states_of(k):
            if type(k) is str:
                k_str = f'"{k}"'
            else:
                k_str = str(k)
            if type(s) is str:
                s_str = f'"{s}"'
            else:
                s_str = str(s)
            name = f"m_.V({k_str}, {s_str})"
            setattr(pyomo_model, name, pyo.Var(name=name))
            v = getattr(pyomo_model, name)
            node_map[v] = k
            tmp[k, State(s)] = v
    pyomo_model.V = PyomoVarWrapper(tmp)

    pyomo_model = smoek.pymodel.pyomo.generate(
        model=smoek_model,
        data=data,
        pyomo_model=pyomo_model,
    )

    # CollectWalk the expression trees to generate string representations
    constraint_strings = []
    for constraint_component in pyomo_model.component_objects(
        pyo.Constraint, active=True
    ):
        # Iterate through each constraint data object (handles indexed constraints)
        for index in constraint_component:
            constraint_data = constraint_component[index]

            # Skip inactive constraints
            if not constraint_data.active:
                continue

            constraint_strings.append(str(constraint_data.expr).replace("'", ""))

            for var in identify_variables(constraint_data.expr):
                nodes_set.add(node_map[var])

    # Convert nodes_set to sorted list for deterministic ordering
    nodes = sorted(list(nodes_set))

    # Generate a single Python function as a string
    # This function will be compiled once and can be called many times
    function_code = _generate_constraint_function_code(constraint_strings, num_args)

    # Compile the function code
    namespace = {}
    try:
        exec(function_code, namespace)
        composite_constraint_function = namespace["constraint_function"]
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile constraint function: {e}\n\nGenerated code:\n{function_code}"
        )

    # Create name if not provided
    if name is None:
        constraint_names = [c.name for c in constraints]
        name = f"composite_{'_'.join(constraint_names)}"

    # Use the factor_constraint_fn decorator to create the FactorConstraint
    factor_constraint = factor_constraint_fn(nodes=nodes, name=name)(
        composite_constraint_function
    )

    return factor_constraint


def _collect_conin_nodes_from_expr(expr, nodes_set):
    """
    Recursively walk an expression tree and collect all ConinVarNode references.

    Parameters:
        expr: The expression node to walk
        nodes_set: Set to accumulate node names into
    """
    # If it's a Constraint wrapper, unwrap it
    if hasattr(expr, "_expr"):
        expr = expr._expr

    # Base case: ConinVarNode
    if isinstance(expr, ConinVarNode):
        nodes_set.add(expr.node)
        return

    # Recursive cases: walk children
    if hasattr(expr, "left") and hasattr(expr, "right"):
        # Binary expression node
        _collect_conin_nodes_from_expr(expr.left, nodes_set)
        _collect_conin_nodes_from_expr(expr.right, nodes_set)
    elif hasattr(expr, "arg"):
        # Unary expression node
        _collect_conin_nodes_from_expr(expr.arg, nodes_set)
    elif hasattr(expr, "body"):
        # Inequality or other compound node
        _collect_conin_nodes_from_expr(expr.body, nodes_set)
        if hasattr(expr, "left"):
            _collect_conin_nodes_from_expr(expr.left, nodes_set)
        if hasattr(expr, "right"):
            _collect_conin_nodes_from_expr(expr.right, nodes_set)


def _generate_constraint_function_code(constraint_strings, num_args):
    """
    Generate Python function code that evaluates all constraints.

    Parameters:
        constraint_strings: List of constraint expression strings
        num_args: Number of arguments (1 or 2) for the function

    Returns:
        String containing Python function definition
    """
    # Build the function signature based on num_args
    if num_args == 1:
        signature = "def constraint_function(args):"
    else:
        signature = "def constraint_function(args, data=None):"

    # Build the V() helper function that maps m_.V() calls to actual state checks
    v_helper = '''
    class m_:
        @staticmethod
        def V(node, state, time=None):
            """Return 1 if args[node] == state, else 0"""
            if node not in args:
                raise KeyError(f"Node {node} not found in args")
            return 1 if args.get(node) == state else 0
'''

    # Build the constraint check expression
    # Combine all constraints with 'and' operator
    if len(constraint_strings) == 1:
        constraint_check = f"    return ({constraint_strings[0]})"
    else:
        # Join constraints with ' and \\\n        ' for readability
        constraints_joined = ") and \\\n        (".join(constraint_strings)
        constraint_check = f"    return ({constraints_joined})"

    # Combine all parts
    function_code = f"{signature}\n{v_helper}\n{constraint_check}"

    return function_code
