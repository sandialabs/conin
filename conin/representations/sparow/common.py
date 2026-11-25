"""Shared Pyomo helpers for the ``"pyomo"`` and ``"pyomo_network_flow"`` representations.

* :class:`PyomoVarWrapper`: the ``model.V(...)`` accessor that user constraint
  functions see (defined in :mod:`conin.util.var_wrapper`, re-exported here);
* :func:`apply_pyomo_constraints`: the single constraint applier for the
  Pyomo-backend families (``"pyomo"`` and ``"smoek"``), including the
  smoek->Pyomo translation (:func:`add_smoek_constraints_to_pyomo_model`);
* :func:`_create_pyomo_map_query_model`: the IP built
  from a factor representation ``(S, J, v, w)``, plus the c5 helpers
  (:func:`add_c5_constraints`, :func:`has_omitted_c5`,
  :class:`C5LazyCutGenerator`);
* :class:`PyomoFactorRepresentation`: the base class of the MN/BN/DBN/HMM
  ``"pyomo"`` representations.
"""

import itertools
from functools import partial

import munch
import pyomo.environ as pyo

import conin.representations.pyomo.common
from conin.representations.base import Representation
from conin.util import State
from conin.util import try_import, State
from conin.util.var_wrapper import PyomoVarWrapper
from conin.util.constraint_data import merge_constraint_data

with try_import() as sparow_available:
    import sparow


class SparowFactorRepresentation(Representation):
    """The factor-repn integer program of a constrained MN/BN/DBN/HMM.

    The representation holds only evidence-independent data: the factor
    representation ``(S, J, v, w)`` of the (unrolled) Markov network, the
    variable-name map and the constraints.
    :meth:`create_model` builds a fresh Pyomo model on every call, fixes the
    evidence and then applies the constraints.

    Attributes
    ----------
    S, J, v, w : dict
        The factor representation (see
        :func:`_create_pyomo_map_query_model`).
    var_index_map : dict or None
        Node -> internal variable name, or ``None`` (nodes used as-is).
    pgm : GraphicalModel
        The model passed to the constraint applier (the MN, BN or unrolled BN/DBN).
    constraints : tuple
        The user constraints (family ``"pyomo"`` or ``"smoek"``).
    nodes : tuple
        The nodes of the (unrolled) model the IP ranges over.
    data : object or None
        The constrained model's application data.  The constraint functions
        see a copy of it with the fields of :meth:`constraint_data` set.

    Notes
    -----
    Nothing is mutated after construction, so one representation can serve
    several queries.  The factor dicts are shared with every model built from
    them, but the IP builder only reads them.
    """

    kind = "sparow"

    def __init__(
        self, *, S, J, v, w, var_index_map, pgm, constraints, nodes, data=None
    ):
        if not sparow_available:
            raise RuntimeError(
                "The sparow package must be installed to solve MAP inference with integer_program."
            )

        self.S = S
        self.J = J
        self.v = v
        self.w = w
        self.var_index_map = var_index_map
        self.pgm = pgm
        self.constraints = tuple(constraints)
        self.nodes = tuple(nodes)
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

    def create_model(self, *, evidence=None, variables=None, timing=False):
        """Build a fresh Pyomo model for one query.

        Parameters
        ----------
        evidence : dict, optional
            ``{node: state}``; each ``model.V(node, state)`` is fixed to 1.
        variables : list, optional
            Passed to the constraint functions as ``data.variables``.
        omit_c5 : bool, optional
            Omit the c5 constraints (they are generated as cuts with gurobi).
        timing : bool, optional
            Print timing information.

        Returns
        -------
        pyomo.environ.ConcreteModel
        """
        #
        # Configure the scenario builder
        #
        _scenario_builder = partial(
            conin.representations.pyomo.common._create_pyomo_map_query_model,
            S=self.S,
            J=self.J,
            v=self.v,
            w=self.w,
            var_index_map=self.var_index_map,
            timing=timing,
        )

        def scenario_builder(data, args):
            model = _scenario_builder()

            for k, val in data.items():
                if k == "ID":
                    continue
                model.V(k, State(val)).fix(1)
            return model

        #
        # Create scenario data
        # For now, we exhaustively sample all marginalized variables
        # NOTE: We treat each of these as equally likely, which seems correct when
        #   enumerating all possible values.
        #
        marginalized = set(self.pgm.nodes) - set(variables)
        if evidence:
            marginalized = marginalized - set(evidence)
        possible_values = {n: self.pgm.states_of(n) for n in marginalized}
        keys, values = zip(*possible_values.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        scenarios = dict(
            scenarios=[dict(ID=i, data=d) for i, d in enumerate(permutations)]
        )

        #
        # Setup the Sparow model, and create the Pyomo extensive form.
        #
        sp = sparow.sp.stochastic_program(
            first_stage_variables=[f"x[{v},*]" for v in variables]
        )
        sp.add_scenarios(data=scenarios, scenario_builder=scenario_builder)
        model = sp.create_EF()
        #
        # We are re-using logic from the pyomo respresentation, so we need to setup a
        # variable wrapper, V, and an expression that captures the objective, o.
        #
        if self.var_index_map is None:
            RS = [(r, s) for r, values in self.S.items() for s in values]
            model.V = PyomoVarWrapper({rs: model.s[None, 0].x[rs] for rs in RS})
        else:
            model.V = PyomoVarWrapper(
                {
                    (r, s): model.s[None, 0].x[index, s]
                    for r, index in self.var_index_map.items()
                    for s in self.S.get(index, [])
                }
            )
        model.o = pyo.Expression(expr=model.obj)
        #
        # Fix evidence variables
        #
        if evidence:
            for k, val in evidence.items():
                model.V(k, State(val)).fix(1)
        #
        # Apply pyomo and smoek constraints
        #
        if self.constraints:
            conin.representations.pyomo.common.apply_pyomo_constraints(
                pgm=self.pgm,
                constraints=self.constraints,
                model=model,
                data=merge_constraint_data(
                    self.data,
                    self.constraint_data(evidence=evidence, variables=variables),
                ),
            )

        return model
