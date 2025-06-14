import munch

try:
    from pgmpy.models import DiscreteBayesianNetwork

    pgmpy_available = True
except Exception as e:
    print(f"pgmpy not available: {e}")
    pgmpy_available = False

from conin.bayesian_network.inference import create_BN_map_query_model


class ConstrainedDiscreteBayesianNetwork:

    def __init__(self, pgm, constraints=None):
        assert pgmpy_available and isinstance(
            pgm, DiscreteBayesianNetwork
        ), "Argument must be a pgmpy DiscreteBayesianNetwork"
        self.pgm = pgm
        self.constraint_functor = constraints

    @property
    def constraints(self, constraint_functor):
        self.constraint_functor = constraint_functor

    def create_constraints(self, model, data):
        if self.constraint_functor is not None:
            model = self.constraint_functor(model, data)
        return model

    def create_map_query_model(self, variables=None, evidence=None):
        model = create_BN_map_query_model(pgm=self.pgm)
        self.data = munch.Munch(
            variables=variables,
            evidence=evidence,
        )
        return self.create_constraints(model, self.data)
