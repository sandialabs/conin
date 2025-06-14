try:
    from pgmpy.models import DynamicBayesianNetwork

    pgmpy_available = True
except Exception as e:
    print(f"pgmpy not available: {e}")
    pgmpy_available = False

from conin.dynamic_bayesian_network.inference import create_DBN_map_query_model


class ConstrainedDynamicBayesianNetwork:

    def __init__(self, pgm, constraints=None):
        assert pgmpy_available and isinstance(
            pgm, DynamicBayesianNetwork
        ), "Argument must be a pgmpy DynamicBayesianNetwork"
        self.pgm = pgm
        self.constraint_functor = constraints

    @property
    def constraints(self, constraint_functor):
        self.constraint_functor = constraint_functor

    def create_constraints(self, model):
        if self.constraint_functor is not None:
            model = self.constraint_functor(model)
        return model

    def create_map_query_model(self):
        model = create_DBN_map_query_model(pgm=self.pgm)
        return self.create_constraints(model)
