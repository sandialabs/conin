try:
    from pgmpy.models import MarkovNetwork

    pgmpy_available = True
except Exception as e:
    print(f"pgmpy not available: {e}")
    pgmpy_available = False

from conin.markov_network.inference import create_MN_map_query_model


class ConstrainedMarkovNetwork:

    def __init__(self, pgm, constraints=None):
        assert pgmpy_available and isinstance(
            pgm, MarkovNetwork
        ), "Argument must be a pgmpy MarkovNetwork"
        self.pgm = pgm
        self.constraint_functor = constraints

    def check_model(self):
        self.pgm.check_model()

    def nodes(self):
        return self.pgm.nodes()

    @property
    def constraints(self, constraint_functor):
        self.constraint_functor = constraint_functor

    def create_constraints(self, model):
        if self.constraint_functor is not None:
            model = self.constraint_functor(model)
        return model

    def create_map_query_model(self, variables=None, evidence=None):
        model = create_MN_map_query_model(
            pgm=self.pgm, variables=variables, evidence=evidence
        )
        return self.create_constraints(model)
