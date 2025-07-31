import munch

try:
    import pgmpy.models

    pgmpy_available = True
except Exception as e:
    pgmpy_available = False

from conin.bayesian_network.inference import create_BN_map_query_model


class ConstrainedDiscreteBayesianNetwork:

    def __init__(self, pgm, constraints=None):
        assert pgmpy_available and isinstance(
            pgm, pgmpy.models.DiscreteBayesianNetwork
        ), "Argument must be a pgmpy DiscreteBayesianNetwork"
        self.pgm = pgm
        self.constraint_functor = constraints

    def check_model(self):
        self.pgm.check_model()

    def nodes(self):
        return self.pgm.nodes()

    @property
    def constraints(self, constraint_functor):
        self.constraint_functor = constraint_functor

    def create_constraints(self, model, data):
        if self.constraint_functor is not None:
            model = self.constraint_functor(model, data)
        return model

    def create_map_query_model(
        self, variables=None, evidence=None, timing=False, **options
    ):
        model = create_BN_map_query_model(
            pgm=self.pgm,
            variables=variables,
            evidence=evidence,
            timing=timing,
            **options,
        )
        self.data = munch.Munch(
            variables=variables,
            evidence=evidence,
        )
        return self.create_constraints(model, self.data)
