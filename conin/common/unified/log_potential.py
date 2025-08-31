from conin.util import try_import
import conin.common.conin

import conin.bayesian_network
import conin.markov_network

with try_import() as pgmpy_available:
    import pgmpy
    import conin.common.pgmpy

with try_import() as pomegranate_available:
    import pomegranate

with try_import() as pgmax_available:
    import pgmax

with try_import() as pyagrum_available:
    import pyagrum


def log_potential(pgm, query_variables, evidence=None, **options):
    """
    A wrapper function that calls log_potential functions for specific libraries that
    conin interfaces with.
    """
    if isinstance(pgm, conin.bayesian_network.DiscreteBayesianNetwork) or isinstance(
        pgm, conin.markov_network.DiscreteMarkovNetwork
    ):
        return conin.common.conin.log_potential(
            pgm, query_variables, evidence=evidence, **options
        )

    if pgmpy_available:
        if isinstance(pgm, pgmpy.models.DiscreteBayesianNetwork) or isinstance(
            pgm, pgmpy.models.MarkovNetwork
        ):
            return conin.common.pgmpy.log_potential(
                pgm, query_variables, evidence=evidence, **options
            )

    if pomegranate_available:
        if isinstance(pgm._pgmpy, pgmpy.models.DiscreteBayesianNetwork) or isinstance(
            pgm._pgmpy, pgmpy.models.MarkovNetwork
        ):
            return conin.common.pgmpy.log_potential(
                pgm._pgmpy, query_variables, evidence=evidence, **options
            )

    if pgmax_available:
        if isinstance(pgm._pgmpy, pgmpy.models.DiscreteBayesianNetwork) or isinstance(
            pgm._pgmpy, pgmpy.models.MarkovNetwork
        ):
            return conin.common.pgmpy.log_potential(
                pgm._pgmpy, query_variables, evidence=evidence, **options
            )

    if pyagrum_available:
        if isinstance(pgm._pgmpy, pgmpy.models.DiscreteBayesianNetwork) or isinstance(
            pgm._pgmpy, pgmpy.models.MarkovNetwork
        ):
            return conin.common.pgmpy.log_potential(
                pgm._pgmpy, query_variables, evidence=evidence, **options
            )

    raise TypeError(f"Unexpected model type: {type(pgm)}")
