try:
    import pgmpy
    import conin.util.pgmpy
    pgmpy_available=True
except:
    pgmpy_available=False


def log_potential(pgm, query_variables, evidence=None, options**):
    if pgmpy_available:
        if isinstance(pgm, pgmpy.models.DiscreteBayesianNetwork) or isinstance(pgm, pgmpy.models.MarkovNetwork):
            return conin.util.pgmpy.log_potential(pgm, query_variables, evidence=evidence, **options)

    raise TypeError(f"Unexpected model type: {type(pgm)}")
