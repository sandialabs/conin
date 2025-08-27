from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.models


def convert_to_DynamicDiscreteBayesianNetwork(pgm):
    return pgm
