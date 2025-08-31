from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy
    import conin.common.pgmpy


def load_model(name, quiet=True):

    assert (
        pgmpy_available
    ), "The pgmpy package must be installed to load a pomegranate model."

    pgmpy_pgm = conin.common.pgmpy.load_model(name, quiet=quiet)
    return conin.common.pgmpy.convert_pgmpy_to_pomegranate(pgmpy_pgm)
