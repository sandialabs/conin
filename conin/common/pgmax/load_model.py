from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401
    import conin.common.pgmpy  # noqa: F401


def load_model(name, quiet=True):

    assert pgmpy_available, "The pgmpy package must be installed to load a pgmax model."

    pgmpy_pgm = conin.common.pgmpy.load_model(name, quiet=quiet)
    pgm = conin.common.pgmpy.convert_pgmpy_to_pgmax(pgmpy_pgm)
    pgm._pgmpy = pgmpy_pgm
    return pgm
