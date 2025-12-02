from conin.util import try_import
import conin.common.conin

with try_import() as pgmpy_available:
    import pgmpy
    import conin.common.pgmpy

#with try_import() as pomegranate_available:
#    import pomegranate
#    import conin.common.pomegranate

#with try_import() as pgmax_available:
#    import pgmax
#    import conin.common.pgmax

#with try_import() as pyagrum_available:
#    import pyagrum
#    import conin.common.pyagrum


def save_model(pgm, name, model_type="conin", quiet=True):

    if model_type == "conin":
        return conin.common.conin.save_model(pgm, name, quiet=quiet)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                f"Missing import pgmpy, which is required to load a pgmpy model."
            )
        return conin.common.pgmpy.save_model(pgm, name, quiet=quiet)

    raise RuntimeError(f"Unexpected model type: {model_type}")
