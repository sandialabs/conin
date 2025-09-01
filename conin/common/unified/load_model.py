from conin.util import try_import
import conin.common.conin

with try_import() as pgmpy_available:
    import pgmpy
    import conin.common.pgmpy

with try_import() as pomegranate_available:
    import pomegranate
    import conin.common.pomegranate

with try_import() as pgmax_available:
    import pgmax
    import conin.common.pgmax

with try_import() as pyagrum_available:
    import pyagrum
    import conin.common.pyagrum


def load_model(name, model_type="conin", quiet=True):

    if model_type == "conin":
        #
        # For now, we load BIF files using pgmpy and convert the model to conin
        #
        if (name.endswith(".bif") or name.endswith("bif.gz")) and pgmpy_available:
            pgm = conin.common.pgmpy.load_model(name, quiet=quiet)
            return conin.common.pgmpy.convert_pgmpy_to_conin(pgm)

        return conin.common.conin.load_model(name, quiet=quiet)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                f"Missing import pgmpy, which is required to load a pgmpy model."
            )
        return conin.common.pgmpy.load_model(name, quiet=quiet)

    elif model_type == "pomegranate":
        if not pomegranate_available:
            raise ImportError(
                f"Missing import pomegranate, which is required to load a pomegranate model."
            )
        return conin.common.pomegranate.load_model(name, quiet=quiet)

    elif model_type == "pgmax":
        if not pgmax_available:
            raise ImportError(
                f"Missing import pgmax, which is required to load a pgmax model."
            )
        return conin.common.pgmax.load_model(name, quiet=quiet)

    elif model_type == "pyagrum":
        if not pyagrum_available:
            raise ImportError(
                f"Missing import pyagrum, which is required to load a pyagrum model."
            )
        return conin.common.pyagrum.load_model(name, quiet=quiet)

    raise RuntimeError(f"Unexpected model type: {model_type}")
