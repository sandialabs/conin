from conin.util import try_import
import conin.common.conin

with try_import() as pgmpy_available:
    import pgmpy
    import conin.common.pgmpy

with try_import() as pomegranate_available:
    import pomegranate
    import conin.common.pomegranate


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

    raise RuntimeError(f"Unexpected model type: {model_type}")
