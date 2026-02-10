from conin.util import try_import
import conin.common.conin

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401
    import conin.common.pgmpy  # noqa: F401


def save_model(pgm, name, model_type="conin", quiet=True):

    if model_type == "conin":
        return conin.common.conin.save_model(pgm, name, quiet=quiet)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                "Missing import pgmpy, which is required to load a pgmpy model."
            )
        return conin.common.pgmpy.save_model(pgm, name, quiet=quiet)

    raise RuntimeError(f"Unexpected model type: {model_type}")
