from conin.util import try_import
import conin.util.conin

with try_import() as pgmpy_available:
    import pgmpy
    import conin.util.pgmpy


def load_model(name, model_type="conin", quiet=True):

    if model_type == "conin":
        conin.util.conin.load_model(name, quiet=quiet)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(f"Missing import pgmpy, which is required to load a pgmpy model.")
        conin.util.pgmpy.load_model(name, quiet=quiet)

    raise RuntimeError(f"Unexpected model type: {model_type}")
