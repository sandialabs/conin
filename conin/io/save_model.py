from conin.util import try_import
from conin.graphical_models import DiscreteBayesianNetwork
from conin.graphical_models import DiscreteMarkovNetwork
from .uai import save_model_uai

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401
    from conin.common.pgmpy import (
        convert_conin_to_pgmpy_bn,
        convert_conin_to_pgmpy_mn,
    )
    from . import pgmpy_files


def save_model(pgm, name, model_type="conin", quiet=True):

    if model_type == "conin":
        return save_conin_model(pgm, name, quiet=quiet)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                "Missing import pgmpy, which is required to load a pgmpy model."
            )
        return pgmpy_files.save_model(pgm, name, quiet=quiet)

    raise RuntimeError(f"Unexpected model type: {model_type}")


def save_conin_model(pgm, name, quiet=True):
    if name.endswith(".uai"):
        return save_model_uai(pgm, name, quiet)

    if name.endswith(".bif") and pgmpy_available:
        if isinstance(pgm, DiscreteBayesianNetwork):
            _pgm = convert_conin_to_pgmpy_bn(pgm)
            return pgmpy_files.save_model(_pgm, name, quiet)
        elif isinstance(pgm, DiscreteMarkovNetwork):
            _pgm = convert_conin_to_pgmpy_mn(pgm)
            return pgmpy_files.save_model(_pgm, name, quiet)
        else:
            raise RuntimeError(
                f"Cannot save conin model.  Unexpected conin model type {type(pgm)}"
            )

    raise RuntimeError(
        f"Cannot save conin model.  Unknown format specified by filename suffix: {name}"
    )
