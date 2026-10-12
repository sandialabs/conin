from conin.util import try_import
import conin.common.conin

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401
    import conin.common.pgmpy  # noqa: F401

with try_import() as pomegranate_available:
    import pomegranate  # noqa: F401
    import conin.common.pomegranate  # noqa: F401

with try_import() as pgmax_available:
    import pgmax  # noqa: F401
    import conin.common.pgmax  # noqa: F401

with try_import() as pyagrum_available:
    import pyagrum  # noqa: F401
    import conin.common.pyagrum  # noqa: F401


def load_model(name, model_type="conin", quiet=True, **options):
    """Load a probabilistic graphical model from a file.

    Parameters
    ----------
    name : str
        Model file name.  CONIN models can be loaded from ``.uai``, ``.bif``
        (requires pgmpy) and Toulbar2 ``.cfn`` files, optionally gzipped.
    model_type : str, optional
        The type of model to create: ``"conin"`` (default), ``"pgmpy"``,
        ``"pomegranate"``, ``"pgmax"`` or ``"pyagrum"``.
    quiet : bool, optional
        If False, print diagnostic information.
    **options
        Format-specific options for CONIN models.  CFN files accept
        ``cost_scale``; see
        :func:`conin.common.conin.load_cfn.load_conin_model_from_cfn`.
    """

    if model_type == "conin":
        #
        # For now, we load BIF files using pgmpy and convert the model to conin
        #
        if (name.endswith(".bif") or name.endswith("bif.gz")) and pgmpy_available:
            pgm = conin.common.pgmpy.load_model(name, quiet=quiet)
            return conin.common.pgmpy.convert_pgmpy_to_conin(pgm)

        return conin.common.conin.load_model(name, quiet=quiet, **options)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                "Missing import pgmpy, which is required to load a pgmpy model."
            )
        return conin.common.pgmpy.load_model(name, quiet=quiet)

    elif model_type == "pomegranate":
        if not pomegranate_available:
            raise ImportError(
                "Missing import pomegranate, which is required to load a pomegranate model."
            )
        return conin.common.pomegranate.load_model(name, quiet=quiet)

    elif model_type == "pgmax":
        if not pgmax_available:
            raise ImportError(
                "Missing import pgmax, which is required to load a pgmax model."
            )
        return conin.common.pgmax.load_model(name, quiet=quiet)

    elif model_type == "pyagrum":
        if not pyagrum_available:
            raise ImportError(
                "Missing import pyagrum, which is required to load a pyagrum model."
            )
        return conin.common.pyagrum.load_model(name, quiet=quiet)

    raise RuntimeError(f"Unexpected model type: {model_type}")
