import os
import gzip

from conin.util import try_import
from .uai import load_conin_model_from_uai
from .cfn import load_conin_model_from_cfn

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401
    from conin.pgmpy import convert_pgmpy_to_conin
    from . import pgmpy_files


def load_model(name, model_type="conin", quiet=True, **options):
    """Load a probabilistic graphical model from a file.

    Parameters
    ----------
    name : str
        Model file name.  CONIN models can be loaded from ``.uai``, ``.bif``
        (requires pgmpy) and Toulbar2 ``.cfn`` files, optionally gzipped.
    model_type : str, optional
        The type of model to create: ``"conin"`` (default) or ``"pgmpy"``.
    quiet : bool, optional
        If False, print diagnostic information.
    **options
        Format-specific options for CONIN models.  CFN files accept
        ``cost_scale``; see
        :func:`conin.io.cfn.load_conin_model_from_cfn`.
    """

    if model_type == "conin":
        #
        # For now, we load BIF files using pgmpy and convert the model to conin
        #
        if (name.endswith(".bif") or name.endswith("bif.gz")) and pgmpy_available:
            pgm = pgmpy_files.load_model(name, quiet=quiet)
            return convert_pgmpy_to_conin(pgm)

        return load_conin_model(name, quiet=quiet, **options)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                "Missing import pgmpy, which is required to load a pgmpy model."
            )
        return pgmpy_files.load_model(name, quiet=quiet)

    raise RuntimeError(f"Unexpected model type: {model_type}")


def load_conin_model(name, quiet=True, **options):
    """Load a CONIN model from a UAI or Toulbar2 CFN file.

    Parameters
    ----------
    name : str
        Name of a ``.uai``, ``.uai.gz``, ``.cfn`` or ``.cfn.gz`` file.
    quiet : bool, optional
        If False, print diagnostic information.
    **options
        Additional options passed to the format-specific loader.  CFN files
        accept ``cost_scale``; see
        :func:`conin.io.cfn.load_conin_model_from_cfn`.
    """

    if not os.path.exists(name):
        raise RuntimeError(f"Missing file {name}")

    if name.endswith(".gz"):
        with gzip.open(name) as INPUT:
            if not quiet:  # pragma:nocover
                print(f"  Loading model from {name}")
            try:
                content = INPUT.read()
            except Exception as e:  # pragma:nocover
                if not quiet:
                    print(f"Error reading file {name}: {e}")
                content = None
            if content is None:  # pragma:nocover
                raise RuntimeError(f"Error loading model data from {name}")

            if name.endswith(".uai.gz"):
                return load_conin_model_from_uai(string=content.decode("utf-8"))
            if name.endswith(".cfn.gz"):
                return load_conin_model_from_cfn(
                    string=content.decode("utf-8"), **options
                )

    elif name.endswith(".uai"):
        return load_conin_model_from_uai(filename=name)

    elif name.endswith(".cfn"):
        return load_conin_model_from_cfn(filename=name, **options)

    raise RuntimeError(f"Cannot load conin model from unexpected file: {name}")
