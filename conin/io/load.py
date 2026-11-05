from conin.exceptions import InvalidInputError

from .uai import load_uai
from .cfn import load_cfn
from .json_format import load_json
from .text import read_text
from .pgmpy import convert_pgmpy_to_conin
from .pgmpy import files as pgmpy_files
from .pgmpy.files import pgmpy_available

# The CONIN loaders, by suffix: each takes string= (and the options).
_LOADERS = {
    ".uai": lambda string, **options: load_uai(string=string, **options),
    ".cfn": lambda string, **options: load_cfn(string=string, **options),
    ".json": lambda string, **options: load_json(string=string, **options),
}


def load_model(name, model_type="conin", quiet=True, **options):
    """Load a probabilistic graphical model from a file.

    Parameters
    ----------
    name : str
        Model file name.  CONIN models can be loaded from ``.uai``, ``.bif``
        (requires pgmpy), Toulbar2 ``.cfn`` and CONIN ``.json`` (hidden
        Markov models) files, optionally gzipped (``.uai.gz``, ...).
    model_type : str, optional
        The type of model to create: ``"conin"`` (default) or ``"pgmpy"``.
    quiet : bool, optional
        If False, print diagnostic information.
    **options
        Format-specific options for CONIN models.  CFN files accept
        ``cost_scale``; see
        :func:`conin.io.cfn.load_cfn`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    InvalidInputError
        For an unknown ``model_type`` or file suffix.
    ModelFileError
        If the file content is malformed or does not describe a valid model.
    ImportError
        For ``model_type="pgmpy"`` or a BIF file without pgmpy.
    """

    if model_type == "conin":
        #
        # For now, we load BIF files using pgmpy and convert the model to conin
        #
        if name.endswith(".bif") or name.endswith(".bif.gz"):
            if not pgmpy_available:
                raise ImportError("Loading a BIF file requires pgmpy.")
            pgm = pgmpy_files.load_model(name, quiet=quiet)
            return convert_pgmpy_to_conin(pgm)

        return load_conin_model(name, quiet=quiet, **options)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                "Missing import pgmpy, which is required to load a pgmpy model."
            )
        return pgmpy_files.load_model(name, quiet=quiet)

    raise InvalidInputError(
        f"Unexpected model type: {model_type!r} (expected 'conin' or 'pgmpy')"
    )


def load_conin_model(name, quiet=True, **options):
    """Load a CONIN model from a UAI, Toulbar2 CFN or CONIN JSON file.

    Parameters
    ----------
    name : str
        Name of a ``.uai``, ``.cfn`` or ``.json`` file, optionally gzipped
        (``.uai.gz``, ``.cfn.gz``, ``.json.gz``).
    quiet : bool, optional
        If False, print diagnostic information.
    **options
        Additional options passed to the format-specific loader.  CFN files
        accept ``cost_scale``; see
        :func:`conin.io.cfn.load_cfn`.

    Raises
    ------
    FileNotFoundError, InvalidInputError, ModelFileError
        As for :func:`load_model`.
    """
    base = name[: -len(".gz")] if name.endswith(".gz") else name
    loader = next((f for s, f in _LOADERS.items() if base.endswith(s)), None)
    if loader is None:
        raise InvalidInputError(
            f"Cannot load a conin model from {name}: unknown file suffix "
            f"(expected one of {', '.join(_LOADERS)}, optionally with .gz)"
        )
    return loader(read_text(name, quiet=quiet), **options)
