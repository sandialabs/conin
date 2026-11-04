import os
import gzip

from conin.exceptions import InvalidInputError, ModelFileError

from .uai import load_uai
from .cfn import load_cfn
from .json_format import load_json
from .bif import load_bif

# The CONIN loaders, by suffix: each takes string= (and the options).
_LOADERS = {
    ".uai": lambda string, **options: load_uai(string=string, **options),
    ".cfn": lambda string, **options: load_cfn(string=string, **options),
    ".json": lambda string, **options: load_json(string=string, **options),
    ".bif": lambda string, **options: load_bif(string=string, **options),
}


def load_model(name, model_type="conin", quiet=True, **options):
    """Load a probabilistic graphical model from a file.

    Parameters
    ----------
    name : str
        Model file name.  CONIN models can be loaded from ``.uai``, ``.bif``
        (Bayesian networks; see :func:`conin.io.load_bif`), Toulbar2
        ``.cfn`` and CONIN ``.json`` (hidden Markov models) files,
        optionally gzipped (``.uai.gz``, ...).
    model_type : str, optional
        The type of model to create: only ``"conin"`` (the default).
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
    NotImplementedError
        For a BIF variable that is not discrete.
    """

    if model_type == "conin":
        return load_conin_model(name, quiet=quiet, **options)

    raise InvalidInputError(f"Unexpected model type: {model_type!r} (expected 'conin')")


def read_text(name, quiet=True):
    """The text of ``name``, decompressed if it ends with ``.gz``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ModelFileError
        If a ``.gz`` file cannot be decompressed or the text is not UTF-8.
    """
    if not os.path.exists(name):
        raise FileNotFoundError(f"Missing file {name}")
    if not quiet:  # pragma:nocover
        print(f"  Loading model from {name}")
    try:
        if name.endswith(".gz"):
            with gzip.open(name) as INPUT:
                return INPUT.read().decode("utf-8")
        with open(name, encoding="utf-8") as INPUT:
            return INPUT.read()
    except (OSError, EOFError, UnicodeDecodeError) as e:
        raise ModelFileError(f"Cannot read {name}: {e}") from e


def load_conin_model(name, quiet=True, **options):
    """Load a CONIN model from a UAI, BIF, Toulbar2 CFN or CONIN JSON file.

    Parameters
    ----------
    name : str
        Name of a ``.uai``, ``.bif``, ``.cfn`` or ``.json`` file, optionally
        gzipped (``.uai.gz``, ``.bif.gz``, ``.cfn.gz``, ``.json.gz``).
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
