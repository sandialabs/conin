import gzip

from conin.exceptions import InvalidInputError
from conin.graphical_models import DiscreteBayesianNetwork
from conin.graphical_models import DiscreteMarkovNetwork
from .uai import save_uai
from .json_format import save_json
from .pgmpy import convert_conin_to_pgmpy_bn, convert_conin_to_pgmpy_mn
from .pgmpy import files as pgmpy_files
from .pgmpy.files import pgmpy_available


def save_model(pgm, name, model_type="conin", quiet=True):
    """Save a probabilistic graphical model to a file.

    Parameters
    ----------
    pgm
        The model: a CONIN ``DiscreteBayesianNetwork``,
        ``DynamicDiscreteBayesianNetwork``, ``DiscreteMarkovNetwork`` or
        ``HiddenMarkovModel`` for ``model_type="conin"``, or a pgmpy model
        for ``model_type="pgmpy"``.
    name : str
        Output file name. CONIN models are saved as ``.uai`` files (see
        :func:`conin.io.save_uai`), as ``.json`` or ``.json.gz`` files
        (hidden Markov models; see :func:`conin.io.save_json`) or, with
        pgmpy installed, as ``.bif`` files (converted to pgmpy and written
        by pgmpy).
    model_type : str, optional
        ``"conin"`` (default) or ``"pgmpy"``.
    quiet : bool, optional
        Accepted for API compatibility.

    Raises
    ------
    InvalidInputError
        For an unknown ``model_type`` or file suffix.
    TypeError
        For a model the format cannot hold.
    ImportError
        For ``model_type="pgmpy"`` or a BIF file without pgmpy.
    """

    if model_type == "conin":
        return save_conin_model(pgm, name, quiet=quiet)

    elif model_type == "pgmpy":
        if not pgmpy_available:
            raise ImportError(
                "Missing import pgmpy, which is required to save a pgmpy model."
            )
        return pgmpy_files.save_model(pgm, name, quiet=quiet)

    raise InvalidInputError(
        f"Unexpected model type: {model_type!r} (expected 'conin' or 'pgmpy')"
    )


def save_conin_model(pgm, name, quiet=True):
    if name.endswith(".uai"):
        return save_uai(pgm, name, quiet)

    if name.endswith(".json"):
        return save_json(pgm, name)

    if name.endswith(".json.gz"):
        text = save_json(pgm)
        with gzip.open(name, "wt", encoding="utf-8") as f:
            f.write(text)
        return

    if name.endswith(".bif"):
        if not pgmpy_available:
            raise ImportError("Saving a BIF file requires pgmpy.")
        if isinstance(pgm, DiscreteBayesianNetwork):
            _pgm = convert_conin_to_pgmpy_bn(pgm)
            return pgmpy_files.save_model(_pgm, name, quiet)
        elif isinstance(pgm, DiscreteMarkovNetwork):
            _pgm = convert_conin_to_pgmpy_mn(pgm)
            return pgmpy_files.save_model(_pgm, name, quiet)
        else:
            raise TypeError(
                "Cannot save a conin model to BIF: expected a "
                f"DiscreteBayesianNetwork or DiscreteMarkovNetwork, got "
                f"{type(pgm).__name__}"
            )

    raise InvalidInputError(
        f"Cannot save conin model.  Unknown format specified by filename suffix: {name}"
    )
