import gzip

from conin.exceptions import InvalidInputError
from .uai import save_uai
from .json_format import save_json
from .bif import save_bif


def save_model(pgm, name, model_type="conin", quiet=True):
    """Save a probabilistic graphical model to a file.

    Parameters
    ----------
    pgm
        The model: a CONIN ``DiscreteBayesianNetwork``,
        ``DynamicDiscreteBayesianNetwork``, ``DiscreteMarkovNetwork`` or
        ``HiddenMarkovModel``.
    name : str
        Output file name. CONIN models are saved as ``.uai`` files (see
        :func:`conin.io.save_uai`), as ``.json`` or ``.json.gz`` files
        (hidden Markov models; see :func:`conin.io.save_json`) or as
        ``.bif`` or ``.bif.gz`` files (Bayesian networks; see
        :func:`conin.io.save_bif`).
    model_type : str, optional
        Only ``"conin"`` (the default).
    quiet : bool, optional
        Accepted for API compatibility.

    Raises
    ------
    InvalidInputError
        For an unknown ``model_type`` or file suffix, or (BIF) node or
        state names that can't be told apart once written.
    TypeError
        For a model the format cannot hold.
    """

    if model_type == "conin":
        return save_conin_model(pgm, name, quiet=quiet)

    raise InvalidInputError(f"Unexpected model type: {model_type!r} (expected 'conin')")


def save_conin_model(pgm, name, quiet=True):
    if name.endswith(".uai"):
        return save_uai(pgm, name, quiet)

    if name.endswith(".json"):
        return save_json(pgm, name)

    for suffix, save in ((".json", save_json), (".bif", save_bif)):
        if name.endswith(suffix + ".gz"):
            text = save(pgm)
            with gzip.open(name, "wt", encoding="utf-8") as f:
                f.write(text)
            return

    if name.endswith(".bif"):
        return save_bif(pgm, name)

    raise InvalidInputError(
        f"Cannot save conin model.  Unknown format specified by filename suffix: {name}"
    )
