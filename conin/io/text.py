"""Reading the text of a model file, gzipped or not."""

import gzip
import os

from conin.exceptions import ModelFileError


def read_text(name, quiet=True, missing="Missing file"):
    """The text of ``name``, decompressed if it ends with ``.gz``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (the message starts with ``missing``).
    ModelFileError
        If a ``.gz`` file cannot be decompressed or the text is not UTF-8.
    """
    name = os.fspath(name)
    if not os.path.exists(name):
        raise FileNotFoundError(f"{missing} {name}")
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
