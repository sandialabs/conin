"""Load and save pgmpy models from BIF and UAI files (requires pgmpy)."""

import os
import gzip

from conin.exceptions import InvalidInputError, ModelFileError
from conin.util import lazy_import

# pgmpy is imported when it is first used (it is slow to import).
pgmpy_available = lazy_import("pgmpy")
pgmpy_readwrite_available = lazy_import("pgmpy.readwrite")
pgmpy_bif = lazy_import("pgmpy.readwrite.BIF")
pgmpy_uai = lazy_import("pgmpy.readwrite.UAI")


def load_model(name, quiet=True):
    if not pgmpy_available:
        raise ImportError("The pgmpy package must be installed to load a pgmpy model.")

    if os.path.exists(name):
        if not pgmpy_readwrite_available:
            raise ImportError(
                "The pgmpy.readwrite package must be available to load pgmpy model files."
            )

        if name.endswith(".gz"):
            with gzip.open(name) as INPUT:
                if not quiet:  # pragma:nocover
                    print(f"  Loading model from {name}")
                try:
                    content = INPUT.read()
                except (OSError, EOFError) as e:  # pragma:nocover
                    raise ModelFileError(f"Cannot read {name}: {e}") from e

                if name.endswith(".bif.gz"):
                    reader = pgmpy_bif.BIFReader(string=content.decode("utf-8"))
                    return reader.get_model()
                elif name.endswith(".uai.gz"):
                    reader = pgmpy_uai.UAIReader(string=content.decode("utf-8"))
                    return reader.get_model()

        elif name.endswith(".bif"):
            reader = pgmpy_bif.BIFReader(name)
            return reader.get_model()

        elif name.endswith(".uai"):
            reader = pgmpy_uai.UAIReader(name)
            return reader.get_model()

        raise InvalidInputError(
            f"Cannot load a pgmpy model from {name}: unknown file suffix "
            "(expected .bif or .uai, optionally with .gz)"
        )

    raise FileNotFoundError(f"Missing file {name}")


def save_model(pgm, name, quiet=True):

    if name.endswith(".bif"):
        if not pgmpy_readwrite_available:
            raise ImportError(
                "Cannot save a pgmpy model to BIF format without importing pgmpy"
            )
        writer = pgmpy_bif.BIFWriter(pgm)
        writer.write(name)

    elif name.endswith(".uai"):
        if not pgmpy_readwrite_available:
            raise ImportError(
                "Cannot save a pgmpy model to UAI format without importing pgmpy"
            )
        writer = pgmpy_uai.UAIWriter(pgm)
        writer.write(name)

    else:
        raise InvalidInputError(
            f"Cannot save a pgmpy model.  Unknown format specified by filename suffix: {name}"
        )
