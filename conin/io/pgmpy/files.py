"""Load and save pgmpy models from BIF and UAI files (requires pgmpy)."""

import os
import gzip

from conin.exceptions import InvalidInputError, ModelFileError
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401

with try_import() as pgmpy_readwrite_available:
    from pgmpy.readwrite.BIF import BIFReader, BIFWriter
    from pgmpy.readwrite.UAI import UAIReader, UAIWriter


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
                    reader = BIFReader(string=content.decode("utf-8"))
                    return reader.get_model()
                elif name.endswith(".uai.gz"):
                    reader = UAIReader(string=content.decode("utf-8"))
                    return reader.get_model()

        elif name.endswith(".bif"):
            reader = BIFReader(name)
            return reader.get_model()

        elif name.endswith(".uai"):
            reader = UAIReader(name)
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
        writer = BIFWriter(pgm)
        writer.write(name)

    elif name.endswith(".uai"):
        if not pgmpy_readwrite_available:
            raise ImportError(
                "Cannot save a pgmpy model to UAI format without importing pgmpy"
            )
        writer = UAIWriter(pgm)
        writer.write(name)

    else:
        raise InvalidInputError(
            f"Cannot save a pgmpy model.  Unknown format specified by filename suffix: {name}"
        )
