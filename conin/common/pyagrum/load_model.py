import gzip
import tempfile
from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy  # noqa: F401
    import conin.common.pgmpy

with try_import() as pyagrum_available:
    import pyagrum


def load_model(name, quiet=True):
    if not pyagrum_available:
        raise ImportError(
            "The pyagrum package must be installed to load a pyagrum model."
        )

    if not pgmpy_available:
        raise ImportError(
            "The pgmpy package must be installed to load a pyagrum model."
        )

    if name.endswith(".gz"):
        suffix = name.split(".")[-2]

        with tempfile.NamedTemporaryFile(suffix="." + suffix) as tmp_file:
            with gzip.open(name, "rt") as f:
                tmp_file.write(
                    f.read().encode("utf-8")
                )  # Write the decompressed content to the temp file
                tmp_filename = tmp_file.name

                # Load the BayesNet from the temporary file
                pgm = pyagrum.loadBN(tmp_filename)
                pgm._pgmpy = conin.common.pgmpy.load_model(name)
    else:
        pgm = pyagrum.loadBN(name)
        pgm._pgmpy = conin.common.pgmpy.load_model(name)

    return pgm
