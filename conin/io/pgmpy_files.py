"""Load and save pgmpy models from BIF and UAI files (requires pgmpy)."""

import os
import gzip

from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.utils
    import pgmpy.models

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
                except Exception as e:  # pragma:nocover
                    if not quiet:
                        print(f"Error reading file {name}: {e}")
                    content = None
                if content is None:  # pragma:nocover
                    raise RuntimeError(f"Error loading model with pgmpy: {name}")

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

        raise RuntimeError(f"Unexpected model type: {name}")

    if not quiet:  # pragma:nocover
        print(f"  Loading model pgmpy examples: {name}")
    return pgmpy.utils.get_example_model(name)


def save_model(pgm, name, quiet=True):

    if name.endswith(".bif"):
        if not pgmpy_readwrite_available:
            raise RuntimeError(
                "Cannot save a pgmpy model to BIF format without importing pgmpy"
            )
        writer = BIFWriter(pgm)
        writer.write(name)

    elif name.endswith(".uai"):
        if not pgmpy_readwrite_available:
            raise RuntimeError(
                "Cannot save a pgmpy model to UAI format without importing pgmpy"
            )
        writer = UAIWriter(pgm)
        writer.write(name)

    else:
        raise RuntimeError(
            f"Cannot save a pgmpy model.  Unknown format specified by filename suffix: {name}"
        )
