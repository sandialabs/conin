import os
import gzip

from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.utils

with try_import() as pgmpy_readwrite_available:
    from pgmpy.readwrite.BIF import BIFReader
    from pgmpy.readwrite.UAI import UAIReader


def load_model(name, quiet=True):

    if os.path.exists(name):
        assert (
            pgmpy_readwrite_available
        ), "Only call conin.common.pgmpy.load_model() if pgmpy.readwrite is available."

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
                assert content is not None, f"Error loading model with pgmpy: {name}"

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

        raise RuntimeError("Unexpected model type: {name}")

    if not quiet:  # pragma:nocover
        print(f"  Loading model pgmpy examples: {name}")
    return pgmpy.utils.get_example_model(name)
