import os
import gzip
from .load_uai import load_pgmpy_model_from_uai

from conin.util import try_import

with try_import() as pgmpy_available:
    import pgmpy.utils

def load_model(name, quiet=True):

    if os.path.exists(name):
        if name.endswith(".gz"):
            with gzip.open(name) as INPUT:
                if not quiet:
                    print(f"  Loading model from {name}")
                try:
                    content = INPUT.read()
                except Exception as e:
                    if not quiet:
                        print(f"Error reading file {name}: {e}")
                    content = None
                assert content is not None, f"Error loading model with pgmpy: {name}"

                if name.endswith(".bif.gz"):
                    reader = BIFReader(string=content.decode("utf-8"))
                    return reader.get_model()
                elif name.endswith(".uai.gz"):
                    return load_pgmpy_model_from_uai(string=content.decode("utf-8"))


        elif name.endswith(".bif"):
            reader = BIFReader(name)
            pgm = reader.get_model()

        elif name.endswith(".uai"):
            pgm = load_pgmpy_model_from_uai(filename=name)

    if not quiet:
        print(f"  Loading model pgmpy examples: {name}")
    return pgmpy.utils.get_example_model(name)
