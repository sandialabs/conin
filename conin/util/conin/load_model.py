import os
import gzip
from .load_uai import load_pgmpy_model_from_uai


def load_model(name, quiet=True):

    if not os.path.exists(name):
        raise RuntimeError(f"Missing file {name}")

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
            assert content is not None, f"Error loading model data from {name}"

            if name.endswith(".uai.gz"):
                return load_conin_model_from_uai(string=content.decode("utf-8"))


    elif name.endswith(".uai"):
        return load_conin_model_from_uai(filename=name)

    raise RuntimeError(f"Cannot load conin model from unexpected file: {name}")
