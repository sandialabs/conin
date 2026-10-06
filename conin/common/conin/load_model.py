import os
import gzip
from .load_uai import load_conin_model_from_uai
from .load_cfn import load_conin_model_from_cfn


def load_model(name, quiet=True, **options):
    """Load a CONIN model from a UAI or Toulbar2 CFN file.

    Parameters
    ----------
    name : str
        Name of a ``.uai``, ``.uai.gz``, ``.cfn`` or ``.cfn.gz`` file.
    quiet : bool, optional
        If False, print diagnostic information.
    **options
        Additional options passed to the format-specific loader.  CFN files
        accept ``cost_scale``; see
        :func:`conin.common.conin.load_cfn.load_conin_model_from_cfn`.
    """

    if not os.path.exists(name):
        raise RuntimeError(f"Missing file {name}")

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
            assert content is not None, f"Error loading model data from {name}"

            if name.endswith(".uai.gz"):
                return load_conin_model_from_uai(string=content.decode("utf-8"))
            if name.endswith(".cfn.gz"):
                return load_conin_model_from_cfn(
                    string=content.decode("utf-8"), **options
                )

    elif name.endswith(".uai"):
        return load_conin_model_from_uai(filename=name)

    elif name.endswith(".cfn"):
        return load_conin_model_from_cfn(filename=name, **options)

    raise RuntimeError(f"Cannot load conin model from unexpected file: {name}")
