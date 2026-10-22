# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


def _missing(name):
    # A broken install (e.g. torch with missing shared libraries) raises OSError.
    try:
        importlib.import_module(name)
    except (ImportError, OSError):
        return True
    return False


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "conin"
copyright = "2025, conin developers"
author = "conin developers"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
]

# Mock only the optional backends that are not installed: mocking an installed
# backend would also replace it in the conin modules that the doctests run.
autodoc_mock_imports = [
    name for name in ("pgmpy", "pytoulbar2", "torch") if _missing(name)
]

# Import the conin modules now, without the mocks.  Otherwise autodoc imports
# some of them under the mocks, and conin's optional-import checks then see a
# mocked backend as installed (e.g. pgmpy classes become mocks), which breaks
# the doctests that run that code.  Modules that need a missing backend (the
# torch MVR modules) fail here and are imported under the mocks by autodoc.
if autodoc_mock_imports:
    import pkgutil

    import conin

    for _info in pkgutil.walk_packages(conin.__path__, "conin."):
        if ".tests" in _info.name or _info.name.startswith("conin.mediation_variables"):
            continue
        try:
            importlib.import_module(_info.name)
        except Exception:
            pass

# The CI docs job sets CONIN_DOCS_REQUIRE_BACKENDS=1 so that a missing backend
# fails the build instead of silently skipping the examples that need it.
# (No example needs torch; its API pages build with the mock.)
if os.environ.get("CONIN_DOCS_REQUIRE_BACKENDS") == "1":
    _required = [name for name in ("pgmpy", "pytoulbar2") if _missing(name)]
    if _required:
        raise RuntimeError(
            "CONIN_DOCS_REQUIRE_BACKENDS=1, but these backends are missing: "
            + ", ".join(_required)
        )
    from conin.config import default_mip_solver as _mip_solver

    if _mip_solver is None:
        raise RuntimeError(
            "CONIN_DOCS_REQUIRE_BACKENDS=1, but no MIP solver is available"
        )

# Names for the ``:skipif:`` options of the doctest blocks that need an
# optional backend.  Those blocks are skipped when it is missing, so the docs
# also build without the backends.
doctest_global_setup = """
def _available(name):
    import importlib
    try:
        importlib.import_module(name)
    except (ImportError, OSError):
        return False
    return True

pgmpy_available = _available("pgmpy")
toulbar2_available = _available("pytoulbar2")

from conin.config import default_mip_solver
mip_solver_available = default_mip_solver is not None

# Examples that write files run in a temporary directory.
import os as _os, tempfile as _tempfile
_doc_cwd = _os.getcwd()
_doc_tmp = _tempfile.TemporaryDirectory()
_os.chdir(_doc_tmp.name)
"""

doctest_global_cleanup = """
_os.chdir(_doc_cwd)
_doc_tmp.cleanup()
"""

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
