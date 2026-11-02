"""Interoperability with pgmpy: BIF and UAI files, and model conversion.

pgmpy is optional: this package imports without it, and the functions raise
``ImportError`` when they need it.  CONIN's algorithms take CONIN models
only; convert a pgmpy model with :func:`convert_pgmpy_to_conin` first.
"""

from .to_conin import convert_pgmpy_to_conin
from .to_pgmpy_bn import convert_conin_to_pgmpy_bn
from .to_pgmpy_mn import convert_conin_to_pgmpy_mn
