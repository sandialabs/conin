import os
import gzip
import re
import itertools
import numpy as np
import pprint

from conin.util import try_import
from .mapcpd import MapCPD

with try_import() as pgmpy_available:
    import pgmpy.utils
    import pgmpy.models
    from pgmpy.factors.discrete import TabularCPD, DiscreteFactor

with try_import() as pgmpy_readwrite_available:
    from pgmpy.readwrite.BIF import BIFWriter
    from pgmpy.readwrite.UAI import UAIWriter


def save_model(pgm, name, quiet=True):

    if name.endswith(".bif"):
        writer = BIFWriter(pgm)
        writer.write(name)

    elif name.endswith(".uai"):
        writer = UAIWriter(pgm)
        writer.write(name)
