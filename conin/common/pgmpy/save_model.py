from conin.util import try_import

with try_import() as pgmpy_readwrite_available:
    from pgmpy.readwrite.BIF import BIFWriter
    from pgmpy.readwrite.UAI import UAIWriter


def save_model(pgm, name, quiet=True):

    if name.endswith(".bif"):
        if not pgmpy_readwrite_available:
            raise RuntimeError("Cannot save a pgmpy model to BIF format without importing pgmpy")
        writer = BIFWriter(pgm)
        writer.write(name)

    elif name.endswith(".uai"):
        if not pgmpy_readwrite_available:
            raise RuntimeError("Cannot save a pgmpy model to UAI format without importing pgmpy")
        writer = UAIWriter(pgm)
        writer.write(name)
