class InvalidInputError(Exception):
    pass


class InfeasibleQueryError(InvalidInputError):
    """No assignment is feasible: the evidence or the constraints exclude all.

    Raised while a backend builds or searches a model that can then have no
    solution (e.g. observations of probability zero).  ``map_query`` returns
    its no-solution result instead (``solution=None``; see
    :mod:`conin.algorithms.inference.results`).  It is an ``InvalidInputError``, so
    callers that caught that error before still do.
    """


class ModelFileError(InvalidInputError, ValueError):
    """A model file (or string) cannot be read: its content is malformed or invalid.

    Raised by :mod:`conin.io` for truncated or malformed UAI, CFN and JSON
    data, and for data that does not describe a valid model.  It is an
    ``InvalidInputError``, and also a ``ValueError`` (what the readers
    raised before), so callers that caught either still do.  A missing file
    raises ``FileNotFoundError``.
    """


class InsufficientSolutionsError(Exception):
    pass
