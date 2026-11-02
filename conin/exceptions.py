class InvalidInputError(Exception):
    pass


class InfeasibleQueryError(InvalidInputError):
    """No assignment is feasible: the evidence or the constraints exclude all.

    Raised while a backend builds or searches a model that can then have no
    solution (e.g. observations of probability zero).  ``map_query`` returns
    its no-solution result instead (``solution=None``; see
    :mod:`conin.inference.results`).  It is an ``InvalidInputError``, so
    callers that caught that error before still do.
    """


class InsufficientSolutionsError(Exception):
    pass
