"""The exceptions of CONIN, and which errors CONIN raises when.

* :class:`InvalidInputError` (a ``ValueError``): an argument has an invalid
  value, e.g. a model or constraint that is not valid, an unknown method,
  state or option value, or evidence that does not fit the model.
* ``TypeError``: an argument that should be a model (or constrained model)
  is of a class the function does not take, or a function is called with
  an option it does not take or the wrong number of arguments.
* ``KeyError``: a model is asked about a node it does not have
  (``states_of``, ``card``).
* ``ImportError``: an optional dependency (pytoulbar2, torch, smoek,
  or_topas, ...) is not installed.
* ``NotImplementedError`` (a ``RuntimeError``): the input is valid, but
  CONIN does not support it, e.g. a partial MAP query with a method other
  than variable elimination, or feasibility checks for Pyomo constraints.
* ``RuntimeError``: a computation failed, e.g. the MIP solver failed, no MIP
  solver is installed, or a sampler gave up
  (:class:`InsufficientSolutionsError` for or_topas).

:mod:`conin.io` raises :class:`ModelFileError` for a file whose content is
malformed or invalid, and ``FileNotFoundError`` for a missing file.
"""


class InvalidInputError(ValueError):
    """An argument has an invalid value.

    It is a ``ValueError``, so callers that catch ``ValueError`` catch it
    too.
    """


class InfeasibleQueryError(InvalidInputError):
    """No assignment is feasible: the evidence or the constraints exclude all.

    Raised while a backend builds or searches a model that can then have no
    solution (e.g. observations of probability zero).  ``map_query`` returns
    its no-solution result instead (``solution=None``; see
    :mod:`conin.algorithms.inference.results`).  It is an ``InvalidInputError``, so
    callers that caught that error before still do.
    """


class ModelFileError(InvalidInputError):
    """A model file (or string) cannot be read: its content is malformed or invalid.

    Raised by :mod:`conin.io` for truncated or malformed UAI, CFN and JSON
    data, and for data that does not describe a valid model.  It is an
    ``InvalidInputError``, and so also a ``ValueError`` (what the readers
    raised before), so callers that caught either still do.  A missing file
    raises ``FileNotFoundError``.
    """


class InsufficientSolutionsError(RuntimeError):
    """A solver found fewer solutions than asked for (or_topas found none)."""
