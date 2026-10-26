"""Time ranges of the representations of unrolled DBNs and HMMs.

The ``"pyomo"``, ``"toulbar2"`` and ``"pgmpy"`` representations of a DBN or an
HMM are built on the model unrolled over ``start..stop`` (``start`` and
``stop`` are structure parameters).  This module holds what they share:

* :class:`UnrolledDBNMixin` and :class:`UnrolledHMMMixin`: the ``start``,
  ``stop`` (and ``hidden_to_internal``) attributes, :attr:`time_steps` and the
  fields these add to the constraint ``data`` (:meth:`structure_fields`);
* :func:`dbn_time_range` and :func:`hmm_time_range`: the default ``stop`` of
  a DBN and the check that an HMM has one.
"""

from conin.exceptions import InvalidInputError
from conin.util.constraint_data import dbn_time_fields, hmm_time_fields

DEFAULT_DBN_STOP = 1
"""int: The ``stop`` of a DBN representation built without one."""


def dbn_time_range(start, stop):
    """Return ``(start, stop)`` of a DBN representation (``stop=None`` means 1)."""
    return start, DEFAULT_DBN_STOP if stop is None else stop


def hmm_time_range(start, stop):
    """Return ``(start, stop)`` of an HMM representation.

    Raises
    ------
    InvalidInputError
        If ``stop`` is ``None``: the last time step of an HMM is that of the
        last observation, so it has no default here.
    """
    if stop is None:
        raise InvalidInputError(
            "The HMM representation needs a stop time (the last observation)"
        )
    return start, stop


class UnrolledDBNMixin:
    """A representation of a DBN unrolled over ``start..stop``.

    Put it before the backend base class (``class
    PyomoDBNRepresentation(UnrolledDBNMixin, PyomoFactorRepresentation)``):
    it sets ``start`` and ``stop`` before the base ``__init__`` runs, so the
    base can already read :meth:`structure_fields`.

    Attributes
    ----------
    start, stop : int
        The first and last unrolled time step.
    """

    def __init__(self, *, start, stop, **kwds):
        self.start = start
        self.stop = stop
        super().__init__(**kwds)

    @property
    def time_steps(self):
        """list of int: ``list(range(start, stop + 1))``."""
        return list(range(self.start, self.stop + 1))

    def structure_fields(self):
        """``Munch(start, stop, T)`` with ``T`` = :attr:`time_steps`."""
        return dbn_time_fields(self.start, self.stop)


class UnrolledHMMMixin(UnrolledDBNMixin):
    """A representation of an HMM unrolled (as a DBN) over ``start..stop``.

    Attributes
    ----------
    start, stop : int
        The first and last unrolled time step.
    hidden_to_internal : dict
        The HMM's map from hidden-state labels to internal indices.
    """

    def __init__(self, *, hidden_to_internal, **kwds):
        self.hidden_to_internal = hidden_to_internal
        super().__init__(**kwds)

    def structure_fields(self):
        """``Munch(start, stop, hmm=Munch(T, hidden_to_internal))``."""
        return hmm_time_fields(self.start, self.stop, self.hidden_to_internal)
