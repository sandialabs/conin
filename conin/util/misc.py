import importlib
import copy
from dataclasses import dataclass
import numpy as np


@dataclass(order=True, frozen=True)
class State:
    value: tuple


def as_rng(rng=None):
    """Return a :class:`numpy.random.Generator` for the ``rng`` argument of a sampler.

    ``rng`` is ``None`` (a new generator seeded from the operating system),
    an int or a :class:`numpy.random.SeedSequence` (a new generator with this
    seed), or a :class:`numpy.random.Generator` (returned as is, so that
    successive calls continue its stream), as for
    :func:`numpy.random.default_rng`.
    """
    return np.random.default_rng(rng)


class Util:
    @staticmethod
    def sample_from_vec(probabilities, rng):
        """Draw an index with the given probabilities, using the generator ``rng``."""
        val = rng.random()
        temp = 0
        for i in range(len(probabilities)):
            temp += probabilities[i]
            if temp >= val:
                return i
        # The probabilities sum to slightly less than 1 (rounding).
        return len(probabilities) - 1

    @staticmethod
    def normalize_vector(vec):
        total = sum(vec)
        if total == 0:
            return [1 / len(vec) for _ in vec]
        return [float(x) / total for x in vec]  # Convert to float explicitly

    @staticmethod
    def normalize_dictionary(dict):
        sum = 0
        for value in dict.values():
            sum += value

        normalized_dict = copy.copy(dict)
        if sum != 0:
            for key in dict:
                normalized_dict[key] /= sum
        else:
            for key in dict:
                normalized_dict[key] = 1 / len(dict)
        return normalized_dict

    @staticmethod
    def normalize_2d_dictionary(dict):
        sums = {}
        len = {}
        for (key1, key2), val in dict.items():
            if key1 not in sums:
                sums[key1] = 0
                len[key1] = 0
            sums[key1] += val
            len[key1] += 1

        normalized_dict = {}
        for (key1, key2), val in dict.items():
            if sums[key1] != 0:
                normalized_val = val / sums[key1]
            else:
                normalized_val = 1 / len[key1]
            normalized_dict[(key1, key2)] = normalized_val
        return normalized_dict


class try_import:

    def __init__(self):
        self.success = False

    def __bool__(self):
        return self.success

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is ImportError or exc_type is ModuleNotFoundError:
            # Catch import errors
            return True
        elif exc_type is not None:
            # Propogate other exceptions
            return False

        # No import errors, so return
        self.success = True
        return True

    def __str__(self):
        return str(self.success)


class lazy_import:
    """An optional module that is imported the first time it is used.

    ``bool(module)`` imports it and says whether the import worked (an
    ``ImportError`` means it did not); an attribute lookup imports it and
    raises ``ImportError`` if it is not available.  ``import conin`` uses
    this for pgmpy, which takes several seconds to import (it imports torch).

    >>> from conin.util import lazy_import
    >>> json = lazy_import("json")
    >>> bool(json), json.dumps([1])
    (True, '[1]')
    >>> bool(lazy_import("no_such_module"))
    False
    """

    def __init__(self, name):
        self._name = name
        self._module = None
        self._error = None

    def _load(self):
        if self._module is None and self._error is None:
            try:
                self._module = importlib.import_module(self._name)
            except ImportError as e:
                self._error = e
        return self._module

    def __bool__(self):
        return self._load() is not None

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        module = self._load()
        if module is None:
            raise ImportError(f"The {self._name} module is not available") from (
                self._error
            )
        return getattr(module, attr)

    def __repr__(self):
        return f"lazy_import({self._name!r})"


try:
    # Python >= 3.12
    from itertools import batched
except ImportError:
    #
    # See  https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    #
    # This implementation is fast, but it assumes the iterable has a len() method.
    #
    def batched(iterable, n=1):
        iterable_len = len(iterable)
        for ndx in range(0, iterable_len, n):
            yield iterable[ndx : min(ndx + n, iterable_len)]
