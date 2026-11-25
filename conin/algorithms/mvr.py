"""The MVR algorithms on constrained HMMs (torch), in one namespace.

They live with the other algorithms of their kind (inference, other queries,
sampling and learning); this module gathers them, so that
``from conin.algorithms.mvr import mvr_satisfaction_probability`` works
without knowing where each one is.

The algorithms need torch, so the names below are imported on first access:
importing this module does not import torch.
"""

import importlib

_EXPORTS = {
    "mvr_viterbi": "conin.algorithms.inference.mvr",
    "mvr_marginal_map": "conin.algorithms.inference.mvr",
    "mvr_satisfaction_probability": "conin.algorithms.other_queries.mvr",
    "mvr_satisfaction_time": "conin.algorithms.other_queries.mvr",
    "mvr_sample": "conin.algorithms.sampling.mvr",
    "mvr_stopped_sample": "conin.algorithms.sampling.mvr",
    "mvr_forward_backward": "conin.algorithms.learning.mvr",
    "mvr_baum_welch": "conin.algorithms.learning.mvr",
    "mvr_generalized_em": "conin.algorithms.learning.mvr",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
