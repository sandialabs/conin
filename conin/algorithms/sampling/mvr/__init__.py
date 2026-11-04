"""MVR sampling on constrained HMMs (torch): forward-filtering backward-sampling
and stopped sampling.

The algorithms need torch, so the names below are imported on first access:
importing this package does not import torch.
"""

import importlib

_EXPORTS = {
    "ffbs_torch_mvr_chmm": ".ffbs",
    "stopped_sampling_torch_mvr_chmm": ".stopped_sampling",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
