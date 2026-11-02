"""MVR learning on constrained HMMs (torch).

The algorithms need torch, so the names below are imported on first access:
importing this package does not import torch.
"""

import importlib

_EXPORTS = {
    "baum_welch_mvr_chmm": ".baum_welch",
    "forward_backward_mvr_chmm": ".baum_welch",
    "generalized_em_mvr_chmm": ".generalized_em",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
