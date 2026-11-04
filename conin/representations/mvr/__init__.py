"""MVR representations.

Importing this package does not import torch; the torch kernels of the MVR
algorithms are in :mod:`conin.algorithms.util.mvr_kernels`.
"""

from .hmm import MVRHMMRepresentation, align_hidden_states
