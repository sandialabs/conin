"""MVR representations.

``kernels`` (the torch kernels of the MVR algorithms) is not imported here, so
importing this package does not import torch.
"""

from .hmm import MVRHMMRepresentation, align_hidden_states
