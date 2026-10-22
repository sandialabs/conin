"""MAP inference with variable elimination (pgmpy).

The models are built by the ``"pgmpy"`` representations
(:mod:`conin.representations.pgmpy`).
"""

from .solve import solve_variable_elimination_map_query_model
from .mn import inference_variable_elimination_map_query_MN
from .bn import inference_variable_elimination_map_query_BN
from .dbn import inference_variable_elimination_map_query_DDBN
from .hmm import inference_variable_elimination_map_query_HMM
