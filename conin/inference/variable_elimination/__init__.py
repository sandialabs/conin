"""MAP inference with variable elimination (pgmpy).

The models are built by the ``"factor"`` representations
(:mod:`conin.representations.factor`).  ``custom_VE`` (``logprob_VE``, used by
:func:`conin.common.pgmpy.log_potential.log_prob_v2`) needs pgmpy at import
time and is not imported here.
"""

from .solve import solve_variable_elimination_map_query_model
from .mn import inference_variable_elimination_map_query_MN
from .bn import inference_variable_elimination_map_query_BN
from .dbn import inference_variable_elimination_map_query_DDBN
from .hmm import inference_variable_elimination_map_query_HMM
