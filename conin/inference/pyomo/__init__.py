"""MAP inference with Pyomo integer programs.

The models are built by the ``"pyomo"`` and ``"pyomo_network_flow"``
representations (:mod:`conin.representations.pyomo`).
"""

from .solve import solve_pyomo_map_query_model
from .hmm_network_flow import lp_inference, ip_inference
