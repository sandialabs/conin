"""Scoring: log-probabilities of assignments.

* :func:`log_probability` and :func:`log_potential` evaluate a total
  assignment (no inference).
* :func:`log_probability_query` computes a marginal or conditional
  log-probability, under the constraints of a constrained model (inference).
"""

from .log_potential import log_potential, log_probability
from .log_probability_query import log_probability_query
