Sampling and Learning
=====================

Sampling
--------

The unconstrained samplers and rejection sampling use NumPy's global random
state (seed it with ``hmm.set_seed`` or ``numpy.random.seed``); the torch MVR
samplers take a ``generator`` argument.

.. autofunction:: conin.sampling.unconstrained.generate_hidden

.. autofunction:: conin.sampling.unconstrained.generate_hidden_until_state

.. autofunction:: conin.sampling.unconstrained.generate_observed_from_hidden

.. autofunction:: conin.sampling.unconstrained.generate_observed

.. autoclass:: conin.sampling.unconstrained.DBNSampler
   :members:
   :show-inheritance:

.. autofunction:: conin.sampling.oracle.rejection_sample_hidden

MVR sampling (torch):

.. autofunction:: conin.sampling.mvr.ffbs.ffbs_torch_mvr_chmm

.. autofunction:: conin.sampling.mvr.stopped_sampling.stopped_sampling_torch_mvr_chmm

Learning
--------

``supervised_learning``, ``mcem``, ``convert_to_simulations`` and
``add_unknowns`` (``conin.learning``) are documented in :doc:`hmm_learning`.
The MVR learning algorithms (torch):

.. autofunction:: conin.learning.mvr.baum_welch.forward_backward_mvr_chmm

.. autofunction:: conin.learning.mvr.baum_welch.baum_welch_mvr_chmm

.. autofunction:: conin.learning.mvr.generalized_em.generalized_em_mvr_chmm
