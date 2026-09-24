Hidden Markov Model Learning
============================

``conin.hidden_markov_model.learning`` contains helpers for estimating HMM
parameters from simulated or labeled sequences. The supervised-learning path is
the most direct and best-tested interface. Monte Carlo EM support exists, but it
is more experimental and expects an application object with domain-specific
sampling behavior.

Simulation Format
-----------------

Learning helpers use a list of simulation records. Each record has aligned
``hidden`` and ``observed`` sequences, plus an ``index`` field. The
``convert_to_simulations`` helper creates this format from parallel lists.

.. code-block:: python

   from conin.hidden_markov_model import learning

   hidden = [["h0", "h0", "h1"], ["h1", "h1", "h1", "h0"]]
   observed = [["o0", "o0", "o0"], ["o1", "o1", "o1", "o1"]]

   simulations = learning.convert_to_simulations(
       hidden_list=hidden,
       observed_list=observed,
   )

   assert simulations[0].hidden == ["h0", "h0", "h1"]
   assert simulations[0].observed == ["o0", "o0", "o0"]
   assert simulations[0].index == 0

Supervised Learning
-------------------

``supervised_learning`` estimates start, transition, and emission probabilities
from simulations where both hidden and observed states are known.

.. code-block:: python

   from conin.hidden_markov_model import learning

   hidden_states = ["h0", "h1"]
   observable_states = ["o0", "o1"]
   hidden = [["h0", "h0", "h1"], ["h1", "h1", "h1", "h0"]]
   observed = [["o0", "o0", "o0"], ["o1", "o1", "o1", "o1"]]

   simulations = learning.convert_to_simulations(
       hidden_list=hidden,
       observed_list=observed,
   )
   hmm = learning.supervised_learning(
       simulations=simulations,
       hidden_states=hidden_states,
       observable_states=observable_states,
       start_tolerance=0,
       transition_tolerance=0,
       emission_tolerance=0,
   )

   start_probs = hmm.get_start_probs()
   transition_probs = hmm.get_transition_probs()
   emission_probs = hmm.get_emission_probs()

   assert start_probs["h0"] == 0.5
   assert transition_probs["h0", "h1"] == 0.5
   assert emission_probs["h1", "o1"] == 0.75

By default, the learner uses small tolerances for start, transition, and
emission estimates. These tolerances keep unobserved events from receiving exact
zero probability. You can set tolerances to ``0`` when you want raw empirical
frequencies.

Unobserved States and Priors
----------------------------

The ``hidden_states`` and ``observable_states`` arguments define the full state
spaces, not just states present in the training data. If a hidden state has no
observed outgoing transitions or emissions, ``supervised_learning`` uses uniform
probabilities unless a prior dictionary is supplied.

.. code-block:: python

   from conin.hidden_markov_model import learning

   simulations = learning.convert_to_simulations(
       hidden_list=[["h0", "h1"]],
       observed_list=[["o0", "o1"]],
   )
   hmm = learning.supervised_learning(
       simulations=simulations,
       hidden_states=["h0", "h1", "h2"],
       observable_states=["o0", "o1", "o2"],
       start_tolerance=0,
       transition_tolerance=0,
       emission_tolerance=0,
       transition_prior={("h2", "h2"): 1.0},
       emission_prior={("h2", "o2"): 1.0},
   )

   assert hmm.get_transition_probs()["h2", "h2"] == 1.0
   assert hmm.get_emission_probs()["h2", "o2"] == 1.0

Unknown Hidden States
---------------------

``add_unknowns`` replaces hidden-state values that occur at most ``num`` times
with a token. This is useful for collapsing rare labels before fitting a model.

.. code-block:: python

   from conin.hidden_markov_model import learning

   hidden = [[0, 0, 1], [2], [0, 1, 3], [4]]
   collapsed = learning.add_unknowns(hidden, num=1)

   assert collapsed == [
       [0, 0, 1],
       ["__UNKNOWN__"],
       [0, 1, "__UNKNOWN__"],
       ["__UNKNOWN__"],
   ]

``add_unknowns`` mutates the nested lists it receives. Pass copies if you need
to preserve the original sequences.

Monte Carlo EM
--------------

``mcem`` is an experimental Monte Carlo EM helper. It is intended for workflows
where an application object can sample feasible hidden-state sequences for a
fixed observed sequence, then refit the HMM with ``supervised_learning``.

At a high level, the application object is expected to provide:

- ``generate_hidden(observed)`` for sampling a feasible hidden sequence.
- ``hmm`` with a ``log_probability(observed, hidden)`` method.

Because this path is less stable and more application-specific, prefer
``supervised_learning`` for labeled data and treat ``mcem`` as an advanced API.

Reference
---------

.. autofunction:: conin.hidden_markov_model.learning.convert_to_simulations

.. autofunction:: conin.hidden_markov_model.learning.supervised_learning

.. autofunction:: conin.hidden_markov_model.learning.add_unknowns

.. autofunction:: conin.hidden_markov_model.learning.mcem
