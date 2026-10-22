Scoring, I/O and Conversion
===========================

Scoring
-------

.. autofunction:: conin.scoring.log_probability

.. autofunction:: conin.scoring.log_potential

.. autofunction:: conin.scoring.log_probability_query

File I/O
--------

.. autofunction:: conin.io.load_model

.. autofunction:: conin.io.save_model

.. autofunction:: conin.io.uai.load_conin_model_from_uai

.. autofunction:: conin.io.uai.save_model_uai

.. autofunction:: conin.io.cfn.load_conin_model_from_cfn

The ``convert_pgm`` console script (``conin.io.convert_pgm``) converts model
files between formats; run ``convert_pgm --help`` for its options.

pgmpy interoperability
----------------------

``conin.pgmpy`` converts models between pgmpy and CONIN in user code (the
file I/O functions above use it for pgmpy models). It needs pgmpy at call
time, not at import time.

.. autofunction:: conin.pgmpy.convert_pgmpy_to_conin

.. autofunction:: conin.pgmpy.convert_conin_to_pgmpy_bn

.. autofunction:: conin.pgmpy.convert_conin_to_pgmpy_mn

.. autofunction:: conin.pgmpy.MapCPD

.. autofunction:: conin.pgmpy.is_polytree

For CONIN Bayesian networks, use :func:`conin.graphical_models.is_polytree`.

Exceptions
----------

.. autoexception:: conin.exceptions.InvalidInputError

.. autoexception:: conin.exceptions.InsufficientSolutionsError
