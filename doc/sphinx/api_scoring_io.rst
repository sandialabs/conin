Scoring, Feasibility, I/O and Conversion
========================================

Scoring
-------

.. autofunction:: conin.algorithms.scoring.log_probability

.. autofunction:: conin.algorithms.scoring.log_potential

.. autofunction:: conin.algorithms.scoring.log_probability_query

Feasibility
-----------

.. autofunction:: conin.algorithms.feasibility.is_feasible

.. autofunction:: conin.algorithms.feasibility.partial_is_feasible

File I/O
--------

.. autofunction:: conin.io.load_model

.. autofunction:: conin.io.save_model

.. autofunction:: conin.io.uai.load_uai

.. autofunction:: conin.io.uai.save_uai

.. autofunction:: conin.io.cfn.load_cfn

.. automodule:: conin.io.json_format

.. autofunction:: conin.io.json_format.save_json

.. autofunction:: conin.io.json_format.load_json

.. automodule:: conin.io.bif

.. autofunction:: conin.io.bif.load_bif

.. autofunction:: conin.io.bif.save_bif

The ``convert_pgm`` console script (``conin.io.convert_pgm``) converts model
files between formats; run ``convert_pgm --help`` for its options.

Exceptions
----------

.. automodule:: conin.exceptions
   :no-members:

.. autoexception:: conin.exceptions.InvalidInputError

.. autoexception:: conin.exceptions.InfeasibleQueryError

.. autoexception:: conin.exceptions.ModelFileError

.. autoexception:: conin.exceptions.InsufficientSolutionsError
