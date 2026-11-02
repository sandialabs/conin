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

The ``convert_pgm`` console script (``conin.io.convert_pgm``) converts model
files between formats; run ``convert_pgm --help`` for its options.

pgmpy interoperability
----------------------

``conin.io.pgmpy`` reads and writes BIF files (and pgmpy's UAI reader and
writer) for the file I/O functions above, and converts models between pgmpy
and CONIN. CONIN's algorithms take CONIN models only: convert a pgmpy model
with ``convert_pgmpy_to_conin`` first. It needs pgmpy at call time, not at
import time.

.. autofunction:: conin.io.pgmpy.convert_pgmpy_to_conin

.. autofunction:: conin.io.pgmpy.convert_conin_to_pgmpy_bn

.. autofunction:: conin.io.pgmpy.convert_conin_to_pgmpy_mn

Exceptions
----------

.. autoexception:: conin.exceptions.InvalidInputError

.. autoexception:: conin.exceptions.InfeasibleQueryError

.. autoexception:: conin.exceptions.InsufficientSolutionsError
