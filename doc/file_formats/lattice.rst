Lattices
========

Overview
--------

The RWTH ASR decoder can produce lattices (or word graphs).
They are stored as finite state transducer (FSTs) together with a file containing the word boundary times.

Lattices can be post-processed using the :ref:`FlfTool`.

Producing lattices
------------------

For producing lattices the following lines needs to be added to any recognition config-file.

.. code-block :: ini

    ...
    [*]
    create-lattice                  = true
    store-lattices                  = true
    lattice-archive.path            = <lattice-archive>
    lattice-archive.compress        = true
    lattice-archive.type            = fsa
    lattice-pruning                 = $(lm-pruning)
    lattice-pruning-limit           = infinity
    optimize-lattice                = true
    time-conditioned-lattice        = false
    ...

The lattices are stored in an :ref:`Archive`.
For each segment three files are stored:
* an FST containing the acoustic score (``*.binfsa.gz``), 
* an FST containing the sum of scaled pronunciation- and LM-score (``*-lm.binfsa.gz``), and
* a file containing the word boundary time information (``*.binwb.gz``).

The FSA format is compatible with the `RWTH FSA Toolkit <http://www-i6.informatik.rwth-aachen.de/web/Software/index.html>`_.
The FSA toolkit can be used to manipulate the lattices or to convert them to plain text (AT&T or RWTH-XML transducer format).

Alternatively, lattices can be written in HTK lattice format, by setting ``lattice-archive.type = htk``.

The ``optimize-lattice`` flag reduces the size of the lattice by collapsing subsequent silence arcs.

If the ``time-conditioned-lattice`` flag is set, then the language model history is not preserved. 
The resulting lattice is much smaller and contains more paths than the standard, word-conditioned(i.e. with language model history) lattice.
A composition of the time-conditioned lattice with the language model transducer yields a word-conditioned lattice again.

References
----------
* `RWTH FSA Toolkit http://www-i6.informatik.rwth-aachen.de/web/Software/index.html`_
* `S. Kanthak and H. Ney: FSA: An Efficient and Flexible C++ Toolkit for Finite State Automata Using On-Demand Computation". In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL 2004), Barcelona, Spain, pp. 510-517, July, 2004. <http://www-i6.informatik.rwth-aachen.de/PostScript/InterneArbeiten/kanthak_acl2004.pdf >`_.
* S. Ortmanns, H. Ney, X. Aubert. "A Word Graph Algorithm for Large Vocabulary Continuous Speech Recognition". Computer, Speech and Language, Vol. 11, No. 1, pp. 43-72, January 1997.

