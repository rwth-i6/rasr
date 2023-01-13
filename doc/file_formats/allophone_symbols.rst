Allophone Symbol
================

Allophone
---------

Format of the string representation of an allophone::

    C{L+R}B

where
* ``C`` is the central phoneme
* ``L`` is the left phoneme context (or the boundary symbol ``#``)
* ``R`` is the right phoneme context (or the boundary symbol ``#``)
* ``B`` specifies allophones at word boundaries: ``@i`` for the first phoneme (initial), ``@f`` for the last phoneme (final)


**Examples**
* ``n{a+s}``
* ``s{n+#}``
* ``s{n+#}@f``

Note/Bug: If the phoneme or allophone string contains one of the following symbols:
* ``+`` = allophone-neighbor delimiter 
* ``-`` = legacy delimiter 
* ``{`` = opening allophone-context delimiter 
* ``}`` = closing allophone-context delimiter 
* ``#`` = boundary symbol 
* ``.`` = allophone-state delimiter
* todo: there might be more symbols here
then a dumped lookup table probably cannot be read in again. See also: :ref:`Create State Tying Lookup Table` and :ref:`Bliss Lexicon`. 

This is especially the case if the RWTH-ASR  system is used for OCR tasks, where e.g. punctuations are used in the orthography and resulting phonetic transcriptions.
* ``b-P0{a-P0+c-P0}`` can be dumped but not read
* ``+{+++}`` can be dumped but not read
* ``#{#+#}`` can be dumped and read but is probably not that what you wanted, i.e. a model of the written sharp symbol.

Allophone State
---------------

Format of the string respresentation of an allophone state::

    A.S

where
* ``A`` is the Allophone string respresentation
* ``S`` is the state index (starting from 0)

**Examples**:
* ``n{a+s}.0``
* ``n{a+s}.1``
* ``n{a+s}.2``


Note: avoid using the state-delimiter symbol as phoneme symbol.
* ``.{.+.}@f.13`` representing three dots in a row and modeled with a 14 state HMM can be dumped but probably not read
