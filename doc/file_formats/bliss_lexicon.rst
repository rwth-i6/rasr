Bliss Lexicon
=============

The lexicon defines the phoneme set and the pronunciation dictionary. The Bliss lexicon format used is an XML format.

In most applications it is necessary to use separate lexica for (acoustic model) training and recognition.

Terminology
-----------

In the area of speech recognition, the term "word" is used in different contexts: acoustic words, language model words, evaluation words, phrases and multi-words, silence words, and noise words. We introduce separate terms to avoid this ambiguousness. A definition of the used terminology is given in the following:

orthographic form
    The written form of a word according to the conventions of the respective language.
pronunciation
    A pronunciation is a way to realize one (or more) words acoustically. A pronunciation is usually defined by a sequence of phonetic symbols (for example in SAMPA notation). However special purpose models, such as whole-word models (sometimes incorrectly called "function words"), also count as pronunciations.
language model
    The term language model is used in this document for any kind of (stochastic) grammar used in recognition, not only multi-grams. The terminal symbols of this grammar are called syntactic or language model tokens.
syntactic token / language model token
    a terminal symbol of the language model
evaluation token / evaluated word
    A word as counted for evaluation purposes.
acoustic phrase
    a pronounciation spanning more than one (orthographic) words, also known as "multi-word". Multi-words are a way to model pronunciation variation across word-boundaries.
language model phrase
    a syntactic token spanning more than one (orthographic) word.
phoneme
    abstract unit describing an underlying sound in speech.
allophone
    a realization of a phoneme occuring in a phonetic transcription, i.e. a phoneme in context, e.g. a triphone.
phonology
    A set of rules that describes the variation of sounds (phonemes) when they occur in speech.


Design Issues
-------------

**Lemmata**

The fundamental unit of a Bliss lexicon is called lemma. A lemma may have

    one or more orthographic forms,
    any number of pronunciations,
    a sequence of syntactic tokens and
    a sequence of evaluation tokens.

**Special Lemmata**

Special lemmata define particular artificial words. They are requested explicitly in the program code. Examples of special words are: silence, unknown and sentence boundary. Bliss requires you to specify all words in the lexicon explicitly. Special lemmata are tagged with the attribute special.

The presence of some lemmata in the lexicon is required. These are called required lemmata. All required lemmata are special lemmata, but not all special lemmata have to be required.

**Training, Recognition and Evaluation**

For the training of the acoustic models, a phonetic transcription of the audio training data is needed. However, in most cases only an orthographic transcription is available. Therefore, all pronunciation variations and non-speech events are considered.

The output of the recognition is a joint sequence of pronunciations and syntactic tokens. To evaluate the recognition performance, the sequence of syntactic tokens is compared to an orthographic transcription. Optionally, non-speech tokens can be mapped to an orthographic word (e.g. [SILENCE], [NOISE], ...). These non-speech tokens are usually not considered for the evaluation.

If the training and the recognition vocabularies are not identical, two separate lexicons are needed. A large background lexicon will contain all words occurring in the training and testing corpora. This can be used immediately for training. The recognition lexicon can be extracted from the background lexicon.

The most common evaluation metric for speech recognizers is the word error rate (WER). Some lemmata need to be mapped to special tokens to be evaluated correctly. These tokens are called evaluation lemma and can be defined using the ``<eval>`` tag. An example is "silence" which is annotated in the training corpus while it is not used in the evaluation.

Special Words
-------------

In the following we describe important frequently used special lemmata.

**Silence**

The typical properties of the "silence word" are

Silence is allowed between all word in forced alignment (see :doc:`training/index`). This is achieved by setting an empty orthogaphy: ``<orth/>``
Silence is not seen by the language model. This is achieved by setting an empty syntactic token sequence: ``<synt/>``
Silence is not evaluated. This is achieved by setting an empty evaluation token sequence: ``<eval/>``

**Example**

.. code-block :: xml

    <lemma special="silence">
      <orth>[pause]</orth>
      <orth/>
      <orth>[hesitates]</orth>
      <phon>si</phon>
      <synt/>
      <eval/>
    </lemma>

Line 2 specifies that the recognition output will read [pause] when a period of silence is detected. In training, it will also force a silence phoneme to be hypothesised, if [pause] occurs in the transcription.
Line 3 states that silence does not necessarily show in the written sentence. Thus during training, silence will be hypothesized between all words.
Line 4: We assume that in the corpus the transcribers put [hesitates] when the speaker makes a pause within a sentence. This line forces a silence phoneme in training, and makes [hesitates] equivalent to [pause] for evaluation.
Line 5 states the acoustic realization of silence.
Line 6 states that an occurrence of silence will be ignored for language modeling purposes.
Line 7 causes all occurrences of silence to be removed before computing the word error rate.

Note:

Line 2 also implicitly specifies the "preferred orthografic form" of the Silence lemma to be [pause] and not a blank. In some algorithms, this orthographic form, i.e. the ordering of the ``<orth>`` tags, is important such as for :doc:`training/mpe`.

**Unknown Lemma**

Corpora often contain words that are not in the lexicon, so-called out-of-vocabulary (OOV) words. When the Bliss parser encounters a character sequence that matches none of the orthographic forms in the lexicon, it will substitute a special lemma called "unknown".

For acoustic model training, it is necessary to create a model acceptor for any sentence containing an OOV word. This is realized via a pronunciation for "unknown". This pronunciation can be a garbage phoneme or even empty. An OOV word in training should be avoided for optimal results.

During recognition the unknown lemma is provided to calculate the number of errors correctly. In rare cases it is necessary to hypothesize "unknown" during recognition. Then a pronunciation for "unknown" has to be provided.

**Example**

.. code-block :: xml

    <lemma special="unknown">
      <orth>[UNKNOWN]</orth>
      <phon>mul</phon>
    </lemma>

The Bliss parser substitutes this lemma for all unknown words. It must be marked special for the parser to find it. Note: Some language models require adding ``<synt>&lt;UNK&gt;</synt>``

**Sentence Boundary**

N-gram language models usually use a special sentence end token to model the length of sentences. The commonly used sentence boundary token is declared like this:

.. code-block :: xml

    <lemma special="sentence-boundary">
      <synt>&lt;/s&gt;</synt> <!-- escaped "</s>" -->
    </lemma>

An approach to punctuation generation might look like this:

.. code-block :: xml

    <lemma special="sentence-boundary">
      <orth>.</orth>
      <phon>si</phon>
      <synt>&lt;/s&gt;</synt>
    </lemma>


Alternatively, sentence start and end can be defined explicitly using the special lemmas sentence-begin and sentence-end. For example:

.. code-block :: xml

    <lemma special="sentence-begin">
        <orth>[SENTENCE-BEGIN]</orth>
        <synt>
          <tok>&lt;s&gt;</tok>
        </synt>
        <eval/>
    </lemma>
    <lemma special="sentence-end">
        <orth>[SENTENCE-END]</orth>
        <synt>
          <tok>&lt;/s&gt;</tok>
        </synt>
        <eval/>
    </lemma>

Examples
--------

**Alternative Spelling**

.. code-block :: xml

    <lemma>
      <orth>Delphin</orth>
      <orth>Delfin</orth>
      <phon>d E l f i: n</phon>
    </lemma>

**Word with Blank**

.. code-block :: xml

    <lemma>
      <orth>New York</orth>
      <phon>n u: j O: k</phon>
      <synt><tok>class:city<tok></synt>
      <eval><tok>new</tok><tok>York</tok></eval>
    </lemma>

**Pronunciation Variants**

.. code-block :: xml

    <lemma>
      <orth>missile</orth>
      <phon>m I s aI l</phon>
      <phon>m I s l,</phon>
    </lemma>

**Pronunciation Scores (Probabilities)**

.. code-block :: xml

    <lemma>
      <orth>missile</orth>
      <phon score="0.223">m I s aI l</phon>
      <phon score="1.609">m I s l,</phon>
    </lemma>

    <!-- or -->

    <lemma>
      <orth>missile</orth>
      <phon weight="0.2">m I s aI l</phon>
      <phon weight="0.8">m I s l,</phon>
    </lemma>

The attribute score (>=0) refers to negative log of the pronunciation variant probability p(v|w), while the attribute weight refers to the probability itself (between 0 and 1). The default - if no attributes are provided - is weight=1.

The default value of *.lexicon.normalize-pronunciation is true, which normalizes the weights so that \sum_v p(v|w) = 1. Setting it to false does not enforce any normalization, keeping the values as specified.

**Acoustic Phrase**

.. code-block :: xml

    <lemma>
      <orth>haben wir</orth>
      <phon>h a m 6</phon>
      <synt>
        <tok>haben</tok>
        <tok>wir</tok>
      </synt>
      <eval>
        <tok>haben</tok>
        <tok>wir</tok>
      </eval>
    </lemma>

**Homonyme**

This is currently not possbile!

.. code-block :: xml

    <lemma>
      <!-- don't try this at home ! -->
      <orth>Altdorf</orth>
      <phon>a l t d O 6 f</phon>
      <synt><tok>class:town</tok></synt>
      <synt><tok>class:surname</tok></synt>
    </lemma>

Instead you have to use:

.. code-block :: xml

    <lemma>
      <orth>Altdorf</orth>
      <phon>a l t d O 6 f</phon>
      <synt><tok>class:town</tok></synt>
    </lemma>
    <lemma>
      <orth>Altdorf</orth>
      <phon>a l t d O 6 f</phon>
      <synt><tok>class:surname</tok></synt>
    </lemma>

**Noise**

.. code-block :: xml

    <lemma>
      <orth>[breathe]</orth>
      <orth/>
      <phon>GLAm</phon>
      <synt/>
      <eval/>
    </lemma>


During training the "phoneme" GLAm is applied in places where [breathe] occurs in the transcription (line 1) and it is optional at all word boundaries (line 2). During recognition [breathe] is output and no language model score is taken into account (line 4). If the event is to be predicted by the language model, line 4 has to be omitted. The last line prevent breath noise from being evaluated.

If the line <eval/> was absent, the evaluation behaviour would be the following: If [breathe] is transcribed but not recongized, this will be counted as an insertion or substitution. If [breathe] is recognized but not transcribed, it will treated as if it is transcribed. This scheme may be desirable, but the number of "spoken words" is no longer constant.

File Format Specification
-------------------------

.. list-table:: Bliss Corpus File Format
    :widths: 10, 20, 20, 50
    :header-rows: 1

    * - Tag
      - Description
      - Context
      - Attributes
    * - ``<lexicon>``
      - root element
      - xml root element
      -
    * - ``<phoneme-inventory>``
      - List of all phonemes (and phoneme modifiers) used in the dictionary.
      - ``<lexicon>``
      -
    * - ``<phoneme>``
      - Define a phoneme
      - ``<phoneme-inventory>``
      -
    * - ``<symbol>``
      - Define a phonemic symbol
      - ``<phoneme>``
      -
    * - ``<variation>``
      - | Define phoneme variation. Either context or none. Use none for
        | context independent phonemes like silence and noise.
      - ``<phoneme>``
      -
    * - ``<lemma>``
      - Definition of an abstract lemma
      - ``<lexicon>``
      - | ``special`` declare this lemma as a special lemma. The attribute value is the special lemma identifier
        |             for which one can query using Bliss::Lexicon::specialLemma().
        |             The meaningful identifiers are (usually) hard-coded.
        | ``id`` explicitly specify the ID number of a lemma. The attribute value must be an integer
        |        (in decimal representation). Normally IDs are assigned internally, but if the lexicon is
        |        edited and data files with lemma references
        |        (e.g. recognition lattices) are reused, it could be necessary to specify IDs externally.
        |        It is not recommended to mix explicit IDs with implicit IDs.
    * - ``<orth>``
      - | specifies an orthographic form of a lemma. If more than one
        | orthographic form is given, the first will be used in recognition
        | output (so-called "preferred orthographic form").
      - ``<lemma>``
      -
    * - ``<phon>``
      - specifies a phonemic pronunciation of a lemma
      - ``<lemma>``
      - | ``weight`` pronunciation weight: probability of the pronunciation given the lemma
        | ``score`` pronunciation score: negative logarithm of the pronunciation weight
        | either weight or score can be defined
    * - ``<synt>``
      - specifies an syntactic token sequence for a lemma.
      - ``<lemma>``
      -
    * - ``<eval>``
      - specifies an evaluation token sequence for a lemma.
      - ``<lemma>``
      -
    * - ``<tok>``
      - one single token in a syntactic or evaluation token sequence.
      - ``<synt>``, ``<eval>``
      -

**Example**

.. code-block :: xml

    <?xml version="1.0" encoding="ascii"?>
    <lexicon>

      <phoneme-inventory>
        <phoneme>
          <symbol>AH</symbol>
        </phoneme>
        <phoneme>
          <symbol>EY</symbol>
        </phoneme>
        <!-- ... -->
        <phoneme>
          <symbol>si</symbol>
          <variation>none</variation>
        </phoneme>
      </phoneme-inventory>

    <!--
        special lemmas
    -->
      <lemma special="silence">
        <orth>[SILENCE]</orth>
        <orth/>
        <phon>si</phon>
        <synt/>
        <eval/>
      </lemma>
      <lemma special="sentence-begin">
        <orth>[SENTENCE-BEGIN]</orth>
        <synt>
          <tok>&lt;s&gt;</tok>
        </synt>
        <eval/>
      </lemma>
      <lemma special="sentence-end">
        <orth>[SENTENCE-END]</orth>
        <synt>
          <tok>&lt;/s&gt;</tok>
        </synt>
        <eval/>
      </lemma>
      <lemma special="unknown">
        <orth>[UNKNOWN]</orth>
        <synt>
          <tok><UNK></tok>
        </synt>
        <eval/>
      </lemma>

    <!--
        regular lemmas
    -->
      <lemma>
        <orth>A</orth>
        <phon>AH</phon>
        <phon>EY</phon>
      </lemma>
      <lemma>
        <orth>AND</orth>
        <phon>AE N D</phon>
        <phon>AH N D</phon>
      </lemma>
      <!-- ... -->
      <lemma>
        <orth>ZERO</orth>
        <phon>Z IH R OW</phon>
      </lemma>
    </lexicon>

