SearchV2 Framework
===================

SearchV2 is RASR's newer decoding framework for neural, end-to-end style recognizers (CTC, RNN-T/transducer,
Attention encoder-decoder, ...). It replaces the acoustic feature scoring of the classic
:ref:`Decoder` (``Search::SearchAlgorithm`` + ``Mm::FeatureScorer``) with a self-contained
:ref:`label scorer <SearchV2 Label Scorers>` abstraction, so the search algorithm itself no longer needs to know
from which model topology the scores come from.

This page describes SearchV2 from a user's perspective: which search algorithms and label scorers are available, how to
configure them, and how to wire them into an Flf network or a ``librasr`` Python session. It does not require reading
any C++ code.

.. contents::
   :local:
   :depth: 2

Overview
--------

``SearchV2`` and the classic ``Search`` framework can both be present in the same RASR build. They are selected
per use case, not globally.

* **Classic search** (``Search::SearchAlgorithm``, mostly ``AdvancedTreeSearch``) drives the search with
  acoustic scores from a ``Mm::FeatureScorer`` (GMMs, hybrid NN/HMM). It is used by the :ref:`recognizer` Flf node
  and by ``speech-recognizer``.
* **SearchV2** (``Search::SearchAlgorithmV2``) drives the search by pulling scores directly from one or more
  :ref:`label scorers <SearchV2 Label Scorers>` (typically wrapping an ONNX model). It is used by the
  :ref:`recognizer-v2` Flf node and by the ``librasr`` Python bindings.

Use SearchV2 if your acoustic/language model is a neural end-to-end model (CTC, transducer, attention
encoder-decoder).
Use the classic search for GMM or hybrid NN/HMM systems, or when you need one of the more specialized classic
searches (WFST-based search, linear search, ...).

Both frameworks produce the same kind of output (a single-best traceback and/or an Flf word lattice), so
downstream lattice processing (rescoring, CTM writing, ...) works the same way regardless of which search
produced the lattice.

The SearchV2 framework is intentionally kept simple. Instead of one large monolithic search implementation with
many different configuration options, it contains smaller, more specialized search algorithms for different use-cases
and various label scorers for different models.

Core workflow
-------------

Every SearchV2 algorithm implements the ``Search::SearchAlgorithmV2`` interface and is driven the same way,
whether it is invoked by the :ref:`recognizer-v2` node or by Python:

#. Determine which parts of a :ref:`Model Combination <Common component configuration>` the algorithm needs
   (lexicon, acoustic model, language model, label scorer(s)) and construct them accordingly.
#. Signal the start of a new segment/utterance.
#. Feed feature vectors one at a time, or in batches, as they become available.
#. Optionally trigger explicit decoding steps and query intermediate ("streaming") results.
#. Signal the end of the segment, which finalizes the search for all fed features.
#. Retrieve the final result, either as a single-best traceback or as a word lattice.

Since features are pushed in and results are pulled out independently, the same algorithm implementation can
be used both for offline decoding of a full corpus and for online/streaming decoding, without any change to
configuration.

Search algorithms
-----------------

Four search algorithms are currently implemented on top of SearchV2. Which one to pick depends on the type of model
and vocabulary you use:

* ``lexiconfree-timesync-beam-search``: time-synchronous decoding, no pronunciation lexicon required. Typical
  use case: CTC / neural monotonic transducer models.
* ``lexiconfree-labelsync-beam-search``: label-synchronous decoding, no pronunciation lexicon required. Typical
  use case: attention encoder-decoder (AED) models and Speech LLMs.
* ``tree-timesync-beam-search``: time-synchronous decoding over a pronunciation lexicon search tree, with a
  word-level language model. The topology of the tree itself (HMM, CTC, RNA, AED, ...) is a separate choice,
  see :ref:`Search tree types` below.
* ``tree-labelsync-beam-search``: label-synchronous counterpart to ``tree-timesync-beam-search``, decoding over
  a pronunciation lexicon search tree built with the ``aed`` tree builder, with a word-level language model.

The "lexiconfree" searches treat every entry of the lexicon as a single output token and do not build a
pronunciation search tree or apply a word-level transition model.
The "tree" searches (also called lexicon-constrained) build a prefix tree from the pronunciation lexicon
and additionally scores a language model at word ends.

Lexicon requirements
^^^^^^^^^^^^^^^^^^^^^

Even though the "lexiconfree" searches do not build a lexical prefix tree, a lexicon is still required.
Each lemma is treated as one output label and the lemma's index in the lexicon must match the corresponding
output index of the label scorer (e.g. the softmax index of a CTC/transducer/AED model). ``blank-label-index``,
``silence-label-index`` and ``sentence-end-label-index`` (see below) are likewise resolved as the ``id()`` of the
lemma marked ``special="blank"``/``"silence"``/``"sentence-end"``/``"sentence-boundary"`` respectively.

There are two ways to provide such a lexicon:

* A full Bliss XML lexicon (``file = /path/to/lexicon.xml.gz``, the default ``xml`` format). Here it is the
  responsibility of whoever writes the lexicon to declare the phoneme inventory and the lemma list in the same
  order, so that the Nth lemma corresponds to the Nth phoneme/output label -- this correspondence is *not*
  verified at runtime, so a mismatch will silently produce wrong labels rather than an error. For example, a
  minimal 3-label lexicon (labels ``A``, ``B``, ``C``, at output indices 0, 1, 2 respectively) looks like this:

  .. code-block:: xml

      <?xml version="1.0" ?>
      <lexicon>
        <phoneme-inventory>
          <phoneme>
            <symbol>A</symbol>
            <variation>none</variation>
          </phoneme>
          <phoneme>
            <symbol>B</symbol>
            <variation>none</variation>
          </phoneme>
          <phoneme>
            <symbol>C</symbol>
            <variation>none</variation>
          </phoneme>
        </phoneme-inventory>
        <lemma>
          <orth>A</orth>
          <phon>A</phon>
        </lemma>
        <lemma>
          <orth>B</orth>
          <phon>B</phon>
        </lemma>
        <lemma>
          <orth>C</orth>
          <phon>C</phon>
        </lemma>
      </lexicon>

* A plain vocabulary text file, one label per line, selected via a format qualifier on the lexicon ``file``
  parameter:

  .. code-block:: ini

      [*.lexicon]
      file = vocab-text:/path/to/labels.txt

  (``vocab-txt:`` is accepted as an equivalent qualifier.) This format was built specifically for lexicon-free
  searches: the phoneme inventory and the lemma list are both constructed from the same line-ordered list, so the
  Nth line always becomes both the Nth phoneme and the Nth lemma. The label-index/lemma-index correspondence is
  therefore guaranteed automatically, with no risk of the two getting out of sync. ``labels.txt`` for the same
  3-label vocabulary as above would simply be:

  .. code-block:: text

      A
      B
      C

  The disadvantage of this format is that it has no way to mark special lemmata. Every line just becomes a plain label.
  So ``blank-label-index``, ``silence-label-index`` and ``sentence-end-label-index`` cannot be inferred from a
  ``vocab-text``/``vocab-txt`` lexicon and must be passed explicitly as parameters to the search algorithm instead.

The "tree" searches, in contrast, always need a full pronunciation lexicon (the ``xml`` format). They build the
prefix search tree directly from its pronunciations, so the ``vocab-text``/``vocab-txt`` format does not apply
there.

Selecting an algorithm
^^^^^^^^^^^^^^^^^^^^^^^

The search algorithm is selected with the ``type`` parameter under the ``search-algorithm`` configuration selector:

.. code-block:: ini

    [*.search-algorithm]
    type = lexiconfree-timesync-beam-search
    ; other options: lexiconfree-labelsync-beam-search, tree-timesync-beam-search, tree-labelsync-beam-search

If unset, ``type`` defaults to ``lexiconfree-timesync-beam-search``.

All parameters shown in the sections below are relative to that same ``search-algorithm`` selector, e.g. the
full path of ``max-beam-size`` is ``*.search-algorithm.max-beam-size``.

lexiconfree-timesync-beam-search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple time-synchronous beam search without a pronunciation lexicon, word-level language model or transition model.
Intended for open-vocabulary decoding with CTC and neural transducer style or similar models. Supports
an optional blank label (for CTC/transducer) and optional silence and sentence-end labels.

* ``max-beam-size`` (int list): maximum number of hypotheses kept in the beam. Pruning is applied after every
  label scorer in the pipeline (see :ref:`SearchV2 Label Scorers`), so one value is expected per configured
  label scorer. Default: unset, i.e. no beam-size pruning is applied.
* ``score-threshold`` (float list): prune hypotheses whose score is worse than the current best by more than
  this amount. Also applied once per label scorer. Default: unset, i.e. no score-based pruning is applied.
* ``num-histogram-bins`` (int): number of bins used for histogram pruning (minor effect on results/speed). Default ``100``.
* ``blank-label-index`` (int): lexicon index of the blank label. Inferred automatically from a lemma with
  ``special="blank"`` if present; if neither is set, blank handling is disabled. Default: disabled.
* ``silence-label-index`` (int): lexicon index of the silence label, inferred from ``special="silence"`` if unset.
  Default: disabled.
* ``sentence-end-label-index`` (int): lexicon index of the sentence-end label, inferred from
  ``special="sentence-end"``/``special="sentence-boundary"`` if unset. Default: disabled.
* ``collapse-repeated-labels`` (bool): collapse repeated emission of the same label into a single output
  (typical for CTC-style label loops). Default ``false``.
* ``recombination-mode`` (``on``/``off``): recombine hypotheses that share the same label scorer state
  (keeping only the best), like word-history recombination in the classic search. Default ``on``.
* ``cache-cleanup-interval`` (int): interval (in search steps) after which cached buffered inputs that are no
  longer needed get freed. Default ``10``.
* ``maximum-stable-delay`` (int): if set, prune away hypotheses that disagree with the current best hypothesis
  further back than this many frames. This makes the traceback "stabilize" after at most this many frames,
  which is useful for low-latency streaming output. Default: disabled (unbounded).
* ``maximum-stable-delay-pruning-interval`` (int): how often (in search steps) the above pruning is applied. Default ``10``.
* ``log-stepwise-statistics`` (bool): log beam statistics at every search step, useful for tuning and debugging. Default ``false``.

Example config:

.. code-block:: ini

    [*.search-algorithm]
    type                     = lexiconfree-timesync-beam-search
    max-beam-size            = 5000
    score-threshold          = 14.0
    blank-label-index        = 0
    collapse-repeated-labels = true
    recombination-mode       = on

lexiconfree-labelsync-beam-search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A label-synchronous beam search, again without a pronunciation lexicon, word-level language model or
transition model. Hypotheses are terminated by an explicit sentence-end symbol rather than by running out of
input frames. Its main purpose is open-vocabulary search with attention encoder-decoder (AED) or similar
models.

* ``max-beam-size`` (int list), ``score-threshold`` (float list), ``num-histogram-bins`` (int),
  ``recombination-mode``, ``log-stepwise-statistics``, ``cache-cleanup-interval``: same meaning as for
  ``lexiconfree-timesync-beam-search`` above.
* ``sentence-end-label-index`` (int): lexicon index of the sentence-end label that terminates a hypothesis.
  Inferred from ``special="sentence-end"``/``special="sentence-boundary"`` if unset.
* ``length-norm-scale`` (float): exponent for length normalization; scaled scores are computed as
  ``score / length^length-norm-scale``. Default ``0.0`` (no normalization). If set, ``score-threshold`` is
  applied to the un-normalized score before normalization.
* ``max-labels-per-timestep`` (float): maximum number of emitted labels per input timestep (as consumed via
  ``putFeature``/``putFeatures``), used to bound hypothesis length relative to the input length. Default ``1.0``.

Example config:

.. code-block:: ini

    [*.search-algorithm]
    type                    = lexiconfree-labelsync-beam-search
    max-beam-size           = 50
    score-threshold         = 5.0
    length-norm-scale       = 1.0
    max-labels-per-timestep = 1.2

tree-timesync-beam-search
^^^^^^^^^^^^^^^^^^^^^^^^^^

A time-synchronous beam search that decodes over a prefix tree built from the pronunciation lexicon. Similar
in structure to the classic tree search, but scored via label scorer(s) instead of a feature scorer. A
language model score is added at word ends, to disable it, don't set a file or type for the LM and set its scale
to ``0.0``. Within-word and word-end hypotheses are pruned separately.

This algorithm requires a lexicon, an acoustic model (used only to build the search tree, i.e. for allophones
and state tying, not for scoring) and a language model, in addition to the label scorer(s).

* ``max-beam-size`` (int list): maximum number of *within-word* hypotheses in the beam, one value per label scorer.
  Default: unset, i.e. no beam-size pruning is applied.
* ``max-word-end-beam-size`` (int): maximum number of *word-end* hypotheses kept. If unset, word-end
  hypotheses are pruned together with within-word hypotheses using the same global beam. Default: unset.
* ``score-threshold`` (float list): score-based pruning of *within-word* hypotheses, one value per label scorer.
  Default: unset, i.e. no score-based pruning is applied.
* ``word-end-score-threshold`` (float): score-based pruning of *word-end* hypotheses, relative to
  ``score-threshold``. If unset, word-end hypotheses use global score pruning. Default: unset.
* ``num-histogram-bins`` (int): see above. Default ``100``.
* ``collapse-repeated-labels`` (bool): see above. Default ``false``.
* ``sentence-end-fall-back`` (bool): if no active word-end hypothesis exists at the end of a segment (i.e. every
  surviving hypothesis is still mid-word), controls what happens instead of failing. If enabled, each
  within-word hypothesis falls back to its last completed word, discarding the incomplete final word. If
  disabled, an empty hypothesis is produced instead. Default ``true``.
* ``recombination-mode``, ``log-stepwise-statistics``, ``cache-cleanup-interval``,
  ``maximum-stable-delay``, ``maximum-stable-delay-pruning-interval``: same meaning as for
  ``lexiconfree-timesync-beam-search`` above.
* ``tree-builder-type`` (enum): which tree builder is used to construct the search tree from the lexicon and
  acoustic model: ``ctc``, ``rna``, ``aed`` or ``hmm`` (``classic-hmm`` and ``minimized-hmm`` only work for
  ``Search`` and are not compatible with ``SearchV2``). Choose the one matching your model's topology.
  See :ref:`Search tree types` below for a full description of each. **This must always be set explicitly**:
  the code-level default when unset is ``minimized-hmm``, which is not compatible with SearchV2 (see below).

Example config:

.. code-block:: ini

    [*.search-algorithm]
    type                     = tree-timesync-beam-search
    max-beam-size            = 5000
    max-word-end-beam-size   = 200
    score-threshold          = 14.0
    word-end-score-threshold = 0.5
    tree-builder-type        = ctc

    [*.lexicon]
    file                     = /path/to/lexicon.xml.gz

    [*.acoustic-model]
    ; allophones / state-tying configuration, see "Common component configuration"

    [*.language-model]
    type                     = ARPA
    file                     = /path/to/lm.gz
    scale                    = 1.5

The search tree is cached on disk next to the lexicon after it is built the first time, so subsequent runs
with the same lexicon/acoustic-model configuration start up faster.

tree-labelsync-beam-search
^^^^^^^^^^^^^^^^^^^^^^^^^^

A label-synchronous beam search that decodes over a prefix tree built from the pronunciation lexicon, similar
in structure to ``tree-timesync-beam-search``, but terminating hypotheses via an explicit sentence-end symbol
rather than by running out of input frames -- the label-synchronous counterpart to
``tree-timesync-beam-search``, analogous to how ``lexiconfree-labelsync-beam-search`` relates to
``lexiconfree-timesync-beam-search``. Intended for AED models decoding over a closed/lexicon-constrained
vocabulary. The search tree should be built with the ``aed`` tree builder (see :ref:`Search tree types`), since
attention encoder-decoder models do not produce the frame-synchronous blank/loop structure the other tree
topologies are built for.

The language model is used just as for the timesync version. The sentence-end label index is derived from the
lexicon's ``special="sentence-end"``/``"sentence-boundary"`` lemma (which must have exactly one phoneme in its
pronunciation) rather than being configurable as a separate parameter, so that it stays consistent with the label
index used for the search tree itself.

* ``max-beam-size``, ``max-word-end-beam-size``, ``score-threshold``, ``word-end-score-threshold``,
  ``num-histogram-bins``, ``sentence-end-fall-back``, ``recombination-mode``, ``log-stepwise-statistics``,
  ``cache-cleanup-interval``, ``maximum-stable-delay``, ``maximum-stable-delay-pruning-interval``: same meaning
  and defaults as for ``tree-timesync-beam-search`` above.
* ``length-norm-scale``, ``max-labels-per-timestep``: same meaning and defaults as for
  ``lexiconfree-labelsync-beam-search`` above.
* ``tree-builder-type`` (enum): the same shared parameter as for ``tree-timesync-beam-search`` (see
  :ref:`Search tree types`). Should always be set to ``aed`` for this algorithm; the other tree topologies are
  built around blank/loop transitions this algorithm does not use.

.. code-block:: ini

    [*.search-algorithm]
    type                     = tree-labelsync-beam-search
    max-beam-size            = 50
    max-word-end-beam-size   = 20
    score-threshold          = 5.0
    word-end-score-threshold = 0.5
    length-norm-scale        = 1.0
    tree-builder-type        = aed

    [*.lexicon]
    file                     = /path/to/lexicon.xml.gz

    [*.acoustic-model]
    ; allophones / state-tying configuration, see "Common component configuration"

    [*.language-model]
    type                     = ARPA
    file                     = /path/to/lm.gz
    scale                    = 0.8

Search tree types
^^^^^^^^^^^^^^^^^

``tree-builder-type`` controls which internal tree builder assembles the search tree that
``tree-timesync-beam-search`` decodes over. This is independent of the label scorer(s) you use -- it only
determines the *shape* of the tree (whether/how blank and repeated-label loops are modeled, whether
context-dependent triphones are built, etc.), so it must match the alignment topology your model and label
scorer(s) actually produce.

``classic-hmm`` and ``minimized-hmm`` are also valid ``tree-builder-type`` choices, but they are not compatible
with SearchV2 -- they build the full/minimized cross-word triphone HMM tree used by the classic
``Search::SearchAlgorithm``/``AdvancedTreeSearch`` and its ``Mm::FeatureScorer``-based scoring, so they are not
listed below. Note that ``minimized-hmm``/``previousBehavior`` is still the code-level default when
``tree-builder-type`` is left unset, so it must always be set explicitly to one of the options below for
``tree-timesync-beam-search``/``tree-labelsync-beam-search``.

* ``ctc``: builds a tree without any cross-word context, with an explicit blank state after every label and
  self-loops on both label and blank states, matching CTC's frame-wise blank/label-repetition topology. Each
  label state also has a direct transition to the next label state, skipping the blank state in between (except
  between two identical consecutive labels when ``force-blank-between-repeated-labels`` is enabled), so passing
  through blank between two different labels is optional rather than mandatory. Sub-parameters (read from the
  same ``search-algorithm`` configuration):

  * ``allow-label-loop`` (bool): allow a label to repeat via a self-loop. Default ``true``.
  * ``allow-blank-loop`` (bool): allow blank to repeat via a self-loop. Default ``true``.
  * ``force-blank-between-repeated-labels`` (bool): require a blank between two identical consecutive labels
    (only takes effect if ``allow-label-loop`` is disabled). Default ``true``.
* ``rna``: a variant of the ``ctc`` tree builder tailored to RNA/monotonic-transducer topology, i.e. exactly
  one output per input frame with no free label repetition. Same blank-skip transitions and sub-parameters as
  ``ctc``, but with different defaults: ``allow-label-loop`` and ``force-blank-between-repeated-labels`` both
  default to ``false``.
* ``aed``: builds a plain per-phoneme tree with no blank state and no label/blank self-loops at all, since
  attention encoder-decoder models do not produce a frame-synchronous alignment with blank/loop symbols the way
  CTC/transducer models do. Use this for AED label scorers when decoding over a closed vocabulary with
  ``tree-labelsync-beam-search``.
* ``hmm``: builds a tree supporting optional diphone (not triphone) cross-word context, without minimization and
  without skip-transitions. Cross-word context is applied per phoneme, not tree-wide: only phonemes marked
  context-dependent in the lexicon's phoneme inventory get a coarticulated root state and thus cross-word
  diphone modeling; phonemes marked context-independent use the plain context-independent root and get no
  cross-word context at all. A lighter-weight option for GMM or hybrid NN/HMM acoustic models when full triphone
  modeling isn't needed. Sub-parameter: ``add-ci-transitions`` (bool, default ``false``) -- insert context-independent
  acoustic transitions between words, useful for non-fluid (isolated-word-like) speech even when the acoustic
  model was trained on fluent speech. Warning: this tree builder is currently experimental and has not been fully
  tested; some edge cases or input configurations may not be handled correctly yet.

Multiple label scorers and per-stage parameters
------------------------------------------------

All three algorithms can use more than one label scorer at once (e.g. an acoustic model scorer plus a separate
language model scorer), applied one after another with pruning in between. The number of label scorers is set
independently via ``num-label-scorers`` (see below); ``max-beam-size`` and ``score-threshold`` then take one
value **per label scorer**, separated by whitespace, applied in the same order the label scorers are scored:

.. code-block:: ini

    [*.search-algorithm]
    max-beam-size   = 2400 1200
    score-threshold = 20.0 14.0

Here the beam is pruned to 2400 hypotheses (with a score threshold of 20.0) after the first label scorer, and
further pruned to 1200 hypotheses (score threshold 14.0) after the second. If only a single value is given, it is
used after the (only) label scorer.

.. _SearchV2 Label Scorers:

Label scorers
--------------

Instead of a ``Mm::FeatureScorer``, SearchV2 algorithms obtain scores from one or more ``Nn::LabelScorer``
instances, configured under the ``label-scorer`` selector (or ``label-scorer-1``, ``label-scorer-2``, ... if
``num-label-scorers`` is greater than one):

.. code-block:: ini

    [*.search-algorithm]
    num-label-scorers = 1

    [*.search-algorithm.label-scorer]
    type              = state-managed-onnx
    scale             = 1.0
    ; further label-scorer specific parameters, e.g. ONNX model file, I/O tensor names, ...

Every label scorer constructed this way (whatever its ``type``) is automatically wrapped in a
``Nn::ScaledLabelScorer``, so a log-linear ``scale`` (float, default ``1.0``) is always available as a
parameter on the same selector, e.g. ``*.search-algorithm.label-scorer.scale``.

Scores are negative log-probabilities (i.e. costs), the same convention as the AM/LM scores in the classic
search: lower is better, and a hypothesis's total score is the sum of the (scaled) scores of everything that
contributed to it. A model producing log-probabilities directly (e.g. a ``log_softmax`` output) therefore needs
its sign flipped before being used as a score -- see ``negate-input`` under :ref:`prior / no-op` below for a
built-in way to do this.

Built-in label scorer types include:

* ``no-context-onnx``, ``fixed-context-onnx``, ``stateful-onnx``, ``state-managed-onnx``: forward
  features (and, depending on type, label history/hidden state) through an ONNX model.
* ``encoder-decoder`` / ``encoder-only``: wrap a separate encoder (see ``encoder`` sub-selector) that
  pre-processes features, combined with a decoder label scorer (or no decoder, for encoder-only models).
* ``ctc-prefix``: wraps a time-synchronous CTC scorer and derives label-synchronous prefix scores from it
  (useful with ``lexiconfree-labelsync-beam-search``).
* ``combine``: log-linearly combines multiple sub-label-scorers into one (e.g. AM + LM), configured via nested
  selectors.
* ``transition``: returns fixed scores per transition type, useful for e.g. modeling label-loop penalties.
* ``prior`` / ``no-op``: pass through externally computed scores as-is, optionally subtracting a prior.

Label scorer configuration is its own (large) topic. The essential thing to know for SearchV2 is only that
``*.search-algorithm.label-scorer.type`` selects the implementation, and that ``num-label-scorers`` /
``max-beam-size`` / ``score-threshold`` must agree in how many scorers/stages are configured.

The rest of this section describes each built-in label scorer type in more detail.

ONNX model configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The four ``*-onnx`` label scorers below (and the ``onnx``/``chunked-onnx`` encoders used by
``encoder-decoder``/``encoder-only``) all wrap one or more ONNX Runtime sessions and share the same
sub-configuration pattern for each wrapped model, under a model-specific selector (e.g. ``onnx-model`` for the
single-model scorers):

* ``<model>.session.file`` (string): path to the exported ``.onnx`` file. No default -- required.
* ``<model>.session.intra-op-num-threads`` / ``inter-op-num-threads`` (int): ONNX Runtime threading options. Default ``1`` each.
* ``<model>.session.execution-provider-type`` (``cpu``/``cuda``): which execution provider runs the model. Default ``cpu``.
* ``<model>.io-map.<logical-name>`` (string): maps a logical input/output role used internally by the label
  scorer (e.g. ``input-feature``, ``scores``, ``history``, see each scorer below) to the actual tensor name in
  the exported ONNX graph, since exporters rarely produce the same names RASR expects. No default -- required
  for every non-optional input/output of the model.

.. code-block:: ini

    [*.search-algorithm.label-scorer.onnx-model]
    session.file                    = /path/to/model.onnx
    session.intra-op-num-threads    = 4
    session.execution-provider-type = cpu
    io-map.input-feature            = source
    io-map.scores                   = log_softmax

no-context-onnx
^^^^^^^^^^^^^^^^

Forwards only the input feature of the current timestep through an ONNX model, without any label history. This
is suitable e.g. for a CTC output layer (linear + log-softmax) that is scored separately from its encoder. If
the CTC output is the *only* output of the model, prefer wrapping encoder and output layer together as a
single ``encoder-only`` label scorer instead.

* ``onnx-model.io-map.input-feature`` : ONNX tensor name that receives the input feature. No default -- required.
* ``onnx-model.io-map.scores`` : ONNX tensor name of the resulting score vector. No default -- required.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type                            = no-context-onnx
    onnx-model.session.file         = /path/to/ctc_head.onnx
    onnx-model.io-map.input-feature = source
    onnx-model.io-map.scores        = log_probs

fixed-context-onnx
^^^^^^^^^^^^^^^^^^^^

Forwards the input feature of the current timestep together with a fixed-size window of previous label-history
tokens through an ONNX model. A typical use case is a neural transducer prediction network with a fixed-size
history instead of a recurrent state.

* ``onnx-model.io-map.input-feature`` / ``history`` / ``scores`` : ONNX tensor names for the current feature,
  the history-token tensor and the resulting score vector, respectively.
* ``start-label-index`` (int): label index used to pad the history before any label has been emitted. Default ``0``.
* ``history-length`` (int): number of previous labels kept and passed as history. Default ``1``.
* ``blank-updates-history`` / ``silence-updates-history`` / ``loop-updates-history`` (bool): whether a
  previously emitted blank/silence/repeated (looped) label is pushed into the history, or skipped over. All
  default to ``false`` (i.e. those labels do not update the history).
* ``vertical-label-transition`` (bool): whether non-blank label transitions are "vertical", i.e. do not advance
  the time step (relevant for certain transducer topologies). Default ``false``.
* ``max-batch-size`` (int): maximum number of histories forwarded through the ONNX model in one call, hypotheses
  beyond this are split into further calls. Default unbounded.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type                    = fixed-context-onnx
    history-length          = 1
    loop-updates-history    = true
    onnx-model.session.file = /path/to/transducer_predictor.onnx

stateful-onnx
^^^^^^^^^^^^^^

Scores by forwarding an arbitrary set of *hidden state* tensors (rather than an explicit label-history window)
through ONNX models, and is built from **three** separate ONNX models instead of one:

* a **state initializer** (``state-initializer-model``) that produces the initial hidden state(s) for the first
  step, optionally based on the encoder input sequence;
* a **state updater** (``state-updater-model``) that produces updated hidden state(s) from the previous
  state(s) and the next token;
* a **scorer** (``scorer-model``) that computes scores from the (updated) hidden state(s).

Every hidden state tensor is identified by a *state name* that must appear consistently in the ONNX metadata of
all three models (mapping each model's own input/output tensor name to that shared state name), so the label
scorer can match up state tensors across the three sessions even though their local input/output names differ.
A common use case is an attention encoder-decoder (AED) model with cross-attention over encoder states, or a
stateful (recurrent) language model.

* ``state-initializer-model.*`` / ``state-updater-model.*`` / ``scorer-model.*`` : each configured like
  ``onnx-model`` above (``session.file``, ``io-map``, ...).
* ``blank-updates-history`` / ``silence-updates-history`` / ``loop-updates-history`` (bool): as above, whether
  the respective label types trigger a state update. Default ``false``.
* ``max-batch-size`` (int): maximum number of hidden states forwarded through the scorer model at once. Default unbounded.
* ``max-cached-score-vectors`` (int): size of an LRU cache mapping scoring contexts to already-computed score
  vectors, to avoid recomputation and bound memory use on very long segments. Default ``1000``.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type                                 = stateful-onnx
    state-initializer-model.session.file = /path/to/state_init.onnx
    state-updater-model.session.file     = /path/to/state_update.onnx
    scorer-model.session.file            = /path/to/scorer.onnx

state-managed-onnx
^^^^^^^^^^^^^^^^^^^^

Similar in spirit to ``stateful-onnx`` (ONNX-based, hidden-state driven), but hidden-state bookkeeping across
hypotheses is delegated to a pluggable ``StateManager`` instead of being handled ad hoc. Each scoring context
only stores the state *slice* produced for its most recent token plus a parent pointer, so states form a tree
rather than duplicating the full prefix state per hypothesis -- this is what allows efficient transformer KV
caches (splitting/merging/rebasing state across beam search steps) without quadratic memory growth.

* ``onnx-model.*`` : the single wrapped ONNX model, configured like ``onnx-model`` above. Relevant logical I/O
  names include ``token``, ``token-length``, ``prefix-length``, ``scores``, ``encoder-states`` and
  ``encoder-states-size`` (the latter two only if the model attends over encoder output directly).
* ``state-manager.type`` (``lstm``/``transformer``/``transformer-16bit``/``transformer-8bit``): which state
  representation/caching strategy is used. Use ``lstm`` for simple recurrent states; the ``transformer*``
  variants implement KV-cache tree management, with the ``-16bit``/``-8bit`` variants storing the cache in
  reduced precision to save memory. Default ``lstm``.
* ``start-labels`` (int list): initial context/history tokens fed for the very first step. Default: unset,
  i.e. no initial context tokens.
* ``blank-updates-history`` / ``silence-updates-history`` / ``loop-updates-history`` (bool): as above. Default ``false``.
* ``max-batch-size`` (int): maximum number of scoring contexts forwarded through the ONNX model at once. Default unbounded.
* ``max-cached-score-vectors`` (int): size of the score-vector cache, as for ``stateful-onnx``.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type                    = state-managed-onnx
    onnx-model.session.file = /path/to/aed_decoder.onnx
    state-manager.type      = transformer

encoder-decoder / encoder-only
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``encoder-decoder`` is a glue label scorer combining a separate **encoder** (which pre-processes raw input
features into encoded representations, without needing any scoring context) with a **decoder**, which is any
other label scorer from this list (typically ``state-managed-onnx`` or ``stateful-onnx``) that consumes the
encoder output. It automatically handles passing encoder output into the decoder as it becomes available.
``encoder-only`` is the same idea without a "real" decoder: the encoder output *is* the score (wrapped in a
trivial pass-through decoder), used when encoder and output/CTC layer are exported as a single ONNX graph.

* ``encoder.type`` (``onnx``/``chunked-onnx``): which encoder implementation to use. Default ``onnx``.
* ``encoder.onnx-model.*`` : the encoder's ONNX model, configured like ``onnx-model`` above.
* ``encoder.inputs-per-output`` (int): number of input features consumed per encoder output frame (e.g. for a
  subsampling encoder). ``0`` infers this at runtime. Default ``0``.
* ``encoder.input-step-size`` (int): shift in input features between consecutive encoder outputs; ``0`` copies
  the value from ``inputs-per-output``. Default ``0``.
* For ``chunked-onnx`` additionally: ``encoder.chunk-size`` / ``step-size`` (int, in input features, default
  ``1`` each) define how the input is split into overlapping chunks fed through the encoder separately (useful
  to bound memory/latency for long or streaming inputs); ``left-padding`` / ``right-padding`` (int, default
  ``0`` each) pad each chunk with additional context on either side; ``zero-padding`` (bool, default ``false``)
  pads the first/last chunk with zeros to a uniform size; ``window-type`` (default ``triangular``) and
  ``interpolation-mode`` (default ``no-interpolation``) control how overlapping chunk outputs are blended back
  together.
* ``decoder.*`` : configuration of the wrapped decoder label scorer (``encoder-decoder`` only), same
  parameters as the chosen ``decoder.type`` label scorer.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type                            = encoder-decoder
    encoder.type                    = chunked-onnx
    encoder.chunk-size              = 50
    encoder.step-size               = 25
    encoder.left-padding            = 10
    encoder.right-padding           = 5
    encoder.onnx-model.session.file = /path/to/encoder.onnx
    decoder.type                    = state-managed-onnx
    decoder.onnx-model.session.file = /path/to/decoder.onnx

ctc-prefix
^^^^^^^^^^^

Wraps a time-synchronous CTC scorer (any other label scorer producing per-frame CTC-style scores, e.g.
``no-context-onnx`` or ``encoder-only``) and derives label-*synchronous* prefix scores from it, by summing over
all the time-synchronous CTC alignments consistent with a given label prefix. This lets a CTC model's scores be
used together with a label-synchronous search such as ``lexiconfree-labelsync-beam-search``, e.g. to combine an
AED and a CTC model at the label level.

* ``blank-label-index`` (int): index of the blank label in the wrapped CTC scorer's vocabulary. Default ``0``,
  but should always be set explicitly to match your vocabulary.
* ``vocab-size`` (int): number of labels in the wrapped CTC scorer's vocabulary. Default ``0``, but must be set
  explicitly -- a default of ``0`` produces a degenerate, empty score matrix.
* ``label-scorer.*`` : configuration of the wrapped time-synchronous CTC label scorer (nested selector).

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type                                 = ctc-prefix
    blank-label-index                    = 0
    vocab-size                           = 5000
    label-scorer.type                    = no-context-onnx
    label-scorer.onnx-model.session.file = /path/to/ctc_head.onnx

combine
^^^^^^^^

Log-linearly combines multiple sub-label-scorers into a single one, assuming all sub-scorers share the same
label alphabet: ``combined_score = sum_i(score_i * scale_i)``. The combined timeframe is the maximum over the
sub-scorers' timeframes. This is the usual way to log-linearly combine e.g. an acoustic model scorer and a
neural language model scorer inside one label-scorer "stage" (as opposed to using two separate
``num-label-scorers`` stages with pruning in between).

* ``num-scorers`` (int): number of sub-scorers to combine. Default ``1``.
* ``scorer-<i>.*`` (for ``i`` from ``1`` to ``num-scorers``): configuration of the ``i``-th sub-scorer,
  including its own ``type``.
* ``scorer-<i>.scale`` (float): log-linear weight of the ``i``-th sub-scorer's score. This is just the same
  implicit ``scale`` parameter every label scorer has (see :ref:`SearchV2 Label Scorers` above), applied here
  once per sub-scorer since each ``scorer-<i>`` is itself a full label scorer. Default ``1.0``.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type                             = combine
    num-scorers                      = 2
    scorer-1.type                    = state-managed-onnx
    scorer-1.scale                   = 1.0
    scorer-1.onnx-model.session.file = /path/to/am.onnx
    scorer-2.type                    = state-managed-onnx
    scorer-2.scale                   = 0.3
    scorer-2.onnx-model.session.file = /path/to/lm.onnx

transition
^^^^^^^^^^^

Returns a fixed, configured score for each transition type, independent of any input features or label
history. Useful to model e.g. label-loop penalties or blank/word/sentence-end exit penalties analogous to HMM
transition penalties, without needing a model to produce them. Recognized transition types are
``label-to-label``, ``label-loop``, ``label-to-blank``, ``blank-to-label``, ``blank-loop``,
``label-to-silence``, ``silence-to-label``, ``silence-loop``, ``initial-label``, ``initial-blank``,
``initial-silence``, ``word-exit``, ``nonword-exit``, ``silence-exit`` and ``sentence-end``.

* ``<transition-type>-score`` (float): fixed score for the given transition type, e.g. ``label-loop-score``.
  Any transition type not explicitly set defaults to a score of ``0.0``.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type             = transition
    label-loop-score = 3.0
    blank-loop-score = 3.0
    word-exit-score  = 5.0

prior / no-op
^^^^^^^^^^^^^^

Both assume the input features passed to the label scorer are *already finished* score vectors (e.g. output of
a log-softmax layer computed elsewhere, such as in a Flow network or transmitted externally via the Python
bindings) and simply return the score at the current step rather than running any model.

``no-op`` passes these scores through completely unchanged.

``prior`` additionally supports negating the input
and/or subtracting a prior from it, which is useful e.g. to convert posteriors into (pseudo-)likelihoods:

* ``negate-input`` (bool): negate the incoming scores before further processing. Default ``false``.
* ``prior-file`` (string): path to a prior file (same format/mechanism as the acoustic model prior, see
  :doc:`common_config`). Default: unset (empty path). Since ``priori-scale`` defaults to a nonzero value, leaving
  this unset does *not* by itself disable the prior -- set ``priori-scale = 0.0`` explicitly to disable it without
  providing a file.
* ``priori-scale`` (float): log-linear scale applied to the prior before subtracting it from the score. Default ``1.0``.

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type         = prior
    negate-input = true
    prior-file   = /path/to/prior.xml
    priori-scale = 0.3

Running SearchV2 in an Flf network
------------------------------------

The :ref:`recognizer-v2` Flf node runs a ``SearchAlgorithmV2`` over incoming speech segments and outputs Flf
lattices, analogous to the classic ``recognizer`` node but working with SearchV2 instead of
``Search::SearchAlgorithm``. See :ref:`Flf Nodes` for the general Flf network mechanism.

.. code-block:: ini

    [*.network.recognizer-v2]
    type            = recognizer-v2

    [*.network.recognizer-v2.feature-extraction]
    ; Flow network for feature extraction, same as for other recognizer nodes

    [*.network.recognizer-v2.search-algorithm]
    type            = lexiconfree-timesync-beam-search
    max-beam-size   = 2000
    score-threshold = 14.0

    [*.network.recognizer-v2.search-algorithm.label-scorer]
    type            = state-managed-onnx
    ; ...

**Port assignment**

.. code-block:: ini

    input:
      0:bliss-speech-segment
    output:
      0:lattice

Compared to ``recognizer``, ``recognizer-v2`` is intentionally minimal: it does not do posterior pruning,
confidence scoring or non-word-closure filtering itself. Chain the usual Flf nodes (e.g. ``posterior-pruning``,
a confidence node, ...) after ``recognizer-v2`` if you need that post-processing. The output lattice's language
model scale is taken from ``*.language-model.scale`` (via the model combination), and the acoustic score axis
is left unscaled.

Using SearchV2 from Python
----------------------------

This is likely the most important section for users who want to run recognition without writing an Flf
network or a C++ tool: the ``librasr`` Python module wraps a ``SearchAlgorithmV2`` for direct use from Python,
e.g. from a RETURNN training/decoding loop or a standalone script.

Installing and importing ``librasr``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``librasr`` is a compiled Python extension (built with pybind11) that is also set up as a proper, pip-installable
package (see ``setup.py``), which builds it via CMake behind the scenes:

.. code-block:: bash

    pip install .
    # or, for an editable/development install:
    pip install -e .

    # pass extra CMake options (e.g. to enable modules) via CMAKE_ARGS, for example:
    CMAKE_ARGS="-DMODULE_ONNX=1 -DMODULE_TENSORFLOW=0" pip install .

Building requires the same toolchain/dependencies as the C++ build (see :doc:`architecture`) plus ``pybind11``
and ``numpy``. Note that ``LibRASR`` (the CMake tool that builds ``librasr``) is enabled by default but also
needs ``MODULE_PYTHON`` and ``MODULE_NN`` enabled to get the ``SearchAlgorithm``/``LabelScorer`` bindings used
below (both are on by default); without them you only get the base ``librasr`` module (config handling,
allophone FSA building, lexicon access).

Once installed, everything lives under a single import:

.. code-block:: python

    import librasr

If you built the extension manually via plain CMake instead of ``pip install .``, the compiled module ends up
at ``arch/<system>-<architecture>-<build-type>/librasr*.so`` -- add that directory to ``PYTHONPATH`` (or
``sys.path``) before importing it.

Configuration object
^^^^^^^^^^^^^^^^^^^^^^

Everything constructed from Python (search algorithm, label scorers, ...) is configured with a
``librasr.Configuration`` object, which mirrors the ``.ini``-style RASR configuration used everywhere else in
this document -- you can build it programmatically, load it from a file, or both:

.. code-block:: python

    import librasr

    config = librasr.Configuration()
    # load a standard RASR config file (same format as used by all other tools/examples in this document)
    config.set_from_file("recognition.config")
    # and/or set individual parameters directly, overriding anything loaded from file
    config.set("*.search-algorithm.type", "lexiconfree-timesync-beam-search")
    config.set("*.search-algorithm.max-beam-size", "12")
    config.set("*.lexicon.file", "/path/to/lexicon.xml.gz")

* ``Configuration()`` : create an empty configuration (default selection root is ``lib-rasr``, so both
  ``*.foo`` and ``lib-rasr.foo`` style keys apply to it).
* ``config.set(name, value="true")`` : set a single configuration key, e.g. ``config.set("*.search-algorithm.type", "tree-timesync-beam-search")``.
* ``config.set_from_file(path)`` : load a ``.ini``-style RASR config file, same as ``--config=path`` for the
  command-line tools. Returns ``True`` on success.
* ``config[name]`` : read back a resolved value (``__getitem__``), returns ``None`` if unset.
* ``config.get_selection()`` / ``config.set_selection(name)`` : get/set the selection root used when resolving relative keys.
* ``config.resolve(value)`` : resolve ``$(VAR)`` substitutions in a string the same way the config parser does.
* ``config.enable_logging()`` : turn on RASR's normal log output (XML log to stderr) for this process.
* ``Configuration(other_config)`` / ``Configuration(other_config, selection)`` : copy a configuration, optionally
  rooted at a different selection, e.g. to build a sub-config for a nested component.

Running recognition: ``SearchAlgorithm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``librasr.SearchAlgorithm`` is the Python-facing wrapper around ``Search::SearchAlgorithmV2``. It is constructed
directly from a ``Configuration`` and internally builds the lexicon/acoustic-model/language-model/label-scorer(s)
it needs (same ``*.search-algorithm.*``, ``*.lexicon.*``, ``*.label-scorer.*``, ... configuration as everywhere
else in this document -- see :ref:`Search algorithms` and :ref:`SearchV2 Label Scorers`):

.. code-block:: python

    import numpy as np
    import librasr

    config = librasr.Configuration()
    config.set_from_file("recognition.config")

    search = librasr.SearchAlgorithm(config)

    # simplest usage: recognize one segment given all its features at once
    features = np.load("segment_features.npy")  # shape [T, F] (or [1, T, F])
    traceback = search.recognize_segment(features)
    for item in traceback:
        print(item.lemma, item.start_time, item.end_time, item.am_score, item.lm_score)

The lower-level, step-by-step API (what ``recognize_segment`` does internally) gives you control over streaming
and intermediate results:

.. code-block:: python

    search.enter_segment()             # start a new segment
    for feature_chunk in feature_stream:
        search.put_feature(feature_chunk)      # single feature, shape [F] or [1, F]
        # or: search.put_features(feature_chunk)  # multiple features, shape [T, F] or [1, T, F]

        # optional: peek at an intermediate (possibly unstable) result while streaming
        partial = search.get_current_best_traceback()
    search.finish_segment()            # signal that the segment is complete

    best = search.get_current_best_traceback()      # final single-best result
    stable = search.get_common_prefix()              # stable prefix shared by all current hypotheses
    n_best = search.get_current_n_best_list(10)       # list of `n` Traceback objects

**SearchAlgorithm methods:**

* ``SearchAlgorithm(config)`` : construct and initialize everything (lexicon, acoustic/language model, label
  scorer(s), search tree if needed) from a ``Configuration``.
* ``enter_segment()`` : start a new segment; clears hypotheses/buffers left over from a previous segment.
* ``put_feature(feature_vector)`` : feed a single feature vector, numpy array of shape ``[F]`` or ``[1, F]``.
* ``put_features(feature_array)`` : feed multiple feature vectors at once, shape ``[T, F]`` or ``[1, T, F]``.
* ``finish_segment()`` : signal that all features for the segment have been passed; finalizes the search.
* ``get_current_best_traceback()`` : run any pending decode steps and return the current single-best
  ``Traceback`` (list of ``TracebackItem``). Safe to call mid-segment for streaming/partial results.
* ``get_common_prefix()`` : return the ``Traceback`` of the stable common prefix shared by all currently active
  hypotheses (the part of the result that cannot change anymore, even before the segment ends).
* ``get_current_n_best_list(n)`` : return a list of up to ``n`` ``Traceback`` objects.
* ``recognize_segment(features)`` : convenience method equivalent to ``enter_segment`` + ``put_features`` +
  ``finish_segment`` + ``get_current_best_traceback`` in one call. ``features`` has shape ``[T, F]`` or ``[1, T, F]``.
* ``recognize_segment_n_best(features, n)`` : same, but returns ``get_current_n_best_list(n)``.
* ``model_combination()`` : return the underlying ``ModelCombination`` (see below) for accessing/adjusting
  scales at runtime.

**Traceback / TracebackItem:** a ``Traceback`` is a plain Python list of ``TracebackItem``, each with
read/write attributes ``lemma`` (str), ``am_score`` (float), ``lm_score`` (float), ``start_time`` (int) and
``end_time`` (int, both in feature-frame units). ``str(item)`` gives just the lemma.

**Adjusting scales at runtime** via ``model_combination()``, without touching the config or rebuilding anything:

.. code-block:: python

    mc = search.model_combination()
    mc.language_model().set_scale(1.5)

    label_scorer = mc.label_scorer()           # index defaults to 0; pass an index for multiple label scorers
    label_scorer.set_scale(0.5)

Writing a custom label scorer in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond the built-in :ref:`SearchV2 Label Scorers`, ``librasr`` lets you implement a label scorer entirely in
Python -- e.g. to call a PyTorch model directly, without exporting to ONNX -- and register it under a type name
that can then be used like any built-in type in ``*.search-algorithm.label-scorer.type``. This works by
subclassing ``librasr.LabelScorer`` and registering the class:

.. code-block:: python

    import librasr

    class MyTorchLabelScorer(librasr.LabelScorer):
        def __init__(self, config):
            super().__init__(config)
            self.model = load_my_torch_model(config)
            self.buffered_inputs = []

        def reset(self):
            self.buffered_inputs = []

        def signal_no_more_features(self):
            pass  # mark that no further add_inputs() calls will come for this segment, if relevant

        def get_initial_scoring_context(self):
            # any hashable python object representing "no history yet", e.g.:
            return ()

        def allowed_transition_types(self):
            return [librasr.TransitionType.LABEL_TO_LABEL, librasr.TransitionType.LABEL_LOOP]

        def extended_scoring_context(self, context, next_token, transition_type):
            return context + (next_token,)  # e.g. extend a label-history tuple

        def add_inputs(self, inputs):
            # inputs: numpy array of shape [T, F]
            self.buffered_inputs.append(inputs)

        def compute_scores_with_times(self, contexts):
            # contexts: list of scoring-context objects (as returned by get_initial_scoring_context /
            # extended_scoring_context) to score in one batch.
            # Return a list with, for each context, either None (not enough input buffered yet to score it)
            # or a tuple (score_list, timestamp).
            return self.model.score_batch(contexts, self.buffered_inputs)

    librasr.register_label_scorer_type("my-torch-scorer", MyTorchLabelScorer)

.. code-block:: ini

    [*.search-algorithm.label-scorer]
    type = my-torch-scorer
    ; any additional keys your __init__ reads from `config`

Register the type *before* constructing the ``SearchAlgorithm``/``Configuration`` that references it. Python
label scorers can be freely combined with native ones, e.g. wrapped in ``combine`` alongside an ONNX-based
scorer, or used as the ``decoder`` of an ``encoder-decoder`` label scorer.

* ``librasr.register_label_scorer_type(name, label_scorer_cls)`` : register ``label_scorer_cls`` (a subclass of
  ``librasr.LabelScorer``) under ``name``, making it selectable via ``label-scorer.type = <name>`` in any config
  from then on, for the lifetime of the process.
* ``librasr.TransitionType`` : enum mirroring the C++ ``Nn::TransitionType`` values used elsewhere in this
  document (``LABEL_TO_LABEL``, ``LABEL_LOOP``, ``LABEL_TO_BLANK``, ``BLANK_TO_LABEL``, ``BLANK_LOOP``,
  ``LABEL_TO_SILENCE``, ``SILENCE_TO_LABEL``, ``SILENCE_LOOP``, ``INITIAL_LABEL``, ``INITIAL_BLANK``,
  ``INITIAL_SILENCE``, ``WORD_EXIT``, ``NONWORD_EXIT``, ``SILENCE_EXIT``, ``SENTENCE_END``).
* Required overrides on a ``librasr.LabelScorer`` subclass: ``reset``, ``signal_no_more_features``,
  ``get_initial_scoring_context``, ``allowed_transition_types``, ``extended_scoring_context``, ``add_inputs``,
  ``compute_scores_with_times`` (signatures shown in the example above).
* A native label scorer wrapping a Python one exposes ``get_sub_scorer(index=0)`` on ``ScaledLabelScorer``
  (e.g. to reach into a ``combine`` or ``encoder-decoder`` scorer instance from Python at runtime) and
  ``set_scale``/``scale`` for adjusting its log-linear weight without touching the config.

Reading results
-----------------

Regardless of how it is invoked, a SearchV2 algorithm exposes results in two forms:

* **Traceback** (``getCurrentBestTraceback``): the single-best sequence of recognized lemmas/pronunciations
  with time and score information, written to the log as an ``<traceback>`` element (and, for
  ``recognizer-v2``, also as a plain ``<orth>`` element with the recognized words).
* **Word lattice** (``getCurrentBestWordLattice`` / ``getCurrentBestLatticeTrace``): the full search lattice
  (or an n-best-ish subset of it, depending on the algorithm's pruning), converted to an Flf lattice by
  ``recognizer-v2`` for further processing (rescoring, CTM export, ...) using the standard :ref:`Flf Nodes`.

Both ``getCurrentBestTraceback`` and ``getCurrentBestWordLattice``/``getCurrentBestLatticeTrace`` can also be
queried mid-segment, before ``finishSegment`` is called, to get an unstable intermediate ("streaming") result.

Tuning tips
------------

* Start with a small vocabulary/model and generous pruning (large ``max-beam-size``, large
  ``score-threshold``) to verify correctness, then tighten both together to trade accuracy for speed. If
  results change noticeably when tightening only one of the two, the other is likely already the binding
  constraint. To get the best speech-accuracy trade-off, you should usually have a well-tuned ``score-threshold``
  (as low as possible) with an additional high ``max-beam-size`` that mitigates peaks in the number of hypotheses.
* Enable ``log-stepwise-statistics = true`` temporarily to see beam sizes and score spreads per step in the
  log, which helps decide which pruning parameter is actually limiting the beam.
* For ``tree-timesync-beam-search``, tune ``max-word-end-beam-size``/``word-end-score-threshold``
  independently from the within-word beam if word-end hypotheses are pruned too aggressively (or not enough)
  relative to within-word hypotheses.
* For low-latency streaming use cases, set ``maximum-stable-delay`` to bound how many frames of output can
  still change, at the cost of small accuracy loss versus fully offline decoding.
* ``recombination-mode = off`` can be used to debug/compare against a search without hypothesis recombination,
  but is normally left at its default (``on``) since recombination is "free" (it never removes the best
  hypothesis for a given state) and reduces the effective beam width needed for a given accuracy.
* When the log shows ``Number of label scorers (...) exceeds/less than number of configured max beam sizes``,
  the number of whitespace-separated values in ``max-beam-size``/``score-threshold`` does not match
  ``num-label-scorers`` -- see :ref:`Multiple label scorers and per-stage parameters`.
* ``recognizer-v2`` logs ``flf-recognizer-time`` and ``flf-recognizer-rtf`` per segment, which is the quickest
  way to check whether a parameter change affected decoding speed.

See also
---------

* :doc:`architecture` -- overall build and code architecture, including the classic ``Search::SearchAlgorithm``.
* :doc:`flf_nodes` -- full list of Flf nodes, including :ref:`recognizer-v2`.
* :doc:`language_model` -- language model types and configuration.
* :doc:`common_config` -- shared acoustic model, lexicon and corpus configuration.
* :doc:`file_formats/bliss_lexicon` -- how to declare special lemmata (``blank``, ``silence``,
  ``sentence-end``, ...) referenced by the lexiconfree search algorithms.