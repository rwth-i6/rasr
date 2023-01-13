LatticeProcessor
================

The lattice processor is a tool to manipulate lattices and is needed for discriminative training (see e.g. :ref:`MPE Training`).

General Description
-------------------

The lattice processor builds a chain of processor nodes described by the list in the 'lattice-processor' configuration part. E.g.::

    [lattice-processor]
    actions                 = read,write
    selections              = lattice-reader,lattice-writer

The lattice processor provides several actions, which are described in more detail below. The names of the selection can be chosen by the user and define the :ref:`configuration selection <Configuration>` of the corresponding action. In the example above this could look like::

    [*.lattice-writer]
    *.lattice-archive.path  = data/output/

After the initialization of the processor nodes the processor walks throw the corpus described in the corpus configuation part. E.g.::

    [*.corpus]
    file                            = data/corpus.gz
    warn-about-unexpected-elements  = no
    capitalize-transcriptions       = no

To interprete the corpus it needs a lexicon definition given in the lexicon configuration part. E.g.::

    [*.lexicon]
    file                            = data/lexicon.gz

The output is defined in the general ('*') section of the configuration::

    [*]
    statistics.channel              = nil
    log.channel                     = output-channel
    warning.channel                 = output-channel
    error.channel                   = output-channel
    configuration.channel           = output-channel
    system-info.channel             = output-channel
    dot.channel                     = nil
    progress.channel                = output-channel
    output-channel.file             = output-channel
    output-channel.append           = false
    output-channel.encoding         = UTF-8
    output-channel.unbuffered       = false
    on-error                        = delayed-exit

Using this configuration a file output-channel will be created during a run of the lattice-processor. These files contain lots of additional information, warnings and errors.

Available actions
-----------------

**read**:

reads a lattice archive, example configuration::

    [*.lattice-reader]
    readers                         = total
    lattice-archive.path            = data/input/
    lattice-archive.type            = {fsa|htk}

The value of the option readers is a substring of the input lattice. For example "total" may correspond to an archive containing lattices like NameOfCorpus/NameOfRecording/NameOfSegment-total.binfsa.gz.

**write**:

writes a lattice archive, example configuration::

    [*.lattice-writer]
    lattice-archive.path            = data/output/
    type                            = {fsa|htk}

**numerator-from-denominator**:

extracts the numerator (i.e. the reference) from denominator lattices in discriminative training, no configuration is needed

**single-best**:

extracts the best path from the lattice, no configuration is needed

**evaluate**:

calculate word error rate, output can be used with analog, input lattice must have a single part, no configuration parameters

**accumulate-discriminatively**:

implements lattice-based discriminative training

general settings ::

    [*.trainer]
    # type of application
    application         = [speech|tagging]

    # criterion of discriminative acoustic model trainer
    criterion           = [MMI|MCE|ME|weighted-MMI|weighted-ME|ME-with-i-smoothing|weighted-ME-with-i-smoothing|...]

    # type of model
    model-type          = [gaussian-mixture|maximum-entropy]

    # port name for features to accumulate
    port-name           = features

    # stream index of features to accumulate
    accumulation-stream-index = 0

    # discard all observations with absolute weight smaller or equal to this threshold
    weight-threshold    = Core::Type<f32>::epsilon

    # tolerance in posterior computation, i.e., error of forward and backward flows w.r.t. least significant bits
    posterior-tolerance = 100

    # name of lattice with total scores
    lattice-name        = total

acoustic model training ::

    # port name for features to accumulate
    port-name            = features

    # stream index of features to accumulate
    accumulation-stream-index = 0

    # only for ME: name of lattice with accuracies
    accuracy-name        = accuracy

log-linear acoustic model training ::

    # emission features are accumulated
    accumulate-emissions  = true

    # transition features are accumulated
    accumulate-tdp        = false

**linear-combination**:

calculate linear combination(s) of parts in lattice based on the passed scaling factors (e.g. scaled acoustic + lm score), two variants are supported

output lattice with single part "total" ::

    [*.linear-combination]
    scales           = 1.0 0.0

output lattice with multiple parts ::

    [*.linear-combination]
    outputs          = total accuracy
    total.scales     = 1.0 0.0
    accuracy.scales  = 0.0 1.0

