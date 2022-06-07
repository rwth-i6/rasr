Language Model
==============

Common parameters:

type (enum):
    type of the used language model. See feature scorer types below.
scale (float):
    scaling exponent for language model probabilities

Zerogram LM
-----------

language model type: ``zerogram``

Uses equal probability for all words.

n-gram LM
---------

Language model in `ARPA <http://www.speech.sri.com/projects/srilm/manpages/ngram-format.5.html>`_ format.
(aka count LM or ARPA LM)

language model type: ``ARPA``

**Configuration**

file (string):
    filename of the language model to load
image (string):
    load the language model from a binary file instead of rebuilding the datastructures in the memory. If the file does not exist, the language model is loaded, built, and written to the image file.

The LM image is a binary format that will be created from the ARPA file on the first run and loaded via mmap at run time.

**Example**

.. code-block :: ini

    [*.lm]
    type         = ARPA
    scale        = 11.5
    file         = background_arpa.lm.gz
    image        = background_arpa.lm.image

Class LM
--------

Class LMs are supported in two ways:
- using a modified / mapped lexicon: set in the lexicon the syntactic token sequence to the desired class. See the description of the :ref:`Bliss Lexicon`_ format.
- using a class language model, consisting of a word-to-class map (see ``classes.*``) and an ARPA format language model trained on the classes.

**Configuration**

language model type: ``ARPA+classes``

classes.file (string):
    word-to-class map, format "<syntactic token> <class> [<p(<syntactic token>| <class>)]"
classes.encoding (string):
    encoding of word-to-class map, see classes.file
classes.scale (float):
    scaling exponent for p(<syntactic token>| <class>), see classes.file

Weighted Grammar
----------------

Define language model by a weighted finite state automaton.

language model type: ``fsa``

See :ref:`Weighted Grammar File`

TensorFlow RNN LM
-----------------

This LM type allows to use a TF graph to calculate word probabilities e.g. with an RNN. The core configuration looks like this

The mapping between strings (syntactic tokens as defined in the recognition lexicon) and the outputs of the softmax layer is defined in a ``vocab.txt``, for example::

    <sb> 0
    <s> 0
    </s> 0
    the 1
    to 2
    you 3
    a 4
    and 5
    i 6
    of 7
    [...]

**Example**

.. code-block :: ini

    [*.lm]
    type                            = tfrnn
    scale                           = 11.0
    vocab-file                      = vocab.txt
    vocab-unknown-word              = <unk>
    
    [*.lm.loader]
    meta-graph-file                 = rnnlm/network.042.meta
    saved-model-file                = rnnlm/network.042
    required-libraries              = NativeLstm2.so
    type                            = meta
    
    [*.lm.input-map.info-0]
    param-name                      = word
    seq-length-tensor-name          = extern_data/placeholders/delayed/delayed_dim0_size
    tensor-name                     = extern_data/placeholders/delayed/delayed
    
    [*.lm.output-map.info-0]
    param-name                      = softmax
    tensor-name                     = output/output_batch_major

Self-normalized LM without full softmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example above assumes that the TF graph has a layer called ``output`` and will evaluate its output every time the decoder polls for a new p(w|h). This can become costly, so a typical work-around is to train a *self-normalized* LM that can provide p(w|h) for an individual word w without evaluating the full softmax layer of size |V|. The corresponding config (a) enables network output compression and (b) defines a new output layer:

.. code-block :: ini

    [*.lm.nn-output-compression]
    bits-per-val                    = 16
    epsilon                         = 0.001
    type                            = fixed-quantization
    
    [*.lm.softmax-adapter]
    type                            = quantized-blas-nce-16bit
    weights-bias-epsilon            = 0.001
    
    [*.lm.output-map.info-0]
    param-name                      = softmax
    tensor-name                     = bottleneck/output_batch_major # assume "bottleneck" is the name of the last hidden layer
    
    [*.lm.output-map.info-1]
    param-name                      = weights
    tensor-name                     = output/W/read
    
    [*.lm.output-map.info-2]
    param-name                      = bias
    tensor-name                     = output/b/read

This approach will evaluate the graph only up to the outputs of the layer called "bottleneck". Then for each requested p(w|h), it will pick the w-th row from the weights and bias and compute a dot-product with the outputs of the bottleneck.

**Further reading**

A. Gerstenberger, K. Irie, P. Golik, E. Beck, and H. Ney. [https://www-i6.informatik.rwth-aachen.de/publications/download/1125/Gerstenberger-ICASSP-2020.pdf Domain Robust, Fast, and Compact Neural Language Models]. In IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), pages 7954-7958, Barcelona, Spain, May 2020.

Rescoring
---------

TODO: separate page

A typical config of the FLF node for lattice rescoring with a TF RNN LM looks e.g. like this:

.. code-block :: ini

    [*.network.rescore]
    history-limit                           = 0
    key                                     = lm
    links                                   = best
    lookahead-scale                         = 1.0
    max-hypotheses                          = 10
    pruning-threshold                       = 10.0
    rescorer-type                           = replacement-approximation
    type                                    = push-forward-rescoring

**Caveats**

* A discrepancy between vocabularies of the ARPA LM and the RNN LM will have a strong (negative) impact on the recognition accuracy.
* For debugging, enable:

.. code-block :: ini

    [*.lm]
    dump-scores        = true # warning: very large output files! test on isolated segments!
    dump-scores-prefix = /tmp/scores
    verbose            = true

When compiling the recognition.meta graph of a model trained with RETURNN, remember to:
* sync the settings of the output layer (e.g. ``'class': 'linear', 'activation' : 'log_softmax'``) with the post-processing options (``transform-output-*``).
* configure the recurrent layers as ``"unit": "nativelstm2", "initial_state" : "keep_over_epoch_no_init"`` .

TensorFlow Transformer LM
-------------------------

TODO

.. code-block :: ini

    [*.lm]
    type                    = tfrnn
    state-manager.type      = transformer

**Further reading**

* E. Beck, R. Schlüter, and H. Ney. [https://www-i6.informatik.rwth-aachen.de/publications/download/1158/Beck--2020.pdf LVCSR with Transformer Language Models]. In Interspeech, pages 1798-1802, Shanghai, China, October 2020.

Combine LM
----------

This class provides means to perform dynamic LM interpolation (i.e. at run time during decoding). The basic configuration principle is to specify how many LMs you need (``num-lms``) and provide configuration for all of ``lm-$x`` (where ``$x`` starts with one). This enables interpolation between any number (and type) of LMs.

**Example**

.. code-block :: ini

    [*.lm]
    type         = combine
    scale        = 11.5
    num-lms      = 2
    lookahead-lm = 1
    
    
    [*.lm.lm-1]
    type         = ARPA
    file         = background_arpa.lm.gz
    scale        = 0.9
     
    
    [*.lm.lm-2]
    type         = ARPA
    file         = domain_arpa.lm.gz
    scale        = 0.1

Please note that the ``*.lm.scale`` refers to the global LM scale while the component LMs use relative scales (that should sum up to one).

**Caveats**

Linear vs Log-Linear interpolation:
This can be switched by setting the boolean parameter ``linear-combination`` that defaults to false.

Lookahead LM:
It is better to use a single (backing off) LM for lookahead purposes by specifying it explicitly either via a separate ``[*.lookahead-lm]`` or by selecting one of the models used in combination: ``lookahead-lm = 1``. A somewhat hidden consequence is that the lookahead scores are going to be scaled with the *relative* LM scale, so it's better to set it explicitly to the ''absolute'' LM scale via


.. code-block :: ini

    [*.lm-lookahead]
    lm-lookahead-scale = $(lm.scale)

Handling of OOVs:
It is common that the component LMs have (slightly) different vocabularies, and some words are OOV. By default, the backing off LMs assign OOVs a probability of zero (i.e. inf score) which prevents the word to be picked even if another component LM has a high probability for it. This can be changed by setting ``*.lm-1.map-oov-to-unk = true``, so that an OOV word will be treated as the syntactic token of the unknown lemma.

