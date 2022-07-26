CART Training
=============

**Description**

Estimation of the phonetic decision tree.
The current implementation optimizes the log-likelihood, where the log-likelihood for a class(or cluster) is modeled by a single Gaussian with diagonal co-variance matrix.

**Inputs**
* :doc:`file_formats/cart_accumulators.rst`
* :doc:`file_formats/cart_question_file.rst`
* :ref:`HMM configuration`
* :ref:`Lexicon configuration`

**Output**
* CART tree

**Parameters**

example-file (string):
    file name with accumulated examples
training-file (string):
    question file / training plan
decision-tree-file (string):
    decision tree file to write
cluster-file (string):
    write clusters to file
log-likelihood-gain.variance-clipping (float):
    minimum variance

**Tool**
:ref:`Acoustic Model Trainer`
action: estimate-cart
:ref:`CART Trainer`
:ref:`CART Viewer`

**Example Configuration**

.. code-block :: ini

    include shared.config
    
    [*]
    action                          = estimate-cart
    
    [*.cart-trainer]
    log-likelihood-gain.variance-clipping   = 5e-6
    training-file                           = config/cart-questions.xml
    example-file                            = data/cart.1.sum
    decision-tree-file                      = data/cart.1.tree
    cluster-file                            = data/cart.1.cluster # optional
    
    # ---------------------------------------------------------------------------
    
    [*.acoustic-model-trainer.channels]
    output-channel.file             = log/cart-estimate.log
    output-channel.append           = false
    output-channel.unbuffered       = true
    output-channel.compressed       = false
    output-channel.encoding         = UTF-8
    
    # ---------------------------------------------------------------------------
    
    [*]
    statistics.channel              = output-channel                    
    system-info.channel             = output-channel
    configuration.channel           = output-channel
    log.channel                     = output-channel
    progress.channel                = output-channel
    warning.channel                 = output-channel, stderr
    error.channel                   = output-channel, stderr

**References**

* K. Beulen. `Phonetische Entscheidungsbäume für die automatische Spracherkennung mit großem Vokabular <http://www-i6.informatik.rwth-aachen.de/publications/download/262/BeulenK.--PhonetischeEntscheidungsb%7Ba%7Dumef%7Bu%7DrdieautomatischeSpracherkennungmitgro%7Bss%7DemVokabular--1999.pdf >`_. PhD Thesis, Aachen, Germany, July 1999.

