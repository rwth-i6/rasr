CART Trainer
============

A CART trainer that uses the ID3 algorithm. For estimating a phonetic decision tree see :ref:`CART Estimation`.
The CART trainer can also be used to verify and dump a CART training setup, in particular a :ref:`CART question file` and a :ref:`CART example file`.

**Input**

* CART Examples
* :ref:`CART question file`

**Output**

* CART tree
* Cluster file

**Parameters**:

| ``dry (bool)``                  : if yes, only check and dump training/question file and example file
| ``scoring-function (string)``   : scoring function for clusters of observations
| ``training-file (string)``      : question file / training plan
| ``example-file (string)``       : file name with examples
| ``decision-tree-file (string)`` : decision tree file to write
| ``cluster-file (string)``       : write clusters to file

**Channels**:

| ``training.xml``         : dump the CART questions in XML-format
| ``training.plain``       : dump the CART questions in text-format
| ``example.xml``          : dump the training examples in XML-format
| ``example.plain``        : dump the training examples in text-format
| ``decision-tree.xml``    : dump the decision tree in XML-format
| ``decision-tree.plain``  : dump the decision tree in text-format
| ``decision-tree.dot``    : dump the decision tree in DOT-format
| ``cluster.xml``          : dump the clustered training examples in XML-format
| ``cluster.plain``        : dump the clustered training examples in text-format

**Example Configuration** ::

    [*]
    # dry-run
    dry                             = false
    
    # scoring; so far only ID3 is supported
    scoring-function                = ID3
    
    # training plan
    training-file                   = config/cart-questions.xml
    # dump training setup
    training.plain.channel          = cart-questions.txt
    training.xml.channel            = cart-questions.xml
    
    # examples
    example-file                    = data/cart-examples.xml
    # dump examples
    example.plain.channel           = cart-examples.txt
    example.xml.channel             = cart-examples.xml
    
    # decision tree
    decision-tree-file              = data/cart-tree.xml
    decision-tree.plain.channel     = cart-tree.txt
    decision-tree.xml.channel       = cart-tree.xml
    decision-tree.dot.channel       = cart-tree.dot
    
    # clusters
    cluster-file                    = data/cart-cluster.xml # optional
    cluster.plain.channel           = cart-cluster.txt
    cluster.xml.channel             = cart-cluster.xml
    
    # Channels
    [*]
    log.channel                     = log-channel
    warning.channel                 = log-channel, stderr
    error.channel                   = log-channel, stderr
    statistics.channel              = log-channel
    configuration.channel           = log-channel
    unbuffered                      = true
    log-channel.file                = $(log-file)
