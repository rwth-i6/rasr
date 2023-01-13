Flf-Tool
========

The FLF lattice tool runs a lattice processing network
for a Bliss :doc:`corpus <bliss_corpus>` or a sequence of batches depending
on the choosen nodes.
The network has to be provided as RWTH ASR :doc:`../configuration` file,
i.e. the use of the option ``--config`` is mandatory for
running a network.


Usage
-----

default options:

| ``--help``   : help
| ``--config`` : configuration file

The following command line commands are supported:

| (none)        : runs the network
| ``init``      : initializes the network, but does not run it
| ``parse``     : parses the network, but neither initializes nor runs it
| ``help``      : gives some general help on how setting up a network
| ``help list`` : lists all nodes; help is available for each node
| ``help NODE`` : gives help to the specific node


Structure of a network
----------------------

::

    [*.network]
    initial-nodes    = INITIAL_NODE ...
    
    [*.network.INITIAL_NODE]
    type             = TYPE
    links            = PORT->TARGET_NODE:PORT ...
    ...
    
    [*.network.TARGET_NODE]
    type             = TYPE
    ...


Common initial nodes are ``speech-segment`` and ``batch``, common final node is ``sink``.
The links syntax can be shorten. The header ``PORT->`` can be skipped; ``source-port``
is set to 0. In addition or alternatively the tail ``:PORT`` can be skipped;
``target-port`` is set to 0.

Further Reading
---------------

* :doc:`Flf Nodes <flf_nodes>`
