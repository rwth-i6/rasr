Allophone Tool
==============

The Allophone-Tool provides information about the allophones used by the acoustic model,
including allophone-state to mixture index mappings and the allophone properties supported by CART.

**Parameters**:

| ``lexicon``        : see :ref:`Lexicon Configuration`
| ``acoustic model`` : HMM setup, see :ref:`Acoustic Model Configuration`
| ``decision tree``  : optional, see :doc:`cart_estimation` and :ref:`CART Trainer`

**Channels and Files**:

| ``allophones.store-to-file``  : store the complete allophone state list
| ``dump-allophone-properties`` : channel, store allophone properties, see [[CART question file]]
| ``dump-allophones``           : channel, all allphones
| ``dump-allophone-states``     : channel, all allophone states
| ``dump-state-tying``          : channel, allophone state tying

**Example Configuration** ::

    [*]
    load-state-tying                  = true
    dump-allophone-properties.channel = $(ALLOPHONE_PROPERTY_MAP)
    
    [*.lexicon]
    file                              = $(LEXICON)
    
    [*.acoustic-model.hmm]
    states-per-phone                  = 3
    
    # mandatory, if load-state-tying=true
    [*.state-tying]
    type                              = decision-tree 
    file                              = data/cart.1.tree
    dump-state-tying.channel          = $(STATE_TYING)
    
    [*.allophones]
    add-from-lexicon                  = true
    add-all                           = false
    store-to-file                     = $(ALLOPHONE_LIST)

