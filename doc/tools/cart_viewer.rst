CART Viewer
===========

The CART-Viewer dumps a CART(aka decision tree) in XML-, text-, or DOT- format and allows the interactive browsing of the CART. 

**Parameters**:

| ``decision-tree-file`` : see :ref:`CART Estimation` and :ref:`CART Trainer`

**Channels**:

| ``decision-tree.xml``   : dump the decision tree in XML-format 
| ``decision-tree.plain`` : dump the decision tree in text-format 
| ``decision-tree.dot``   : dump the decision tree in DOT-format 

**Interactive CART browsing**:

An input mask asks the user to define values for the properties used by the decision tree. The value of a property can be left open (empty input).
The viewer can be run in two modes: classification and multification. In classification mode a single final class is found and the path trough the tree, i.e. the questions asked, are printed. In case of a question addressed to an undefined property, the answer NO is assumed.
In multification mode all possible classes are printed. That is, in case of a question addressed to an undefined property both paths, i.e. the YES- and the NO-path, are followed.

**Example Configuration** ::

    [*]
    # interactive, browse the cart
    interactive                     = false
    
    # dump decision tree
    decision-tree-file              = data/cart.1.tree
    decision-tree.xml.channel       = cart.xml
    decision-tree.plain.channel     = cart.txt
    decision-tree.dot.channel       = cart.dot
    
    # Channels
    [*]
    log.channel                     = log-channel
    warning.channel                 = log-channel, stderr
    error.channel                   = log-channel, stderr
    statistics.channel              = log-channel
    configuration.channel           = log-channel
    unbuffered                      = true
    log-channel.file                = $(log-file)

