Configuration
=============

All software components set the values of their configurable parameters using a global configuration instance. This global configuration keeps a set of configuration resources.

Resources
---------

A configuration resource is a key/value pair. The key part of the resource has the form 
``<selector1>.<selector2>. ... .<selectorN>``. Any of the selectors may be the wildcard ``*``. The ``*`` can replace also a sequence of selectors.

Selections are created according to the object hierarchy of the software system
(e.g. ``speech-recognizer.acoustic-model.hmm`` is the selection corresponding to the object describing 
the HMM structure used by the recognizer). The first selector is always the name of the executable.

The request for the value of a parameter ``<name>`` w.r.t. to a given selection ``<selector1>.<selector2>. ... .<selectorM>``
is resolved as follows:
* The selection ``<selector1>.<selector2>. ... .<selectorM>.<name>`` is matched against the key part of each resource.
* If matching resources exist, the value of the best matching resource is returned.

**Example** ::

    Resources:
    *.hmm.number-of-states = 3
    *.hmm.*.exit = 43.34
    *.phoneme-look-ahead.*.hmm.number-of-states = 1

    Requests:
    speech-recognizer.acoustic-model.hmm.number-of-states -> 3
    speech-recognizer.acoustic-model.hmm.silence.exit -> 43.34
    speech-recognizer.phoneme-look-ahead.acoustic-model.hmm.number-of-states -> 1
    speech-recognizer.phoneme-look-ahead.acoustic-model.hmm.silence.exit -> 43.34


Configuration Input
-------------------

Configuration resources are set up from:

* configuration files
* command line arguments of the form ``--selectors.name=value``, e.g. ``--*.output-channel.file=myfile.log``
* environment variables

References
----------

The value of a resource may contain a reference of the form ``$(selector)``. When this resource is looked up (i.e. matched against a parameter specification), the reference is textually replaced by its resolved value. The resolved value is determined by appending the resources selector to the matched parameter specification and searcing for a matching resource. This implies that resources are context dependent. If this fails the matched parameter specification is truncated until either a match is found or the resolution fails.

**Example**

Given the resources::

    *.foo.abc = cat
    *.foo.xyz = dog 
    foo.*.bar = /tmp/$(foo).txt

then ``foo.abc.bar`` is resolved to ``/tmp/cat.txt`` and ``foo.xyz.bar`` is resolved to ``/tmp/dog.txt``


Configuration Files
-------------------

Configuration files are plain text files. The recommended file name extension is ``.config``. Everything after the comment character ``#`` is ignored.

In configuration files you can group resources as follows:

    [<selector-1>.<selector-2>. ... .<selector-N>]
    name-1 = value-1
    name-2 = value-2

Configuration files can include other configuration files by a ``include`` directive::

    include filename.config

The referenced file is included at exactly this position. If the included file does not contain a group selector, the preceding group selector is used.

**Example**
::

    [*.output-channel]
    file   = logfile.log
    append = true
    # equivalent to
    # *.output-channel.file = logfile.log 
    # *.output-channel.append = true
    
    [*.acoustic-model.hmm]
    include shared-hmm-settings.config   # include some shared parameters
    silence.loop = 0.0                   # set specific value

Arithmetic Expressions
----------------------

Arithmetic Expressions of the form ``$[expression,format]`` are resolved to their value.

``expression`` may be any valid arithmetic expression, including standard math functions.
``format`` specifies if the result should contain decimals, <float> or <int>

The expression may contain references to resources.

**Example**

::

    lm-scale            = 13.34
    reciprocal-lm-scale = $[1 / $(lm-scale)]
    
    # a more theoretical example:
    cmp-2.value = 45.89
    val         = 3
    *.foo = $[ $(cmp-$[ $(val) - 1, int]).value * log(6.23) ]

Pitfalls
--------

Cmdline
"""""""

Note that there is a difference between specifying variables by cmdline ::

    <tool> tool.config --MY_VAR1=<value> --*.MY_VAR2=<value>

The first variable can only be used as a value ``$(MY_VAR1)`` within the config files whereas the second variable ``$(MY_VAR2)`` could also be used within flow files, as it is specified as a node. Note also that if one would have already specified ``MY_VAR1`` in ``tool.config`` the config parameter would not have been overwritten by the cmdline value ``--MY_VAR1``, whereas ``MY_VAR2`` in ``tool.config`` whould have been overwritten by ``--*.MY_VAR2``

Helpful for debugging such matching problems can be::

    --log-configuration=yes --log-resolved-resources=yes

Config Filenames
""""""""""""""""
You cannot use the char ``.`` to separate a config filename, e.g. ``my.funky.settings.config``  as it is used as a special ``resource_separation_char`` in the configuartion mechanism. Otherwise you might wonder about the following exit error message: 

    PROGRAM DEFECTIVE:
    precondition add_selection.find(resource_separation_char) == std::string::npos violated
    in Core::Configuration::Configuration(const Core::Configuration&, const std::string&) file Configuration.cc 


Code
----

Internally the configuration is represented by ``Core::Configuration`` objects and usually only a single one is created (as part of the Application class).

.. doxygenclass:: Core::Configuration
