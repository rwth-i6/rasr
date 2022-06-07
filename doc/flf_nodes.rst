Flf Nodes
=========

Flf nodes are the processing units used by the :ref:`Flf-Tool`.

CN-archive-reader
-----------------

Read CNs from archive;
the CN is buffered for multiple access.

**Configuration**


.. code-block: ini

    [*.network.CN-archive-reader]
    type                        = CN-archive-reader
    format                      = xml
    path                        = <archive-path>
    suffix                      = .<format>.cn.gz
    encoding                    = utf-8

**Port assignment**

.. code-block: ini

    input:
    1:segment | 2:string
    output:
    0:CN


CN-archive-writer
-----------------

Store CNs in archive

**Configuration**


.. code-block: ini

    [*.network.cn-archive-writer]
    type                        = CN-archive-writer
    format                      = text|xml*
    path                        = <archive-path>
    archive.suffix              = .<format>.cn.gz
    archive.encoding            = utf-8

**Port assignment**

.. code-block: ini

    input:
    0:CN, 1:segment | 2:string
    output:
    0:CN
    


CN-combination
--------------

Combine and decode incoming posterior CNs

**Configuration**


.. code-block: ini

    [*.network.CN-combination]
    type                        = CN-combination
    cost                        = expected-loss|expected-error*
    posterior-key               = confidence
    score-combination.type      = discard|*concatenate
    beam-width                  = 100
    cn-0.weight                 = 1.0
    cn-0.posterior-key          = <unset>
    ...

**Port assignment**

.. code-block: ini

    input:
    0:normalized-CN [1:normalized-CN [...]]
    output:
    0:top-best-lattice 1:normalized-CN 2:normalized-CN-lattice
    


CN-decoder
----------

Decode incoming CN, where the CN is provided at port 0 or
alternatively a lattice with sausage topology at port 1.
The posterior key defines the dimension of the semiring which
provides a word-wise probability distribution per slot and
is to be used for slot-wise decoding.

**Configuration**


.. code-block: ini

    [*.network.CN-decoder]
    type                        = CN-decoder
    posterior-key               = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:CN | 1:sausage-lattice
    output:
    0:best-lattice 1:sausage-lattice
    


CN-features
-----------

*WARNING: beta status*
Per arc, set the value for a feature derived from the CN to
the corresponding dimension.
Features:
* confidence:    slot based confidence
* score:         negative logarithm of confidence
* cost:          oracle alignment based cost;
0, if oracle label equals arc label, 1, else
* oracle-output: store oracle alignment as output label
* entropy:       entropy of normalized slot
* slot:          number of the slot the lattice arc falls into
* non-eps-slot:  Same as "slot", but slots containing only epsilon arcs
are ignored; epsilon arcs do not get this feature.
If the threshold is < 1.0, then all slots with an
epsilon mass >= threshold are ignored; the input of
lattice arcs pointing at these slots are set to epsilon.
Attention: confidence, score, and entropy feature require the
defintion of "cn.posterior-key".

**Configuration**


.. code-block: ini

    [*.network.CN-features]
    type                        = CN-features
    compose                     = false
    duplicate-output            = false
    # features
    confidence.key              = <unset>
    score.key                   = <unset>
    cost.key                    = <unset>
    oracle-output               = false
    entropy.key                 = <unset>
    slot.key                    = <unset>
    non-eps-slot.key            = <unset>
    non-eps-slot.threshold      = 1.0
    [*.network.CN-features.cn]
    posterior-key               = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice 1:CN
    output:
    0:lattice
    


CN-gamma-correction
-------------------

Perform a in-situ gamma correction of the slot-wise
posterior probability distribution.
The CN must be normalized.

**Configuration**


.. code-block: ini

    [*.network.CN-gamma-correction]
    type                        = CN-gamma-correction
    gamma                       = 1.0
    normalize                   = true

**Port assignment**

.. code-block: ini

    input:
    0:CN(normalized)
    output:
    0:CN
    


FB-builder
----------

Build Fwd./Bwd. scores from incoming lattice(s).
In the case of multiple incoming lattices, the result
is the union of all incoming lattices.
There are some major differences between doing single or multiple
lattice FB:
Single lattice:
The semiring from the incoming lattice is preserved;
all dimensions used need to be present in this semiring,
e.g. score.key.
The topology of the incoming and outgoing lattice are equal.
Risk calculation is available.
Multiple lattices:
Union of all lattices are build.
The union lattice has a new semiring consisting either of
score.key only, or of the concatenated scores of the incoming lattices.
Optionally, a label identifying the source system is set as the output label
in the union lattice.

**Configuration**


.. code-block: ini

    [*.network.FB-builder]
    type                        = FB-builder
    [*.network.FB-builder.multi-lattice-algorithm]
    force                       = false
    [*.network.FB-builder.fb]
    configuration.channel       = nil
    statistics.channel          = nil
    
    # single lattice FB
    score.key                   = <unset>
    risk.key                    = <unset>
    risk.normalize              = false
    cost.key                    = <unset> # required, if risk.key is specified
    # Default alpha is 1/<max scale>; alpha is ignored, if a
    # semiring is given (see below).
    alpha                       = <unset>
    # If a semiring is specified, then the number of dimensions
    # of old and new semiring must be equal.
    semiring.type               = <unset>*|tropical|log
    semiring.tolerance          = <default-tolerance>
    semiring.keys               = key1 key2 ...
    semiring.key1.scale         = <f32>
    semiring.key2.scale         = <f32>
    ...
    
    # multiple lattice FB
    score-combination.type      = discard|*concatenate
    score.key                   = <unset>
    system-labels               = false
    set-posterior-semiring      = false
    [*.network.FB-builder.fb.lattice-0]
    weight                      = 1.0
    # Default alpha is 1/<max scale>; alpha is ignored, if a
    # semiring is given (see below).
    alpha                       = <unset>
    # If a semiring is specified, then the number of dimensions
    # of old and new semiring must be equal.
    semiring.type               = <unset>*|tropical|log
    semiring.tolerance          = <default-tolerance>
    semiring.keys               = key1 key2 ...
    semiring.key1.scale         = <f32>
    semiring.key2.scale         = <f32>
    ...
    label                       = system-0
    # experimental
    norm.key                    = <unset>
    norm.fsa                    = false
    weight.key                  = <unset>
    [*.network.FB-builder.fb.lattice-1]
    ...
    

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    0:lattice 1:FwdBwd
    


ROVER-combination
-----------------

Combine and decode incoming lattices

**Configuration**


.. code-block: ini

    [*.network.ROVER-combination]
    type                        = ROVER-combination
    cost                        = sclite-word-cost|*sclite-time-mediated-cost
    null-word                   = @
    null-confidence             = 0.7
    alpha                       = 0.0
    posterior-key               = confidence
    score-combination.type      = discard|*concatenate
    beam-width                  = 100
    lattice-0.weight            = 1.0
    lattice-0.confidence-key    = <unset>
    ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    0:top-best-lattice 1:normalized-CN 2:normalized-CN-lattice 3:n-best-CN 4:n-best-CN-lattice
    


add
---

Manipulate a single dimension:
f(x_d) = x_d + <score>

**Configuration**


.. code-block: ini

    [*.network.add]
    type                        = add
    append                      = false
    key                         = <symbolic key or dim>
    score                       = 0.0
    rescore-mode                = {clone*, in-place-cached, in-place}

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


add-word-confidence
-------------------

*DEPRECATED: see "fcn-confidence" and/or "fcn-features*

**Configuration**


.. code-block: ini

    [*.network.add-word-confidence]
    type                        = add-word-confidence
    ... see fCN-confidence

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:fCN]
    output:
    0:lattice
    


aligner
-------

Align a linear hypothesis against a reference lattice or
a reference fCN.
The algorithm works as follows:
# try intersection with reference lattice,
if intersection is empty then
# align against reference fCN
If a reference fCN is required and a connection at port 1 exist,
the reference fCN is taken from port 1, else the fCN is
calculated from the lattice at port 2.
If intersection is false, step 1) is skipped.

**Configuration**


.. code-block: ini

    [*.network.aligner]
    type                        = aligner
    intersection                = true
    [*.network.aligner.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    0:hypothesis-lattice {1:reference-fCN | 2:reference-lattice}
    output:
    0:aligned-lattice
    


append
------

Append two lattices score-wise;
both lattices must have equal topology
(down to state numbering).
The resulting lattice has a semiring consisting
of the concatenatation
of the two incoming semirings.

**Configuration**


.. code-block: ini

    [*.network.append]
    type                        = append

**Port assignment**

.. code-block: ini

    input:
    0:lattice 1:lattice
    output:
    0:lattice
    


approximated-risk-scorer
------------------------

*DEPRECATED: see "local-cost-decoder*<br/>
**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    0:lattice(best) 1:lattice(rescored)
    


archive-reader
--------------

Read lattices from archive;
the lattice is buffered for multiple access.

**Configuration**


.. code-block: ini

    [*.network.archive-reader]
    type                        = archive-reader
    format                      = flf|htk
    path                        = <archive-path>
    info                        = false
    # if format is flf
    [*.network.archive-reader.flf]
    suffix                      = .flf.gz
    [*.network.archive-reader.flf.partial]
    keys                        = key1 key2 ...
    [*.network.archive-reader.flf.append]
    keys                        = key1 key2 ...
    key1.scale                  = 1.0
    key2.scale                  = 1.0
    ...
    # if format is htk
    [*.network.archive-reader.htk]
    suffix                      = .lat.gz
    fps                         = 100
    encoding                    = utf-8
    slf-type                    = forward|backward
    capitalize                  = false
    word-penalty                = <f32>
    silence-penalty             = <f32>
    merge-penalties             = false
    set-coarticulation          = false
    eps-symbol                  = !NULL
    # archive specific options
    [*.network.archive-reader.*.semiring]
    type                        = tropical|log
    tolerance                   = <default-tolerance>
    keys                        = key1 key2 ...
    key1.scale                  = <f32>
    key2.scale                  = <f32>
    ...
    # if format is flf AND semiring is specified
    [*.network.archive-reader.flf]
    input-alphabet.name         = {lemma-pronunciation*|lemma|syntax|evaluation}
    input-alphabet.format       = bin
    input-alphabet.file         = <alphabet-file>
    output-alphabet.name        = {lemma-pronunciation*|lemma|syntax|evaluation}
    output-alphabet.format      = bin
    output-alphabet.file        = <alphabet-file>
    boundaries.suffix           = <boundaries-file-suffix>
    key1.format                 = bin
    key1.suffix                 = <fsa-file-suffix>
    ...

**Port assignment**

.. code-block: ini

    input:
    1:segment | 2:string
    output:
    0:lattice
    


archive-writer
--------------

Store lattices in archive

**Configuration**


.. code-block: ini

    [*.network.archive-writer]
    type                        = archive-writer
    format                      = flf|htk|lattice-processor
    path                        = <archive-path>
    info                        = false
    # if format is flf
    [*.network.archive-writer.flf]
    suffix                      = .flf.gz
    input-alphabet.format       = bin
    input-alphabet.file         = bin:input-alphabet.binfsa.gz
    output-alphabet.format      = bin
    output-alphabet.file        = bin:output-alphabet.binfsa.gz
    alphabets.format            = 
    alphabets.file              = 
    [*.network.archive-writer.flf.partial]
    keys                        = key1 key2 ...
    add                         = false
    # if format is htk
    [*.network.archive-writer.htk]
    suffix                      = .lat.gz
    fps                         = 100
    encoding                    = utf-8
    # if format is htk
    [*.network.archive-writer.lattice-processor]
    pronunciation-scale         = <required>

**Port assignment**

.. code-block: ini

    input:
    0:lattice, 1:segment | 2:string
    output:
    0:lattice
    


batch
-----

Read argument list(s) either from command line or from file;
in the case of a file, every line is interpreted as an argument list.
Argument number x is accessed via port x.


**Configuration**


.. code-block: ini

    [*.network.batch]
    type                        = batch
    file			     = <batch-list>
    encoding                    = utf-8

**Port assignment**

.. code-block: ini

    no input
    output:
    x: argument[x]
    


best
----

Find the best path in a lattice.
Usually, Dijkstra is faster than Bellman-Ford,
but Dijkstra does not guarantee correct results in the
presence of negative arc scores.

**Configuration**


.. code-block: ini

    [*.network.best]
    type                        = best
    algorithm                   = dijkstra*|bellman-ford|projecting-bellman-ford

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


buffer
------

Incoming lattice is buffered until next sync and
manifolded to all outgoing ports.

**Configuration**


.. code-block: ini

    [*.network.buffer]
    type                        = buffer

**Port assignment**

.. code-block: ini

    input:
    x:lattice (at exactly one port)
    output:
    x:lattice
    


cache
-----

State requests to incoming lattice are cached;
see Fsa for details.

**Configuration**


.. code-block: ini

    [*.network.cache]
    type                        = cache
    max-age                     = 10000

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


center-frame-CN-builder
-----------------------

Build CN from incoming lattice(s).
The algorithm is based on finding an example or prototype frame for each word.

**Configuration**


.. code-block: ini

    [*.network.center-frame-CN-builder]
    type                        = frame-CN-builder
    statistics.channel          = nil
    confidence-key              = <unset>
    map                         = false
    [*.network.center-frame-CN-builder.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    output:
    0:lattice(best)
    1:CN(normalized)   2:lattice(normalized CN)
    3:CN               4:lattice(CN)
    5:fCN              6:lattice(union)
    


change-semiring
---------------

Replace the semiring.
The target semiring might have a different dimensionality;
mapping from the old to the new semiring is done via keys, i.e.
the names of the dimensions.
The operation does not affect the scores.

**Configuration**


.. code-block: ini

    [*.network.change-semiring]
    type                        = change-semiring
    [*.network.change-semiring.semiring]
    type                        = tropical|log
    tolerance                   = <default-tolerance>
    keys                        = key1 key2 ...
    key1.scale                  = <f32>
    key2.scale                  = <f32>
    ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


clean-up
--------

Clean up lattice. Arcs that
* close a cycle
* have an invalid label id
* have an invalid or semiring-zero score in at least one dimension
are discarded and the lattice is trimmed.
Thus, the resulting lattice is guaranteed to be
acyclic, trim, and zero-sum free.

**Configuration**


.. code-block: ini

    [*.network.clean-up]
    type                        = clean-up

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


cluster-CN-builder
------------------

*DEPRECATED: see "state-cluster-CN-builder*

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    0:lattice(best)
    1:CN(normalized)   2:lattice(normalized CN)
    3:CN               4:lattice(CN)
    6:lattice(state cluster)
    


compose
-------

see compose-matchin

**Configuration**


.. code-block: ini

    [*.network.compose]
    type                        = compose

**Port assignment**

.. code-block: ini

    see compose-matchin
    


compose-matching
----------------

Compose two lattices; for algorithm details see FSA.
If the left lattice is unweighted, then its weights are
set to semiring one (of the semiring of the right lattice)
and its word boundaries are invalidated.

**Configuration**


.. code-block: ini

    [*.network.compose-matching]
    type                        = compose-matching
    unweight-left               = false
    unweight-right              = false

**Port assignment**

.. code-block: ini

    input:
    0:lattice, 1:lattice
    output:
    0:lattice
    


compose-sequencing
------------------

Compose two lattices; for algorithm details see FSA

**Configuration**


.. code-block: ini

    [*.network.compose-sequencing]
    type                        = compose-sequencing

**Port assignment**

.. code-block: ini

    input:
    0:lattice, 1:lattice
    output:
    0:lattice
    


compose-with-fsa
----------------

Compose with an fsa and rescore a single lattice dimension.
Composition uses the "compose sequencing" algorithm, see FSA.

**Configuration**


.. code-block: ini

    [*.network.compose-with-fsa]
    type                        = compose-with-fsa
    append                      = false
    key                         = <symbolic key or dim>
    scale                       = 1
    rescore-mode                = clone*|in-place-cached|in-place
    # i.e. if port 1 is not connected
    file                        = <fsa-filename>
    # in case of acceptor
    alphabet.name               = {lemma-pronunciation|lemma|syntax|evaluation}
    # in case of transducer
    input-alphabet.name         = {lemma-pronunciation|lemma|syntax|evaluation}
    output-alphabet.name        = {lemma-pronunciation|lemma|syntax|evaluation}

**Port assignment**

.. code-block: ini

    input:
    0:lattice[, 1: fsa]
    output:
    0:lattice
    


compose-with-lm
---------------

Compose LM with lattice and rescore a single lattice dimension.
The "force-sentence-end=true", then each segment end is treated as a
sentence end, regardless of any arcs labeled with the sentence end symbol.


**Configuration**


.. code-block: ini

    [*.network.compose-with-lm]
    type                        = compose-with-lm
    append                      = false
    key                         = <symbolic key or dim>
    scale                       = 1
    force-sentence-end          = true
    project-input               = false
    [*.network.compose-with-lm.lm]
    (see module Lm)

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


concatenate-fCNs
----------------

Concatenate all segments corresponding to the same recording:
At port 1 a list of segments has to be provided, where each
segment defines uniquely a recording.
At port 0 a list of segments has to be provided with arbitrary
many segments per recording. The segments do not need to
partitionate the recording: gaps and overlaps are allowed.
At port 0 the concatenated fCN is provided. And at port 1
the corresponding segment, i.e. the "recording"-segment that
was provided at port 1.
Attention:
Nodes being providing segments to this node must NOT be
connected to any other node.

**Configuration**


.. code-block: ini

    [*.network.concatenate-fCNs]
    type                        = concatenate-fCNs
    dump.channel                = <unset>
    see fCN-archive-reader

**Port assignment**

.. code-block: ini

    input:
    0:segment 1:segment
    output:
    0:fCN 1:segment
    


concatenate-lattices
--------------------

Concatenate all segments corresponding to the same recording:
At port 1 a list of segments has to be provided, where each
segment defines uniquely a recording.
At port 0 a list of segments has to be provided with arbitrary
many segments per recording. The segments do not need to
partitionate the recording: gaps and overlaps are allowed.
At port 0 the concatenated lattice is provided. And at port 1
the corresponding segment, i.e. the "recording"-segment that
was provided at port 1.
Attention:
Nodes being providing segments to this node must NOT be
connected to any other node.

**Configuration**


.. code-block: ini

    [*.network.concatenate-lattices]
    type                        = concatenate-lattices
    dump.channel                = <unset>
    see archive-reader

**Port assignment**

.. code-block: ini

    input:
    0:segment 1:segment
    output:
    0:lattice 1:segment
    


copy
----

Make static copy of incoming lattice.
By default, scores are copied by reference.
Optional in-sito trimming and/or state numbering normalization
is supported.

**Configuration**


.. code-block: ini

    [*.network.copy]
    type                        = copy
    # make deep copy, i.e. copy scores by value and not by reference
    deep                        = false
    trim                        = false
    normalize                   = false

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


ctm-reader
----------

Read a single ctm-file.
CTM format is:
<name> <track> <start> <duration> <word> [<score1> [<score2> ...]]
For a given segment a linear lattice is build from the
from overlapping part.
A semiring can be specified as well as list of keys mapping
the CTM scores to the semiring dimensions.
If no keys are given, the keys from the semiring are used.
If no semiring is given, the keys are used to build a semiring.
If none is given, the empty semiring is used.
Example:
Configuration for a CTM file providing confidence scores.
scores             = confidence
confidence.default = 1.0

**Configuration**


.. code-block: ini

    [*.network.ctm-reader]
    type                        = ctm-reader
    path                        = <path>
    encoding                    = utf-8
    scores                      = key1 key2 ...
    key1.default                = <f32>
    key2.default                = <f32>
    ...
    [*.network.ctm-reader.semiring]
    type                        = tropical|log
    tolerance                   = <default-tolerance>
    keys                        = key1 key2 ...
    key1.scale                  = <f32>
    key2.scale                  = <f32>
    ...

**Port assignment**

.. code-block: ini

    input:
    1:segment
    output:
    0:lattice
    


determinize
-----------

Determinize lattice; for algorithm details see FSA

**Configuration**


.. code-block: ini

    [*.network.determinize]
    type                        = determinize
    log-semiring                = true|false*
    log-semiring.alpha          = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


difference
----------

Difference of two lattices; for algorithm details see FSA

**Configuration**


.. code-block: ini

    [*.network.difference]
    type                        = difference

**Port assignment**

.. code-block: ini

    input:
    0:lattice, 1:lattice
    output:
    0:lattice
    


drawer
------

Draw lattice(s) in dot format to file.
For filename generation see "writer".

**Configuration**


.. code-block: ini

    [*.network.drawer]
    type                        = drawer
    hints                       = {detailed best probability unscaled}
    # to draw a single lattice
    file                        = <dot-file>
    # to draw multiple files,
    # i.e. if incoming connection at port 1
    path                        = <dot-base-dir>
    prefix                      = <file-prefix>
    suffix                      = <file-suffix>

**Port assignment**

.. code-block: ini

    input:
    0:lattice[, 1:segment | 2:string]
    output:
    0:lattice
    


dummy
-----

If it gets input at port 0, it behaves like a filter,
but passes lattices just through.
Else it does nothing, ignoring any input from other ports.

**Port assignment**

.. code-block: ini

    input:
    0:lattice or no input
    output:
    0:lattice, if input at port 0
    


dump-CN
-------

Dump a textual representation of a
confusion network.
At port 0 the lattice representation of the CN is
provided. Port 1 provides the CN itself and port 2
provides an empty dummy lattice which can be connected
to a sink.

**Configuration**


.. code-block: ini

    [*.network.dump-CN]
    type                        = dump-CN
    dump.channel                = nil
    format                      = text|xml*

**Port assignment**

.. code-block: ini

    input:
    0:CN [1:segment]
    output:
    0:lattice 1:CN 2-n:dummy-lattice
    


dump-all-pairs-best
-------------------

Calculates and dumps the shortest distance between all state pairs
and dump them in plain text. The shortest distance is the minimum sum
of the projected arc scores; thus the distance is a scalar.
If time threshold is set, then only pairs of states are considered,
where the distance in time does not exceed the threshold.

**Configuration**


.. code-block: ini

    [*.network.dump-all-pairs-best]
    type                        = dump-all-pairs-best
    dump.channel                = <file>
    time-threshold              = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice[, 1:segment]
    output:
    0:lattice
    


dump-fCN
--------

Dump a textual representation of a
frame wise confusion network (or any
posterior CN).
Slots are sorted by decreasing probability.
At port 0 the lattice representation of the CN is
provided. Port 1 provides the fCN itself and port 2
provides an empty dummy lattice which can be connected
to a sink.

**Configuration**


.. code-block: ini

    [*.network.dump-CN]
    type                        = dump-CN
    dump.channel                = nil
    format                      = text|xml*

**Port assignment**

.. code-block: ini

    input:
    0:fCN [1:segment]
    output:
    0:lattice 1:fCN 2-n:dummy-lattice
    


dump-n-best
-----------

Dumps a linear or n-best-list lattice

**Configuration**


.. code-block: ini

    [*.network.dump-n-best]
    type                        = dump-n-best
    dump.channel                = <file>
    scores                      = <key-1> <key-2> ... # default is all scores

**Port assignment**

.. code-block: ini

    input:
    0:n-best-lattice[, 1:segment]
    output:
    0:n-best-lattice
    


dump-traceback
--------------

Dumps a linear lattice or an n-best list in a traceback format,
i.e. the output includes time information for each item.
For tracebacks in Bliss format, the lattice is mapped to lemma-pronunciation.
The CTM format is independent of the input alphabet; if the "dump-orthography"
option is active, the lattice is mapped to lemma.
For phoneme or subword alignments, the input alphabet must be lemma or lemma-
pronunciation and at port 1 a valid Bliss-segment is required. If an alignment
for a lemma is requested, the result is the Viterbi alignment over all matching
pronunciations.

**Configuration**


.. code-block: ini

    [*.network.dump-traceback]
    type                        = dump-traceback
    format                      = bliss|corpus|ctm*
    dump.channel                = <file>
    [*.network.dump-traceback.ctm]
    dump-orthography            = true
    dump-coarticulation         = false
    dump-non-word               = false
    dump-eps                    = <dump-non-word>
    non-word-symbol             = <unset> # use lexicon representation for non-words 
                                          # and !NULL for eps arcs; if set, then use
                                          # for non-word and for eps arcs.
    scores                      = <key-1> <key-2> ...
    dump-type                   = false
    dump-phoneme-alignment      = false
    dump-subword-alignment      = false
    subword-map.file            = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice[, 1:segment]
    output:
    0:lattice
    


dump-vocab
----------

Extracts and dumps all words occuring at least once
as input token in a lattice.

**Configuration**


.. code-block: ini

    [*.network.dump-vocab]
    type                        = dump-vocab
    dump.channel                = <file>

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


evaluator
---------

Calculate WER and/or GER

**Configuration**


.. code-block: ini

    [*.network.evaluator]
    type                        = evaluator
    single-best                 = true
    best-in-lattice             = true
    word-errors                 = true
    letter-errors               = false
    phoneme-errors              = false
    [*.network.evaluator.layer]
    use                         = true
    name                        = <node-name>
    [*.network.evaluator.edit-distance]
    format                      = bliss*|nist
    allow-broken-words          = false
    sub-cost                    = 1
    ins-cost                    = 1
    del-cost                    = 1
    #semiring used for decoding lattice
    [*.network.evaluator.semiring]
    type                        = tropical|log
    tolerance                   = <default-tolerance>
    keys                        = key1 key2 ...
    key1.scale                  = <f32>
    key2.scale                  = <f32>
    ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice, {1:segment | 2: reference string}
    output:
    0:lattice
    


exp
---

Manipulate a single dimension:
f(x_d) = exp(<scale> * x_d)

**Configuration**


.. code-block: ini

    [*.network.exp]
    type                        = exp
    append                      = false
    key                         = <symbolic key or dim>
    scale                       = 1.0
    rescore-mode                = {clone*, in-place-cached, in-place}

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


expand-transits
---------------

Modifies the lattice by expanding the transitions so that each
state corresponds to one left and right coarticuled phoneme,
or to a non-coarticulated transition.
This may be required for correct word boundary information if
the decoder doesn't produce it correctly.

**Configuration**


.. code-block: ini

    [*.network.expand-transits]
    type                        = expand-transits


**Port assignment**

.. code-block: ini

    input:
      0:lattice
    output:
      0:lattice


extend-by-penalty
-----------------

Penalize a single dimension. The penalty can be made input-label
dependent:
First, a list of class labels is defined. Second, each class label gets
a list of othographies and a penalty assigned.
Class penalties overwrites the default penalty.

**Configuration**


.. code-block: ini

    [*.network.extend-by-penalty]
    type                        = extend-by-penalty
    append                      = false
    key                         = <symbolic key or dim>
    scale                       = 1.0
    rescore-mode                = {clone*, in-place-cached, in-place}
    # default penalty
    penalty                     = 0.0
    # class dependent penalties (optional)
    [*.network.extend-by-penalty.mapping]
    classes                     = <class1> <class2> ...
    <class1>.orth               = <orth1> <orth2> ...
    <class1>.penalty            = 0.0

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


extend-by-pronunciation-score
-----------------------------

A single dimension is extended by the pronunciation score.
The pronunciation score is derived form the lexicon.

**Configuration**


.. code-block: ini

    [*.network.extend-by-pronunciation-score]
    type                        = extend-by-pronunciation-score
    append                      = false
    key                         = <symbolic key or dim>
    scale                       = 1.0
    rescore-mode                = {clone*, in-place-cached, in-place}

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


fCN-archive-reader
------------------

Read posterior CNs, i.e. normally frame-wise CNs,
from archive; the CN is buffered for multiple
access.

**Configuration**


.. code-block: ini

    [*.network.fCN-archive-reader]
    type                        = fCN-archive-reader
    format                      = xml
    # xml format
    [*.network.fCN-archive-reader.archive]
    path                        = <archive-path>
    suffix                      = .<format>.fcn.gz
    encoding                    = utf-8

**Port assignment**

.. code-block: ini

    input:
    1:segment | 2:string
    output:
    0:fCN
    


fCN-archive-writer
------------------

Store posterior CNs in archive

**Configuration**


.. code-block: ini

    [*.network.fCN-archive-writer]
    type                        = fCN-archive-writer
    format                      = text|xml*|flow-alignment
    # text|xml format
    [*.network.fCN-archive-writer.archive]
    path                        = <unset>
    suffix                      = .<format>.fcn.gz
    encoding                    = utf-8
    # flow-alignment format
    [*.network.fCN-archive-writer.flow-cache]
    path                        = <unset>
    compress                    = false
    gather                      = inf
    cast                        = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:fCN, 1:segment | 2:string
    output:
    0:fCN
    


fCN-builder
-----------

Build fCN from incoming lattice(s).
First, the union of the lattices is builde and the
weighted fwd/bwd scores of the union are calculated.
Second, from the union the fCN is derived.

**Configuration**


.. code-block: ini

    [*.network.fCN-builder]
    type                        = fCN-builder
    [*.network.fCN-builder.fb]
    see FB-builder ...
    # Pruning is applied before fwd/bwd score calculation

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    0:lattice(union) 1:fCN 2:lattice(fCN)
    


fCN-combination
---------------

Build joint fCN over all incoming fCNs
by bulding the frame and word-wise joint probability.
Optionally use the word-wise maximum approximation.

**Configuration**


.. code-block: ini

    [*.network.fCN-combination]
    type                        = fCN-combination
    weighting                   = static*|min-entropy|inverse-entropy
    fCN-0.weight                = 1.0
    ...

**Port assignment**

.. code-block: ini

    input:
    0:fCN [1:fCN [...]]
    output:
    0:lattice 1:fCN
    


fCN-confidence
--------------

Calculate word confidence using Frank Wessel's approach.
Take fCN from port 1, if provided, else build the
frame-wise fCN for the incoming lattice.

**Configuration**


.. code-block: ini

    [*.network.fCN-confidence]
    type                        = fCN-confidence
    gamma                       = 1.0
    append                      = false
    key                         = <symbolic key or dim>
    rescore-mode                = clone*|in-place-cached|in-place
    [*.network.fCN-confidence.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:fCN]
    output:
    0:lattice
    


fCN-features
------------

Take fCN from port 1, if provided, else build the
frame-wise fCN either from the lattice provided at port 2
or from the incoming lattice itself.
A gamma != 1.0 performs a slot-wise gamma-correction on the
frame-wise word posterior distributions.
Per arc, set the value for a feature derived from the fCN
to the corresponding dimension.
Features:

* confidence: Frank-Wessel's confidence scores
* error:      smoothed, expected time frame error
** alpha=0.0 -> unsmoothed error
** fCN[t]=0.0|1.0 -> (smoothed) time frame error
* Min.fWER-decoding: select the path with the lowest error

"Accuracy/Error lattices:
The calculation of arc-wise frame errors can be done by
providing the reference as a linear lattice at port 2.
Alternatively, a fCN or lattice storing the "true" frame-
wise posterior distribution can be used.

**Configuration**


.. code-block: ini

    [*.network.fCN-features]
    type                        = fCN-features
    gamma                       = 1.0
    rescore-mode                = clone*|in-place-cached|in-place
    # features
    confidence-key              = <unset>
    error-key                   = <unset>
    error.alpha                 = 0.05
    [*.network.fCN-features.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:fCN] [2:lattice]
    output:
    0:lattice
    


fCN-gamma-correction
--------------------

Perform a in-situ gamma correction of the slot-wise
posterior probability distribution.

**Configuration**


.. code-block: ini

    [*.network.fCN-gamma-correction]
    type                        = fCN-gamma-correction
    gamma                       = 1.0
    normalize                   = true

**Port assignment**

.. code-block: ini

    input:
    0:fCN
    output:
    0:fCN
    


fWER-evaluator
--------------

Calculate smoothed and unsmoothed (expected) time frame error.
Hypothesis and reference lattice must be linear. Alternatively,
an fCN can be provided as reference allowing to calculate an
expected fWER; see min.fWER-decoding.

**Configuration**


.. code-block: ini

    [*.network.fWER-evaluator]
    type                        = fWER-evaluator
    dump.channel                = <unset>
    alpha                       = 0.05

**Port assignment**

.. code-block: ini

    input:
    0:lattice 1:reference-lattice|2:reference-fCN
    output:
    0:lattice
    


filter
------

Filter lattice by input(output)

**Configuration**


.. code-block: ini

    [*.network.filter]
    type                        = filter
    input                       = <unset>
    output                      = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


fit
---

Fit lattice into segment boundaries.
The fitted lattice has the following properties:
* single initial state (id=0) s_i and single final state s_f (id=1)
* weight of the final state s_f is semiring one
* 0 = time(s_i) <= time(s) < time(s_f)
* for each path in the original lattice, there exist a path in the fitted lattice with the same score (w.r.t to the used semiring); and vice versa
* optional: each arc ending in s_f has </s>-label
The bounding box is given by the segment provided at port 1.
If no segment is provided, start time is 0 and end time is
is the max. time of all states in the lattice.
Remark: This node can be used to normalize the final states
of a lattice.

**Configuration**


.. code-block: ini

    [*.network.fit]
    type                        = fit
    force-sentence-end-labels   = false

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:segment]
    output:
    0:lattice [1:segment]
    


frame-CN-builder
----------------

*DEPRECATED: see "center-frame-CN-builder*

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    output:
    0:lattice(best)
    1:CN(normalized)   2:lattice(normalized CN)
    3:CN               4:lattice(CN)
    5:fCN              6:lattice(union)
    


fsa-reader
----------

Read fsas.
All filenames are interpreted relative to a given directory,
if specified, else to the current directory.
The current fsa is buffered for multiple access


**Configuration**


.. code-block: ini

    [*.network.fsa-reader]
    type                        = fsa-reader
    path                        = <path>
    # in case of acceptors
    alphabet.name               = {lemma-pronunciation|lemma*|syntax|evaluation}
    # in case of transducers
    input-alphabet.name         = {lemma-pronunciation|lemma*|syntax|evaluation}
    output-alphabet.name        = {lemma-pronunciation|lemma*|syntax|evaluation}

**Port assignment**

.. code-block: ini

    input:
    1:segment | 2:string
    output:
    0:lattice, 1:fsa
    


info
----

Dump information and statistics for incoming lattice.
Runtime/memory requirements:
cheap:    O(1), lattice is not traversed.
normal:   O(N), lattice is traversed once; no caching.
extended: O(N), lattice is traversed multiple times,
lattice is cached.
memory:   n/a
Attention: "extended" requires an acyclic lattice.

**Configuration**


.. code-block: ini

    [*.network.info]
    type                        = info
    info-type                   = cheap|normal*|extended|memory

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


intersection
------------

Intersection of two lattices; for algorithm details see FSA

**Configuration**


.. code-block: ini

    [*.network.intersection]
    type                        = intersection
    append                      = false

**Port assignment**

.. code-block: ini

    input:
    0:lattice, 1:lattice
    output:
    0:lattice
    


local-cost-decoder
------------------

Computes an arc-wise score comprised of a
word penalty and an approximated risk.
The approximated risk is based on the
time overlap of hypothesis and reference
arc, e.g. 

**Configuration**


.. code-block: ini

    [*.network.local-cost-decoder]
    type                        = approximated-risk-scorer
    rescore-mode                = clone
    score-key                   = <unset>
    confidence-key              = <unset>
    word-penalty                = 0.0
    search-space                = union|mesh*
    risk-builder                = overlap*|local-alignment
    [*.network.local-cost-decoder.overlap]
    scorer                      = path-symetric*|arc-symetric
    path-symetric.alpha         = 0.5 # [0.0,1.0]
    [*.network.local-cost-decoder.local-alignment]
    scorer                      = approximated-accuracy|continous-cost1|continous-cost2*|discrete-cost
    continous-cost1.alpha       = 1.0 # [0.0,1.0]
    continous-cost2.alpha       = 0.5 # [0.0,0.5]
    discrete-cost.alpha         = 0.5 # [0.0,0.5]
    [*.network.local-cost-decoder.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    0:lattice(best) 1:lattice(rescored)
    


log
---

Manipulate a single dimension:
f(x_d) = <scale> * log(x_d)

**Configuration**


.. code-block: ini

    [*.network.log]
    type                        = log
    append                      = false
    key                         = <symbolic key or dim>
    scale                       = 1.0
    rescore-mode                = {clone*, in-place-cached, in-place}

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


map-alphabet
------------

Map the input(output, or both) labels of the incoming lattice to another
alphabet. The concrete mapping is specified by the used lexicon.
If the incoming lattice is an acceptor and output mapping is activated,
the resulting lattice is a transducer.
For lemma-pronunciation <-> lemma correct time boundary preservation
is guaranteed, for all other mappings it is not.
For lemma -> preferred-lemma-pronunciation a successfull mapping is
guaranteed, if the lexicon's read-only flag is not set, i.e. for a
lemma with no pronunciation, the empty pronunciation is added.
If project input(output) is activated, the resulting lattice is an
acceptor, where the labels are the former input(output) labels.
If invert is activated and the lattice is a transducer, input and
output labels are toggled.
All mappings have a lazy implementation.

**Configuration**


.. code-block: ini

    [*.network.map-alphabet]
    type                        = map-alphabet
    map-input                   = to-phoneme|to-lemma|to-lemma-pron|to-preferred-lemma-pron|to-synt|to-eval|to-preferred-eval
    map-output                  = to-phoneme|to-lemma|to-lemma-pron|to-preferred-lemma-pron|to-synt|to-eval|to-preferred-eval
    project-input               = false
    project-output              = false
    invert                      = false

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


map-labels
----------

Map the input labels of the incoming lattice according to the
specified mappings:
* non-words, i.e. words having the empty eval. tok. seq., to epsilon
* compound word splitting, i.e. split at " ", "_", or "-"
* static mapping, where the mappings are loaded from a file; the
format is "<source-word> <target-word-1> <target-word-2> ...\n
All mappings preserve or interpolate time boundaries, all mappings
have a static implementation.

**Configuration**


.. code-block: ini

    [*.network.map-labels]
    type                        = map-labels
    map-to-lower-case           = false
    map-non-words-to-eps        = false
    split-compound-words        = false
    map.file                    = 
    map.encoding                = utf-8
    map.from                    = lemma
    map.to                      = lemma
    project-input               = false

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


mesh
----

Reducde lattice to its boundary-conditioned form:
either using the full boundary information or only
the time information, i.e. building the purely
time-conditioned form.

**Configuration**


.. code-block: ini

    [*.network.mesh]
    type                        = mesh

**Port assignment**

.. code-block: ini

    mesh-type                   = full*|time
    input:
    0:lattice
    output:
    0:lattice
    


min-fWER-decoder
----------------

Decode over all incoming lattices.
Search space:
union: Decode over union of all lattices.
mesh:  Decode over time-conditioned lattice build
build from the union of all lattices.
cn:    Decode from fCN directly, unrestriced
search space
If no fCN is provided at port 0, then
a fCN is calculated over all incoming lattices.

**Configuration**


.. code-block: ini

    [*.network.min-fWER-decoder]
    type                        = min-fWER-decoder
    search-space                = union|mesh*|cn
    [*.network.min-fWER-decoder.union]
    alpha                       = 0.05
    non-word-alpha              = 0.05
    confidence-key              = <unset>
    [*.network.min-fWER-decoder.mesh]
    alpha                       = 0.05
    non-word-alpha              = 0.05
    confidence-key              = <unset>
    [*.network.min-fWER-decoder.cn]
    word-penalty                = 2.5# fwd/bwd scores are used for calculating fCN, if not specified
    # and for applying fwd/bwd pruning
    [*.network.min-fWER-decoder.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    [0:fCN] 1:lattice [2:lattice [...]]
    output:
    0:lattice
    


minimize
--------

Determinize and minimize lattice; for algorithm details see FSA

**Configuration**


.. code-block: ini

    [*.network.minimize]
    type                        = minimize

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


multiply
--------

Manipulate a single dimension:
f(x_d) = <scale> * x_d

**Configuration**


.. code-block: ini

    [*.network.multiply]
    type                        = multiply
    append                      = false
    key                         = <symbolic key or dim>
    scale                       = 1.0
    rescore-mode                = {clone*, in-place-cached, in-place}

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


n-best
------

Find the n best paths in a lattice.
The algorithm is based on Eppstein and is
optimized for discarding duplicates, i.e.
the algorithm is not optimal (compared to
the origianl Eppstein algorithm) for generating
n-best lists containing duplicates.
The algorithm seems to scale well at least up
to 100.000-best lists without duplicates.
If the "ignore-non-word" option is activated,
then two hypotheses only differing in non-words
are considered duplicates.
The resulting n-best list preserves all non-word-
and epsilon-arcs and has correct time boundaries.

**Configuration**


.. code-block: ini

    [*.network.n-best]
    type                        = n-best
    n                           = 1
    remove-duplicates           = true
    ignore-non-words            = true
    score-key                   = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


non-word-closure-filter
-----------------------

Given states s and e. Pathes_w(s,e) is the set of all pathes from
s to e having exactly one arc labeled with w and all others labeled
with epsilon. Arcs_w(s,e) is the set of all arcs in Pathes_w(s,e) 
labeled with w. Arcs_s'/w(s,e) is the set of all arcs in Arcs_w(s,e)
having source state s'. Pathes_s'/w(s,e) is the subset of Pathes_w(s,e),
such that each path in Pathes_s'/w(s,e) includes an arc in Arcs_s'/w(s,e).

for each w, (s,e):
for each a in Arcs_w(s,e) keep only the best scoring path in
Pathes_w(s,e) that includes a.
-> see classical epsilon-removal over the tropical semiring

The resulting graph is a subgraph of the original input and contains the
Viterbi path of the original graph.
The implementation is static, i.e not lazy.

**Configuration**


.. code-block: ini

    [*.network.non-word-closure-filter]
    type                        = non-word-closure-filter

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


non-word-closure-normalization-filter
-------------------------------------

If a state s has at least one outgoing arc, and all outgoing arcs
are non-word arcs, then s is disacarded and all outgoing arcs are
joined with previous/next non-word arcs to a new eps-arc. All scores
and word-arc times are kept w.r.t. to the given semiring.


**Configuration**


.. code-block: ini

    [*.network.non-word-closure-normalization-filter]
    type                        = non-word-closure-normalization-filter

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


non-word-closure-removal-filter
-------------------------------

For each state s and each word arc a leaving a state of the
non-word closure of s, let a start from s, attach the correct
score w.r.t to the used semiring (e.g. score of best path for
the tropical semiring) and add the additional time (i.e. the
time nedded for "crossing" the closure.

**Configuration**


.. code-block: ini

    [*.network.non-word-closure-removal-filter]
    type                        = non-word-closure-removal-filter

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


non-word-closure-strong-determinization-filter
----------------------------------------------

Given states s and e. Pathes_w(s,e) is the set of all pathes from
s to e having exactly one arc labeled with w and all others labeled
with epsilon. Arcs_w(s,e) is the set of all arcs in Pathes_w(s,e) 
labeled with w. Arcs_s'/w(s,e) is the set of all arcs in Arcs_w(s,e)
having source state s'. Pathes_s'/w(s,e) is the subset of Pathes_w(s,e),
such that each path in Pathes_s'/w(s,e) includes an arc in Arcs_s'/w(s,e).

for each w, (s,e):
keep only the best scoring path in Pathes_w(s,e)
-> classical epsilon-removal over the tropical semiring with
determinization over all pathes from s to e

Attention: 
Due to the retaining of non-word arcs the determinization can not
always be guaranteed.

The resulting graph is a subgraph of the original input and contains the
Viterbi path of the original graph.
The implementation is static, i.e not lazy.

**Configuration**


.. code-block: ini

    [*.network.non-word-closure-strong-determinization-filter]
    type                        = non-word-closure-strong-determinization-filter

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


non-word-closure-weak-determinization-filter
--------------------------------------------

Given states s and e. Pathes_w(s,e) is the set of all pathes from
s to e having exactly one arc labeled with w and all others labeled
with epsilon. Arcs_w(s,e) is the set of all arcs in Pathes_w(s,e) 
labeled with w. Arcs_s'/w(s,e) is the set of all arcs in Arcs_w(s,e)
having source state s'. Pathes_s'/w(s,e) is the subset of Pathes_w(s,e),
such that each path in Pathes_s'/w(s,e) includes an arc in Arcs_s'/w(s,e).

for each w, (s,e):
for each s' keep only the best scoring path in Pathes_s'/w(s,e)
-> classical epsilon-removal over the tropical semiring with statewise
determinization

The resulting graph is a subgraph of the original input and contains the
Viterbi path of the original graph.
The implementation is static, i.e not lazy.

**Configuration**


.. code-block: ini

    [*.network.non-word-closure-weak-determinization-filter]
    type                        = non-word-closure-weak-determinization-filter

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


oracle-alignment
----------------

Compute oracle alignment between CN and reference.
The oracle loss requires a posterior score, i.e.
Cost functions:
* oracle-error
0, if word in slot
1, else
* weighted-oracle-error
i**alpha, where
i is the position of the reference word in the slot,
resp. 100, if the reference word is not in the slot
* oracle-loss
1 - p(word|slot), if word in slot
100, else,
i.e. align w.r.t to minimum oracle error as primary criterion
and minimum expected error as secondary criterion
either a normalized CN or posterior key defined.

**Configuration**


.. code-block: ini

    [*.network.oracle-alignment]
    type                        = oracle-alignment
    cost                        = oracle-cost*|weighted-oracle-cost|oracle-loss
    weighted-oracle-cost.alpha  = 1.0
    posterior-key               = <unset>
    beam-width                  = 100

**Port assignment**

.. code-block: ini

    input:
    0:CN 1:lattice|2:string|3:CN|4:segment(with orthography)
    output:
    0:oracle-CN
    


pivot-CN-builder
----------------

*DEPRECATED: see "pivot-arc-CN-builder*

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    output:
    0:lattice(best)
    1:CN(normalized)   2:lattice(normalized CN)
    3:CN               4:lattice(CN)
    6:lattice(union)
    


pivot-arc-CN-builder
--------------------

Build CN from incoming lattice(s).
The pivot elements are the arcs form the lattice path with the
maximum a posterior probability, i.e. lowest fwd/bwd score.
Setting map=true stores a lattice <-> CN mapping, which is
required for producing CN based lattice features.

**Configuration**


.. code-block: ini

    [*.network.pivot-arc-CN-builder]
    type                        = pivot-arc-CN-builder
    statistics.channel          = nil
    confidence-key              = <unset>
    map                         = false
    distance                    = weighted-time*|weighted-pivot-time
    [*.network.pivot-arc-CN-builder.weighted-time]
    posterior-impact            = 0.1
    edit-distance               = false
    [*.network.pivot-arc-CN-builder.weighted-pivot-time]
    posterior-impact            = 0.1
    edit-distance               = false
    fast                        = false
    [*.network.pivot-arc-CN-builder.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    output:
    0:lattice(best)
    1:CN(normalized)   2:lattice(normalized CN)
    3:CN               4:lattice(CN)
    6:lattice(union)
    


project
-------

Change the semiring by projecting the source semiring onto the target semiring

**Configuration**


.. code-block: ini

    [*.network.projection]
    type                        = project
    scaled                      = true
    [*.network.projection.semiring]
    type                        = tropical|log
    tolerance                   = <default-tolerance>
    keys                        = key1 key2 ...
    key1.scale                  = <f32>
    key2.scale                  = <f32>
    ...
    [*.network.projection.matrix]
    key1.row                    = <old-key[1,1]> <old-key[1,2]> ...
    key2.row                    = <old-key[2,1]> <old-key[2,2]> ...
    ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


properties
----------

Change and/or dump lattice and fsa properties

**Configuration**


.. code-block: ini

    [*.network.properties]
    type                        = properties
    dump                        = true|false

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


prune-CN
--------

Prune CN slotwise; CN must be normalized.
If a threshold is given, probability mass pruning is
applied, i.e. per slot only the first n entries having
in sum the desired probability mass are kept.
If the maximum slot size n is given, then at most n
arcs are kept per slot.
On request, the slot-wise probability distribution
is re-normalized.
If epsilon slot removal is activated, then all slots will
be removed, where the posterior probability of the epsilon
arc exceeds the threshold.
Attention: In situ pruning is performed.

**Configuration**


.. code-block: ini

    [*.network.prune-CN]
    type                        = prune-CN
    threshold                   = <unset>
    max-slot-size               = <unset>
    normalize                   = true
    remove-eps-slots            = false
    eps-slot-removal.threshold  = 1.0

**Port assignment**

.. code-block: ini

    input:
    x:CN
    output:
    x:CN
    


prune-fCN
---------

Prune fCN slotwise.
If a threshold is given, probability mass pruning is
applied, i.e. per slot only the first n entries having
in sum the desired probability mass are kept.
If the maximum slot size n is given, then at most n
arcs are kept per slot.
On request, the slot-wise probability distribution
is re-normalized.
If epsilon slot removal is activated, then all slots will
be removed, where the posterior probability of the epsilon
arc exceeds the threshold.
Attention: In situ pruning is performed.

**Configuration**


.. code-block: ini

    [*.network.prune-fCN]
    type                        = prune-fCN
    threshold                   = <unset>
    max-slot-size               = <unset>
    normalize                   = true
    remove-eps-slots            = false
    eps-slot-removal.threshold  = 1.0

**Port assignment**

.. code-block: ini

    input:
    x:fCN
    output:
    x:fCN
    


prune-posterior
---------------

Prune arcs by posterior scores.
By default, the fwd/bwd scores are calculated over the normalized
log semiring derived from the lattice's semiring. Alternatively,
a semiring can be specified.
If the lattice is empty after pruning, the single best result is
returned (only if trimming is activated).

**Configuration**


.. code-block: ini

    [*.network.prune-posterior]
    type                        = prune-posterior
    configuration.channel       = nil
    statistics.channel          = nil
    trim                        = true
    # pruning parameters
    relative                    = true
    as-probability              = false
    threshold                   = inf
    ...
    # parameter for fwd./bwd. calculation
    [*.network.prune-posterior.fb]
    see FB-builder ...
    

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


reader
------

Read lattice(s) from file;
All filenames are interpreted relative to a given directory,
if specified, else to the current directory.
The current lattice is buffered for multiple access.

**Configuration**


.. code-block: ini

    [*.network.reader]
    type                        = reader
    format                      = flf|htk
    path                        = <lattice-base-dir>
    # if format is flf
    [*.network.reader.flf]
    context-mode                = trust|adapt|*update
    [*.network.reader.flf.partial]
    keys                        = key1 key2 ...
    [*.network.reader.flf.append]
    keys                        = key1 key2 ...
    key1.scale                  = 1.0
    key2.scale                  = 1.0
    ...
    # if format is htk
    [*.network.reader.htk]
    context-mode                = trust|adapt|*update
    log-comments                = false
    suffix                      = .lat
    fps                         = 100
    encoding                    = utf-8
    slf-type                    = forward|backward
    capitalize                  = false
    word-penalty                = <f32>
    silence-penalty             = <f32>
    merge-penalties             = false
    set-coarticulation          = false

**Port assignment**

.. code-block: ini

    input:
    1:segment | 2:string
    output:
    0:lattice
    


recognizer
----------

The Sprint Recognizer.
Output are linear or full lattices in Flf format.
The most common operations on recognizer output can be directly
performed by the node (in the given order):
# apply non-word-closure filter
# confidence score calculation
# posterior pruning
If lattices are provided at port 0, the search-space is restricted
to the lattice, i.e. the lattice is used as language model.
The parameter "grammar-key" allows to choose a dimension that
provides the lm-score, otherwise the projection defined by the
semiring is used.

**Configuration**


.. code-block: ini

    [*.network.recognizer]
    type                        = recognizer
    grammar.key                 = <unset>
    grammar.arcs-limit          = <unset>
    grammar.log.channel         = <unset>
    <all parameters belonging to the search configuration>
    add-pronunciation-score     = false
    add-confidence-score        = false
    apply-non-word-closure-filter= false
    apply-posterior-pruning     = false
    posterior-pruning.threshold = 200
    fb.alpha                    = <1/lm-scale>

**Port assignment**

.. code-block: ini

    input:
    [0:lattice] 1:bliss-speech-segment
    output:
    0:lattice
    


reduce
------

Reduce the scores of two or more dimensions to the first given dimension.
Basically the weighted score of the second, third, and so on key
are added to the first score of the first key and then set to semiring one,
i.e. 0, and the scale of the dimension is set to 1.
The weighted sum of the score vector remains unchanged.

**Configuration**


.. code-block: ini

    [*.network.scores-reduce-scores]
    type                        = reduce-scores
    keys                        = <key1> <key2> ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


remove-epsilons
---------------

Those arcs are removed having epsilon as input and output

**Configuration**


.. code-block: ini

    [*.network.remove-epsilons]
    type                        = remove-epsilons
    log-semiring                = true|false*
    log-semiring.alpha          = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


remove-null-arcs
----------------

Remove arcs of length 0(regardless if input/output is eps)

**Configuration**


.. code-block: ini

    [*.network.remove-null-arcs]
    type                        = remove-null-arcs
    log-semiring                = true|false*
    log-semiring.alpha          = <unset>

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


rescale
-------

Rescales and rename single dimensions of lattice's current semiring.
Technically, the semiring of the lattice is replaced by a new one.

**Configuration**


.. code-block: ini

    [*.network.rescale]
    type                        = rescale
    <key1>.scale                = <keep existing scale>
    <key1>.key                  = <keep existing key name>
    ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice
    output:
    0:lattice
    


segment-builder
---------------

Combines incoming data to a segment;
missing data fields are replaced by default values.

**Configuration**


.. code-block: ini

    [*.network.segment-builder]
    type                        = segment-builder
    progress.channel            = <unset>

**Port assignment**

.. code-block: ini

    input:
    [0:bliss-speech-segment]
    [1:audio-filename(string)]
    [2:start-time(float)]
    [3:end-time(float)]
    [4:track(int)]
    [5:orthography(string)]
    [6:speaker-id(string)]
    [7:condition-id(string)]
    [8:recording-id(string)]
    [9:segment-id(string)]
    output:
    0: segment
    


select-n-best
-------------

Gets an n-best lattice as input and provides at port x
the xth best lattice, or the empty lattice if x exceeds
the size of the n-best list; the indexing starts from 0.

**Configuration**


.. code-block: ini

    [*.network.select-n-best]
    type                        = select-n-best

**Port assignment**

.. code-block: ini

    input:
    0:n-best-lattice
    output:
    x:linear-lattice
    


sink
----

Let all incoming lattices/CNs/fCNs sink

**Configuration**


.. code-block: ini

    [*.network.sink]
    type                        = sink
    sink-type                   = lattice*|CN|fCN
    warn-on-empty               = true
    error-on-empty              = false

**Port assignment**

.. code-block: ini

    input:
    x:lattice/CN/fCN
    no output
    


speech-segment
--------------

Distributes the speech segments provided
by the Bliss corpus visitor.
The segment is provided as Bliss speech segment
and as Flf segment.

**Configuration**


.. code-block: ini

    [*.network.speech-segment]
    type                   = speech-segment

**Port assignment**

.. code-block: ini

    no input
    output:
    0:segment 1:bliss-speech-segment
    


state-cluster-CN-builder
------------------------

Build CN from incoming lattice(s).
The algorithm builds state clusters first and deduces
from them arc clusters.
Setting map=true stores a lattice <-> CN mapping, which is
required for producing CN based lattice features.
The algorithm is a little picky w.r.t. to the structure of
the incoming lattice; try remove-null-arcs(Remark: this is
only a hack, better someone fixes this in general!)

**Configuration**


.. code-block: ini

    [*.network.state-cluster-CN-builder]
    type                        = cluster-CN-builder
    statistics.channel          = nil
    confidence-key              = <unset>
    map                         = false
    remove-null-arcs            = false
    allow-bwd-match             = false
    [*.network.state-cluster-CN-builder.fb]
    see FB-builder ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [...]]
    output:
    0:lattice(best)
    1:CN(normalized)   2:lattice(normalized CN)
    3:CN               4:lattice(CN)
    6:lattice(state cluster)
    


string-to-lattice
-----------------

Convert a string to a linear lattice

**Configuration**


.. code-block: ini

    [*.network.string-to-lattice]
    type                        = string-to-lattice
    alphabet                    = lemma-pronunciation|lemma|syntax|evaluation
    [*.network.string-to-lattice.semiring]
    type                        = tropical|log
    tolerance                   = <default-tolerance>
    keys                        = key1 key2 ...
    key1.scale                  = <f32>
    key2.scale                  = <f32>
    ...

**Port assignment**

.. code-block: ini

    input:
    0:string
    output:
    0:lattice
    


unite
-----

Build union of incoming lattices.
Incoming lattices need to have
* same alphabets and
* same semiring
or a new semiring is defined.

**Configuration**


.. code-block: ini

    [*.network.unite]
    type                        = unite
    [*.network.unite.semiring]
    type                        = tropical|log
    tolerance                   = <default-tolerance>
    keys                        = key1 key2 ...
    key1.scale                  = <f32>
    key2.scale                  = <f32>
    ...

**Port assignment**

.. code-block: ini

    input:
    0:lattice [1:lattice [2:lattice ...]]
    output:
    0:lattice
    


writer
------

Write lattice(s) to file;
If input at port 1, use segment id as base name,
if input at port 2, use string as base name,
else, get filename from config.
Base name is modified by adding suffix and prefix, if given.
All filenames are interpreted relative to a given directory,
if specified, else to the current directory.

**Configuration**


.. code-block: ini

    [*.network.writer]
    type                        = writer
    format                      = flf|htk
    # to store a single lattice
    file                        = <lattice-file>
    # to store multiple lattices,
    # i.e. if incoming connection at port 1 or 2
    path                        = <lattice-base-dir>
    prefix                      = <file-prefix>
    suffix                      = <file-suffix>
    [*.network.writer.flf.partial]
    keys                        = key1 key2 ...
    add                         = false
    [*.network.writer.htk]
    fps                         = 100
    encoding                    = utf-8

**Port assignment**

.. code-block: ini

    input:
    0:lattice[, 1:segment | 2:string]
    output:
    0:lattice


