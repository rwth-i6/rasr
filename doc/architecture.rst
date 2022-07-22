Architecture
============

Build environment
-----------------

The build process is organized in Makefiles ::

    Config.make
    Makefile
    Makefile.cfg
    Modules.make
    Options.make
    Rules.make
    config/*.make

The low-level ("atomic") makefiles are:

* ``Modules.make`` lists enabled modules. Modules are high level concepts that are defined as macros in ``Modules.make`` and used in order to enable and disable parts of code using ``ifdef`` directive.
* ``Options.make`` sets some high level options (debug mode, compiler, binary naming etc)
* ``Rules.make`` are actual build rules (including different levels of "clean")

  * ``config/cc-*.make`` compiler selection, code-relevant compiler/linker flags
  * ``config/os-*.make`` compiler/linker flags related to external paths and modules
  * ``config/proc-*.make`` compiler flags related to CPU architecture

The "aggregate" makefiles to be included from src/* are

* ``Makefile.cfg`` (simple wrapper for ``Config.make``)
* ``Config.make`` (main aggregate, included in ``Makefile``, defining high level build targets

The source code is compiled by two levels of makefiles that include the aggregate makefiles:

* ``src/$Foo/Makefile`` - builds a library ``libSprintFoo.a``
* ``src/Tools/$Bar/Makefile`` - builds an executable ``$Bar`` and links multiple libraries ``libSprint*.a`` as well as third-party libraries

The sources are .cc and .hh files, that are compiled into individual object files in ``src/$Foo/.build/linux-x86_64-standard/*.o`` that are later archived into a ``libSprint*.a`` via ``ar rucs``.

Docker environment
^^^^^^^^^^^^^^^^^^

Most modules are straightforward to enable/disable and do not require complex configuration. Usually, it boils down to a few -I flags for the includes, -L and -l for the linker flags.

The modules that require the most attention are MODULE_PYTHON and MODULE_TENSORFLOW. In order to simplify the building process, we provide a docker environment that illustrates how to get all dependencies right.

MODULE_PYTHON
^^^^^^^^^^^^^

It should support both Python 2 and Python 3.
Use `numpy.get_include() <https://numpy.org/devdocs/reference/generated/numpy.get_include.html>`_ to obtain include path from a local virtual environment.

It can be used for various purpose:

* ``PythonFeatureScorer`` (anywhere for feature scoring, such as decoding or generating alignments)
* ``PythonTrainer`` (interface for the legacy RASR NN training code, also used in decoding together with TrainerFeatureScorer)
* ``PythonSegmentOrdering`` (sorting or filtering the dataset)
* ``PythonLayer`` (plug in for the legacy RASR NN code, decoding/training/anything)
* ``PythonControl`` (very custom scriptable control to e.g. get data out of dataset, or calculate soft/hard alignments, generate FSAs, etc.)
* :doc:`training/sequence_training.rst` (RASR gets the posteriors and returns loss and error signal)

On RETURNN side, several implementations for these interfaces exists:

* `SprintDataset <https://github.com/rwth-i6/returnn/blob/master/SprintDataset.py>`_ and `ExternSprintDataset <https://github.com/rwth-i6/returnn/blob/master/SprintExternInterface.py>`_. (Which can either use PythonTrainer or PythonControl as the RASR entry point.)
* Sequence training
* `Full-sum training <https://www-i6.informatik.rwth-aachen.de/publications/download/1035/Zeyer--2017.pdf>`_ (gets FSA via PythonControl)

Note that there is also the Tensorflow module which provides a separate FeatureScorer.

MODULE_TENSORFLOW
^^^^^^^^^^^^^^^^^

In order to use TF C++ API, we have to be able to include TF headers (e.g. tensorflow/core/framework/tensor.h) and link against libtensorflow_cc.so and libtensorflow_framework.so. Also, we want TF to include Intel MKL support and link against libiomp5.so and libmklml_intel.so. The only way to get libtensorflow_cc.so is to build TF from sources.

Dependencies
^^^^^^^^^^^^

You can run ``scripts/requirements.sh`` to verify if your build environment meets the requirements. As of January 2020, RASR is known to compile with gcc between 4.8 and 7.3, optionally with Python 3.6 and TF 1.12 to 1.15. 

Currently, RASR requires C++11. As of January 2020, we plan to upgrade to C++14 or 17 at some later point in time because of some internal dependencies in our infrastructure.

Run time environment
--------------------

Once the binaries have been built, external shared libraries need to be made available to the run time environment via ``$LD_LIBRARY_PATH``. Use ``ldd`` to check the run time dependencies.

Any code that relies on an external BLAS library (OpenBLAS or Intel MKL) will respect the environment variable ``$OMP_NUM_THREADS``. If not set, the value defaults to the number of all available CPU cores.

Source code
-----------

Coding conventions
^^^^^^^^^^^^^^^^^^

* use clangformat
* no tabs, 4 spaces per indentation level
* camel case for files and directories (start with a capital)
* camel case for variables (classes: start with capital, everything else: start with lower case)
* avoid abbreviations
* avoid subdirectories
* private variables should end with an underscore
* `include guard <https://en.wikipedia.org/wiki/Include_guard>`_ should include folder name
* no exceptions

Structure of src/
^^^^^^^^^^^^^^^^^

**Directories**

* Am - acoustic model
* Audio - processing of different audio formats
* Bliss (Better Lexical Information Sub-System) - corpus and lexicon processing
* Cart - classification and regression tree for state tying (training and evaluation)
* Core - fundamental RASR building blocks, helper functions
* Flf - lattice processing
* :ref:`Flow` - Flow network for feature extraction pipelines
* `Fsa <https://www-i6.informatik.rwth-aachen.de/~kanthak/fsa.html>`_ - finite state automaton library (similar to OpenFST)
* Lattice - legacy lattice format used by the :doc:`tools/speech_recognizer.rst`
* Lm - language model
* Math - basic math data structures and algorithms, BLAS wrappers, interface to CUDA
* Mc - model combination
* Mm - mixture models
* Nn - legacy feed forward neural networks, Python interfaces
* OpenFst - `OpenFST <http://www.openfst.org OpenFST>`_ interface
* Python - Python interface
* Search - HMM decoder
* Signal - signal processing data structures and algorithms to be used in Flow nodes
* Sparse - sparse data structures (vectors)
* Speech - high level data structures and algorithms for processing speech data
* Tensorflow - TF interfface
* Test - `cppunit <https://en.wikipedia.org/wiki/CppUnit>`_ test suite
* Tools - executable binaries

**Files**

* Makefile - wrapper to define which subfolders (and in which order) to build; creates SourceVersion* based on git status.
* Modules.hh - list of enabled modules, automatically generated from Modules.make; sources will include this file so changing modules might require re-compilation.
* SourceVersion* - git status for version tracking

Data types
^^^^^^^^^^

RASR is designed to operate on 32- or (preferably) 64-bit CPUs. src/Core/Types.hh provides several typedefs (u32, s32, f32 etc) that should be used throughout the code. Also, the file provides some template wrappers that allow to access the type name as a string for some specific applications (logging, XML data formats, Flow), the minimum and maximum representable value etc.

Structure of executables
^^^^^^^^^^^^^^^^^^^^^^^^

Each binary in ``Tools/*`` defines the ``main()`` entry function by using the macro ``APPLICATION(ToolName)`` from ``src/Core/Application.hh`` on a class ``ToolName`` derived from ``Core::Application``. It will call ``ToolName::main()`` to start performing actual work. The constructor of ``ToolName`` has to invoke macro ``INIT_MODULE(Foo)`` defined in ``src/Core/Application.hh`` in order to initialize a module "Foo" (where Foo is a namespace in which src/Foo/Module.hh defines a Module class). The initialization creates a singleton wrapper object and its constructor takes care of run time registry of available features or any other one-time init work.

The singleton object can be accessed from other parts in the code via ``Core::Application::us()``, e.g. to call tool logging methods.

The motivation for such structure is the following:

* unification of interfaces between modules and tools
* enabling common configuration and logging mechanisms for all tools
* run time registry of available file formats, Flow nodes, available features (enabled via modules)

The constructor of an application usually calls some methods derived from Core::Application like 

* INIT_MODULE(Foo) to enable features exposed from module Foo
* setTitle() to give the application a name to be used in logging
* setDefaultLoadConfigurationFile() to (optionally) disable looking for a default config file

Applications can override ``Core::Application::getUsage()`` to print usage upon call with ``--help``.

Most applications are multi-purpose tools (contrary to the `Unix philosophy <https://en.wikipedia.org/wiki/Unix_philosophy>`_) because of the research nature of speech recognition. Their main() function will parse the configuration made available through the config mechanisms (config files, command line, etc) and decide which "action" to execute once and return EXIT_SUCCESS on success.

Typically, there is no need for a destructor since Core::Application does not allocate anything complex. Even if the derived classes do, the d'tor is called before exiting the outer most main() function such that all allocated memory will be freed anyway.

Acoustic model
^^^^^^^^^^^^^^

Acoustic model (AM) provides emission probabilities p(x_t|s_t) for the HMM decoder, e.g. during acoustic training or recognition. For a given feature vector x_t, it will calculate the distribution over all HMM states s_t and make them available to the decoder.

In order to instantiate an AM, we can call the static method ``Am::Module::instance().createAcousticModel(config, lexicon)`` which in turn will (almost always) instantiate ``Am::ClassicAcousticModel``. It needs the lexicon in order to correctly initialize special cases of the state model.

Alternatively, the AM can be created implicitly via ``Speech::ModelCombination()`` which is a wrapper object for consistent access to AM, LM and a lexicon.

In order to obtain the emission probabilities from an AM, we have to use its ``Am::ClassicAcousticModel::featureScorer()`` interface. We feed the scorer a feature vector via ``addFeature(v)`` and obtain the scores via ``flush()``. This function returns an object of type ``Mm::FeatureScorer::Scorer`` that provides the number of classes (``nEmissions()``) as well as the actual acoustic scores ``-log p(x|e)`` (``score(s)``).

**TODO**: discuss 

* minus log prob domain
* buffered access
* reset()
* finalize()

**Gaussian mixture models**

GMMs are configured via selector ``mixture-set``.

**Legacy DNN**

Feed forward DNNs (as used e.g. in segmenter) are configured via selector ``neural-network`` (Nn/NeuralNetwork.cc).

**Tensorflow models**

TF models are configured via selector ``loader`` (Tensorflow/TensorflowForwardNode.cc, Tensorflow/TensorflowFeatureScorer.cc) and have to specify the parameters

* type - meta or vanilla
* meta-graph-file - meta file with the TF graph
* saved-model-file - file name prefix that can be expanded to .data and .index that contain the actual parameters
* required-libraries - colon-separated list of shared libraries that are loaded via `tf::Env::Default->LoadLibrary <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/env.h#L314>`_, e.g. for loading pre-compiled custom TF ops like LSTM kernels.

Alignment
^^^^^^^^^

See :ref:`Alignment`, :doc:`training/alignment_generation.rst` and :doc:`training/converting_alignment_formats.rst`

The alignment is a frame-level mapping from feature vector index within a segment (no time stamps are stored here, just integer indices 0, 1, 2, ... relative to segment start) to a triphone HMM state. Please note that the alignment always stores triphones, even if the user is only interested in monophones. Also, we don't store tied states (although we support this feature) so that we can apply any state tying strategy on top of an existing alignment. Alignment format can store frame weights (float), e.g. for sophisticated AM training. The alignment stores both, the allophone index and the state index in one 32-bit integer: the 26 most significant bits of the allphone state id are used for the allophone index; the 6 least-significant bits for the state. This means that the maximal id for the allophone index is 2^26 and for HMM state is 2^6 (we mostly use 3 in ASR and a little bit more in sign language or handwriting recognition).

The alignment only stores integer indices instead of strings, so in order to be able to read an existing alignment, we have to specify an :ref:`Allophone Symbol` (similar to symbol or alphabet file in OpenFST) where the line number corresponds to the integer index and the string represents the triphone label.

The format is specified in ``src/Speech/Alignment.cc``. You can use the :ref:`Archiver Tool` for reading the alignment format in plain text format or `SprintCache.py <https://github.com/rwth-i6/returnn/blob/master/SprintCache.py>`_ for reading from Python.

You can instantiate ``Speech::Alignment``, which inherits from ``std::vector<Speech::AlignmentItem>`` and use regular ``std::vector`` operations. The wrapper provides some additional features including serialization and ``operator<<() operator>>()``.

Allophone
^^^^^^^^^

An allophone is a general term that can be a monophone or a triphone, depending on the configuration. An allophone state is therefore an HMM state that can be represented as a string (see :ref:`Allophone Symbol`). An allophone file is a text file that lists allophone (not allophone states!) a strings, one per line. The line number corresponds to an integer (starting with zero), such that the allophone file can be understood as a symbol map from indices to strings to be used when reading :ref:`alignments <Alignment>`.

The total number of allophone states is very large, this is why it's useful to restrict the map. The allophones are configured via the config selector ``allophones`` (see ``Am::AllophoneAlphabet`` in ``src/Am/ClassicStateModel.hh``). The main parameters are

* ``add-from-file`` - read allophone map from this allophone file
* ``add-from-lexicon`` - index all allophones occurring in the current lexicon
* ``add-all`` - index all possible allophones
* ``store-to-file`` - before destruction, the allophone alphabet dumps itself to this file

When using add-from-lexicon, the order of the allophones corresponds to the order in which they occur in the lexicon when reading lemmas sequentially top to bottom. This means that re-ordering the pronunciations in the lexicon would change the indices such that reading from existing alignments will result in garbage. It is therefore best practice to 

1. always keep a compatible allophone file next the alignments
2. when extending the training lexicon, only append lemmas at the end
3. when eyeballing an existing alignment, make sure the neighboring frames have correct triphone context (errors indicate a mismatch in allophone file)

Please note that no allophones are needed during recognition, such that both ``add-from-lexicon`` and ``add-all`` should be set to false.

Cache
^^^^^
See :ref:`Archive`

RASR uses its own data structures for features, alignments, lattices etc. The usual storage format is an archive (sometimes also called cache), which is a binary format that supports compression. The available formats inherit from Core::Archive (class names = file names in src/Core)

* ``Core::BundleArchive`` - see :ref:`Bundle Archive`
* ``Core::DirectoryArchive`` - rarely used
* ``Core::FileArchive`` - used for features, alignments, lattices etc.

A ``FileArchive`` can be considered a "tarball" that can hold multiple independent files. Its format is specified in ``src/Core/FileArchive.cc`` and can be read using the :ref:`Archiver Tool` or `SprintCache.py <https://github.com/rwth-i6/returnn/blob/master/SprintCache.py>`_. See also ``src/Tools/Archiver/Archiver.cc`` for example usage. The low level I/O is implemented in ``Core::BinaryStream``.

The zlib compression is specified in ``src/Core/Archive.cc``

Channel
^^^^^^^

See :ref:`Channel`

Configuration
^^^^^^^^^^^^^

See :ref:`Configuration`
The configuration mechanism is instantiated in ``Core::Application`` when the c'tor calls ``getConfig()`` and creates a static instance that is unique to the whole application. 

Any class can inherit from ``Core::Configurable`` in order to get access to the configuration mechanism. This will ensure automatic creation of the selection hierarchy and passing of the configuration object. The creation hierarchy defines the sequence of selectors. For this, a ``Configurable`` object stores its corresponding ``Configuration`` object as member ``config``. In order to access its values, it has to 

1. declare a private ``static const`` variable of type ``Core::Parameter`` (``src/Core/Parameter.hh`` provides a lot of default parameter types) and give it a name and a short description (and possibly a default value)
2. call the parameter's ``operator()`` and pass it the ``config`` member to obtain the resolved value.

``Core::Component`` is a ``Core::Configurable`` with default logging facilities (:ref:`Channels <Channel>`) - log, warning, error, critical. In particular, a ``Core::Application`` is a Component. Thus, for logging, one can intuitively call ``log()``, ``warning()``, ``error()`` or ``criticalError()`` from every Component. It is possible to delay errors by ignoring and responding later. XML logs support time stamps (for each event) in different format using the parameter ``log-timing`` which can take on the values

* ``no`` - default, no time stamps
* ``yes`` - ``strftime()`` format ``"%Y-%m-%d %H:%M:%S"`` + milliseconds
* ``unix``-time - seconds since epoch
* ``milliseconds`` - milliseconds since epoch

Config files support simple :ref:`arithmetic <Arithmetic Expressions>` syntax based on `GNU Bison <https://www.gnu.org/software/bison/>`_. The support is included in ``src/Core/ArithmeticExpressionParser.hh``.

From the usage site, it's not necessary to do anything about the config objects. Just inherit from Configurable/Component and pass ``config`` object

* to ``operator()`` of ``Core::Parameter`` to read values
* to the constructor of the instantiated sub-objects to maintain the selection hierarchy

Corpus
^^^^^^

See :ref:`Bliss Corpus` and :ref:`Corpus Configuration`
A corpus is an XML file that contains information required by the :ref:`Corpus Visitor`: sequence of recordings holding a sequence of segments with meta-information like start and end time stamps. It is rarely needed to operate on ``Bliss::Corpus`` objects directly (defined in ``src/Bliss/CorpusDescription.hh``). Instead, the access is implemented by means of the visitor pattern, so that the user only has to implement handlers for dealing e.g. with "visitSegment" or "enterRecording" events. 

The :ref:`corpus configuration <Corpus Configuration>` supports sophisticated mechanisms for parallelization (by automatic partitioning or segment lists) and segment ordering.

Corpus visitor
^^^^^^^^^^^^^^

Corpus visitor is a fundamental pattern in most RASR applications. This is innate to data driven processing, since many operations are linear in time and require a pass over the data. In particular, accumulation of statistics (e.g. for LDA, CART, or GMM training), forced alignment, recognition, lattice post-processing in the Flf network and many other operations that require one pass over the corpus are implemented via the `visitor pattern <https://en.wikipedia.org/wiki/Visitor_pattern visitor pattern>`_. This has the benefit of not having to care about the structure of the corpus file (which supports nesting via <include> tags) or the parallelization and segment ordering parameters.

``Bliss::CorpusDescription`` can be thought of as "configured corpus" after all partitioning and segment ordering settings have been set up.

``Speech::CorpusProcessor`` is the base class for visiting algorithms. It offers a channel ``real-time-factor`` to measure the processing time for each segment. A processor needs to sign on to a ``Speech::CorpusVisitor``. Any number of Processors can ``singOn()`` to a single Visitor.  ``Speech::AlignedFeatureProcessor`` is another interface to a Processor (not inherited from ``Speech::CorpusProcessor``) that is better suited for accessing labeled features.

``Speech::CorpusVisitor`` inherits from ``Bliss::CorpusVisitor`` and offers some data structures relevant for speech processing.

In summary, the user has to

1. create a ``Speech::CorpusVisitor`` **v**
2. create any number of ``CorpusProcessor`` and sign them on to **v** by calling ``p.signOn(v)``
3. create a ``Bliss::CorpusDescription`` **d** which is configured via the selector ``"corpus"``
4. let **d** accept the visitor **v** by calling ``d.accept(v)``

Now in order to implement some data processing algorithms, a user has to implement their own ``Speech::CorpusProcessor`` by inheritance and apply the scheme outlined above. See a plethora of examples:

* feature extraction processor: ``Speech::DataExtractor`` that manages ``Speech::DataSource`` (and Flow)
* speech recognition: ``Speech::OfflineRecognizer`` inherits from ``Speech::FeatureExtractor`` which again is a ``Speech::DataExtractor``
* estimate mean of the features: ``Speech::MeanEstimator`` inherits from ``Speech::FeatureExtractor``, as above
* forced alignment: ``Speech::AcousticModelTrainer`` inherits from ``Speech::AlignedFeatureProcessor`` and calls ``processAlignedFeature()`` for each tuple (x_t, s_t) in the data.
* GMM training: ``Speech::TextDependentMixtureSetTrainer`` inherits from ``Speech::AlignedFeatureProcessor``, as above
* Flf lattice processing: ``Flf::CorpusProcessor`` inherits from ``Speech::CorpusProcessor``
* and many more

Decoder
^^^^^^^

The HMM decoder is the actual speech recognizer. See ``Speech::OfflineRecognizer`` (which is a `Speech::CorpusProcessor <Corpus visitor>`) as usage example:

1. ``Speech::OfflineRecognizer`` c'tor calls 

   * ``createRecognizer()``: instantiate ``Search::SearchAlgorithm``
   * ``initializeRecognizer()``: create a ``Speech::ModelCombination`` that combines AM, LM and the lexicon and registers with the ``SearchAlgorithm``

2. as a ``CorpusProcessor``, ``Speech::OfflineRecognizer`` implements ``leaveSpeechSegment()`` to handle a fully ingested segment
3. inside ``Speech::OfflineRecognizer::leaveSpeechSegment()``, in a loop over the buffered feature vectors, the acoustic scores are obtained from the AM and passed to ``Search::SearchAlgorithm::feed()`` that takes care of HMM decoding
4. finally, ``leaveSpeechSegment()`` calls ``Search::SearchAlgorithm::getCurrentBestSentence()`` to obtain a traceback which is a ``std::vector`` of lemmas on the best path.
5. additionally, we can call ``Search::SearchAlgorithm::getCurrentWordLattice()`` to get a lattice

An online recognizer cannot follow the same scheme as there is no corpus (yet) and the input features have to be fed to the SearchAlgorithm continuously. Also, we have to check ``getCurrentBestSentence()`` continuously to update the running hypothesis. But the principle of getting the acoustic scores from the AM and feeding to the SearchAlgorithm is the same.

``SearchAlgorithm`` is mostly ``Search::WordConditionedTreeSearch`` (old) or ``Search::AdvancedTreeSearchManager`` (new), but it might be easier to first read through ``Search::LinearSearch`` which is a very simple and naive implementation. Its ``feed()`` function illustrates how the HMM states are expanded using the acoustic scores and the LM scores are applied at word ends; ``getCurrentBestSentence()`` illustrates back tracing.

General search procedure consists of repeatedly starting new search networks based on previous word end hypotheses (initially there is only one fake starting word at the beginning). Then HMM state expansion is done within each search network by applying scores from different models. Then score-based and histogram-based pruning are applied to all state hypotheses. After that, possible word end hypotheses are detected whenever we reach the last the state of a path in the tree. This leads to exiting the tree as a word end hypothesis with LM score added. Then score-based and histogram-based pruning are applied to all word end hypotheses. Word end hypotheses that have survived pruning then spawn new trees in the next frame. This procedure is repeated until the last frame and final decision can be made based on the final score.

**See also**

* Chapter 1 in `David Nolden's PhD thesis <https://www-i6.informatik.rwth-aachen.de/publications/download/1059/Nolden--2017.pdf>`_

Flf network
^^^^^^^^^^^

FLF = Flexible Lattice processing Framework

An Flf network is a data processing network, mainly used for lattices, but the :ref:`Flf nodes` support arbitrary data types. Similar to :ref:`Flow`, it's a directed acyclic computational graph defined by accessible nodes. The input nodes are specified via ``*.network.initial-nodes``. All node links must be connected to some successor nodes or the sink (a virtual end node). Each node can have multiple input/output ports, enumerated starting with zero. The link syntax is ``$output_port->$target_node:$target_port``.

The :ref:`Flf-Tool` creates a single ``Flf::Network`` and a single ``Flf::Processor`` that is associated with the network. The network is the executed by calling ``processor.run()`` and ``processor.finalize()``. Internally, a ``NetworkCrawler`` will take care of traversing the nodes in the topological order by pulling, starting from the network's final nodes (typically a sink). Because the sink ports are "typed", they will call ``requestLattice()`` (or other ``request*()`` functions) and issue a call to ``sendLattice()`` (or other ``send*()`` functions) of the predecessor nodes. Thus, in order to implement an flf node, you can e.g.

* inherit from ``Flf::Node`` and override ``sendLattice()`` or other ``send*()`` methods for other data types
* inherit from ``Flf::FilterNode`` and override ``filter()``, which will call ``sendLattice()`` with the return value of ``filter()``

The nodes are implemented in ``src/Flf/*`` by inheriting from some of the generic flf nodes. They are registered in ``src/Flf/NodeRegistration.hh`` and the construction is called via ``Flf::NodeFactory`` during flf network creation. 

One of the most fundamental nodes is :ref:`speech-segment`, as it has no inputs (i.e. "source node") and implements the :ref:`Corpus visitor` pattern by inheriting from ``Speech::CorpusProcessor`` and passing the ``Bliss::SpeechSegment`` information from the corpus to the successor flf nodes.

Another fundamental flf node is the :ref:`recognizer`, which holds ``Flf::Recognizer``, a wrapper around ``Search::SearchAlgorithm``. Upon a call to ``sendLattice()`` it will perform the usual recognition steps and return the output of ``Search::SearchAlgorithm::getCurrentWordLattice()``.

Flow
^^^^

See :ref:`Flow`, :ref:`Feature Extraction`
The Flow network is a pull network, a computational graph that is operated by pulling on the output node and engaging all required input nodes.

In many cases, the feature extraction is triggered by means of ``Speech::DataExtractor`` (which is a ``CorpusProcessor`` that is evaluated during corpus visit). The processor operates by wrapping a ``Speech::DataSource`` that is configured via the selector ``"feature-extraction"`` and calling its function ``getData()``.

``Speech::DataSource`` inherits from ``Flow::DataSource``, which is essentially a ``Flow::Network``.

A Flow network, just like any Flow node, specifies input and output ports. We pull on a network (or a node) via ``getData()`` on a certain port. Flow offers different flavors of nodes (see ``src/Flow/Node.hh``): ``SourceNode`` (no inputs), ``SinkNode`` (no outputs), ``SleeveNode`` (single input and single output). The links between input/output ports maintain a queue that can be operated via ``getData() / putData()`` methods from within a node's ``work()``.

See examples in ``src/Signal/`` that inherit from ``Flow::Node``.

**TODO:** discuss

* data types
* pointers, ownership

Hidden Markov model
^^^^^^^^^^^^^^^^^^^

.. image :: /images/hmm.png

* α language model scale: ``lm.scale``
* β transition probability scale: ``acoustic-model.tdp.scale``
* γ acoustic model scale: ``acoustic-model.mixture-set.scale``
* pronuncation scale: weight the pronuncation scores ``model-combination.pronunciation-scale`` (can be thought of as part of the LM term, since the argmax considers :ref:`lemmas <Lemma>` rather than just orthographies).

The :ref:`acoustic model <Acoustic model>` owns a ``Am::ClassicStateModel`` and a ``Am::ClassicHmmTopologySet``. The latter defines the number of states per phoneme and whether or not the across word modeling is enabled. There is also support for "sub-states", which is an artificial duplication of HMM states:

* with 3 states per phone and ''no'' repetition, the state sequence for a phoneme is ``s1 s2 s3``
* with repetition enabled (``hmm.state-repetitions=2``), the state sequence becomes ``s1 s1' s2 s2' s3 s3'`` with all valid transitions defined as usual, but

  * ``p(x|si') := p(x|si)  for i = 1, 2, 3``
  * no new emission classes are introduced

Please note how "repetition" is a misnomer, because ``hmm.state-repetitions=1`` means "only one state instance, no additional repetition".

Also, the AM owns a ``Am::TransitionModel``, which holds a vector of ``Am::StateTransitionModel``, one for each state (depending on the value of ``hmm.tying-type``). The transition model can execute ``TransitionModel::apply()`` on an FST to add loop and skip
transitions to a "flat" automaton (meaning that it does not
contain loops and skips).

The emission labels, i.e. the labels
that are repeated or skipped are on the input side of the
automaton, while the output labels will be unchanged.

This can be viewed as a specialized compose algorithm for the
time-distortion transducer (left) and a given automaton (right).
If you read on, you will discover that considerable care must be
taken to creating compact results.

How it works: The state space is expanded so that we remember the
most recent emission label, this is called "left state" in the
following.  "Right state" refers to the corresponding state in the
original automaton.  This expansion is necessary to provide the
loop transitions.  The representation of the left state is rather
verbose.  It consists of a mask stating which kinds of transition
are possible, a reference to the state's transition model, and of
course the most recent emission label.  In fact only a small number
of combinations of the possible values are actually used.  (One
could slim down the data structure to represent only the valid
combinations.  However priority was given to clarity and
maintainability of the code, over the small increase in efficiency.)
The function isStateLegitimate() specifies which potential states
can be used.  It is good to make these constraints as tight as
possible in order to ensure the result automaton does not contain
unnecessary states.

The most recent emission label may be empty (Fsa::Epsilon).  We
call this a "discharged" state.  This happens for three reasons: 1)
At the word start no emission label has been consumed.  2) After
processing an input epsilon arc.  3) In some situations we
deliberately forget the emission label (see below).

In the expanded state space, loop transitions are simple to
implement.  (In "discharged" states they are not allowed.)
Concerning the other (forward, exit and skip) transitions, there is
a little twist: When a right state has multiple incoming and
outgoing arcs, we choose to first discharge the recent-most
emission label by going to an appropriate left state via an epsilon
transition.  The alternative would be to avoid the epsilon
transition and directly connect to all successor states.  However,
in practice this would dramatically increase the total number of
arcs needed.  So discharging is the preferable alternative.  The
discharged state can be thought of as the state when we have
decided to leave the current state, but not yet chosen where to go
to.  As mentioned before, the appropriate set of transition weights
is recorded, so that we know what to do when we forward or skip
from the discharged state.

Concerning skips: In general, a skip consists of two transitions:
First an epsilon transition goes to an intermediate state that
allows a forward only, and then another transition leads to the
target state.  In "favorable" situations this is optimized into a
single transitions (skip optimization).  If you have read so far,
you are certainly able to figure out what these favorable
conditions are.

As you noticed, there is some freedom in designing the discharge
transitions.  It turns out that compact results can be obtained by
combining the forward discharge with the intermediate skip states,
and to combine exit and loop discharge states.

Any disambiguator label is interpreted as a word boundary and is
given the following special treatment: No loop transition, since
the word boundary cannot be repeated.  No skip transitions: The word
boundary cannot be skipped and the final state before the boundary
cannot be skipped.  The latter is done for consistency with
WordConditionedTreeSearch.

Once the state space is constructed as describe above, it is
relatively straight forward to figure out, which transition weight
(aka time distortion penalty or TDP) must be applied to which arc.
Unfortunately the current scheme is not able to distinguish phone-1
from phone-2 states.  This will require additional state space
expansion by counting repetitive emission labels.  Alternatively,
and probably simpler, we change the labels to allow the distinction
between phone-1 and phone-2.

Language model
^^^^^^^^^^^^^^

The decoder will communicate to the LM from ``src/Search/AdvancedTreeSearch/SearchSpace.cc`` through the interface of ``Lm::ScaledLanguageModel``.

The object is created via ``Core::Ref<Lm::ScaledLanguageModel> lm_ = Lm::Module::instance().createScaledLanguageModel(select("lm"), lexicon_);``

and linked to a lexicon, because from the decoder point of view, the basic unit is a lemma, which combines an orthography and a pronunciation variant. A scaled LM wraps an LM (e.g. ARPA, FST or RNN) and applies a language model scale.

The usage inside the decoder typically consists of 
* creating an ``Lm::History`` via call to ``startHistory()``
* expanding the history by new words via ``extendHistoryByLemmaPronunciation()``
* obtaining LM scores via ``Lm::LanguageModel::score()`` or ``addLemmaPronunciationScore()``

Lattice
^^^^^^^

A lattice can be created by the decoder and processed in various ways by different algorithms. A typical lattice is a ``Lattice::StandardWordLattice``, which holds two independent ``Fsa::StaticAutomaton`` with identical state topology but different weights (AM and LM scores). It also holds a ``Lattice::WordBoundaries`` that contains time information associated with nodes. The constructor requires a lexicon, because the input alphabet of the FSAs is ``Bliss::LemmaPronunciationAlphabet`` (remember, we operate on ''lemmas''). Both FSAs are acyclic acceptors (i.e. no transducers, i.e. there is only one symbol on the edges). The lattice is populated via ``newState()`` and ``newArc()`` methods as well as ``setWordBoundaries()``.

There are several different interfaces that can be converted back and forth:

* :ref:`Flf lattices <Flf network>` inherit directly from ``Ftl::Automaton<Semiring, State>``, which enables lazy evaluation
* :doc:`tool/speech_recognizer.rst` and :ref:`LatticeProcessor` operate on ``Lattice::WordLattice`` (or ``Lattice::StandardWordLattice``), which are based on ``Fsa::Automaton``, which again inherit from ``Ftl::Automaton<Fsa::Semiring>``. These are compatible with the `RWTH FSA toolkit <https://www-i6.informatik.rwth-aachen.de/~kanthak/fsa.html>`_.

Reading and writing of `HTK lattice format <https://labrosa.ee.columbia.edu/doc/HTKBook21/node257.html>`_ is supported via :ref:`flf nodes (archive-reader and archive-writer) <archive-reader>`.

Lemma
^^^^^

A ``Bliss::Lemma`` (see ``src/Bliss/Lexicon.hh``) is an entry in the lexicon. Its combines four levels of representation: orthographies, pronunciations, evaluation tokens and syntactic tokens.

Lexicon
^^^^^^^

Lexicon class reflects the XML lexicon file given by the config. All unique pronunciations are stored in a container and the same for all lemmas and synt-tokens. Additionally we have ``Bliss::LemmaPronunication`` class for each pronunciation of each lemma, which  stores the pointer to its corresponding pronunciation and lemma. Each ``LemmaPronunciation`` is essentially an word end exit of the state tree (search network). Each lemma stores its synt-tokens for LM scoring.

Logging
^^^^^^^

Logging is implemented via :ref:`channels <Channel>`. Please avoid writing directly to stdout/stderr.

Individual components are free to provide own channels to offer different pieces of information, relevant e.g. for different levels of logging. The default **global** facilities (info, log, warning and error) are available via a pointer obtained by a call to ``Core::Application::us()`` from every application, e.g. ``Core::Application::us()->warning("cannot read file '%s'", f.c_str());`` This is meant for high level messages relevant for the application.

Alternatively, any class derived from ``Core::Component`` can call its own, **local** facilities via ``this->log("all your base are belong to us");``

It can be convenient to use a ``sprintf`` wrapper for ``std::string`` available in ``Core::form()``.

We recommend using local logging facilities if possible, which can be controlled for each component separately, e.g.:

.. code-block :: ini

    *.feature-extraction.*.info.channel  = log/feat-ex.log
    *.lm.*.warning.channel               = nil
    *.recognizer.*.warning.channel       = stderr

Memory mapped archives
^^^^^^^^^^^^^^^^^^^^^^

``src/Core/MappedArchive.hh`` provides an interface to memory mapped files via ``MappedArchiveReader`` and ``MappedArchiveWriter``. These are used for storing data structures in a ready-to-use binary format. Most prominently, the state tree and other search relevant data structures are stored in a global cache, which is a MappedArchive. An application can hold multiple caches, and each cache can store multiple objects. The caller requests a reader/writer via ``MappedArchiveReader Application::getCacheArchiveReader(const std::string& archive, const std::string& item)`` and accesses the ``std::istream/std::ostream`` interface, e.g.

.. code-block :: cpp

    MappedArchiveReader in = Core::Application::us()->getCacheArchiveReader("global-cache", "state-network-image");
    if (in.good()) {
        int storedTransformation = 0;
        in >> storedTransformation;
        // ...
     }

Precursor
^^^^^^^^^

Often you see the following pattern in the code:

.. code-block :: cpp

    class Foo : public Bar {
        typedef Bar Precursor;
        
        Foo() : Precursor() {}
        void f() {
            Precursor::g();
        }
    }

This is meant to increase flexibility when changing the parent class: the string "Bar" only needs to be modified in the first two lines.

Reference counted objects
^^^^^^^^^^^^^^^^^^^^^^^^^

**TODO** discuss

* inheriting ``Core::ReferenceCounted``
* ``Core::ref(new Foo())`` 
* ``Core::Ref<Foo>(new Foo())``
* convention: typdef name ending in "Ref", e.g. ``typedef Core::Ref<const Phonology> ConstPhonologyRef;``

Singleton
^^^^^^^^^

The singleton pattern is implemented in ``src/Core/Singleton.hh`` and used to hold a unique static instance of an object. It is used in RASR as part of the module structure: each module provides an interface with factories for creating various objects depending on run time configuration. An application instantiates relevant singleton module objects via ``INIT_MODULE(Foo)`` macro called in a ``Core::Application``'s constructor.

The caller gets access to the wrapped singleton object via ``SingletonHolder::instance()``, e.g. ``Foo::Module::instance()``.

Another example of singleton pattern in RASR is a pointer to ``Core::Application``, available via ``Core::Application::us()`` from every application in order to access the logging facilities. This pointer is not wrapped by a SingletonHolder for simplicity.

State model
^^^^^^^^^^^
TODO

State tree
^^^^^^^^^^

State tree is an HMM state network (search network) constructed completely based on the lexicon. All pronunciations are converted to HMM state sequence and inserted into the tree with prefix sharing. The last state leads to word end exit which leads to a ``Bliss::LemmaPronunciation`` object (see ``src/Bliss/Lexicon.hh``). Then corresponding lemma and synt-tokens can be found for LM scoring. For additional across-word modeling (Fan-In/-Out) and pushed word end, please refer to chapters 1-2 of `David Nolden's PhD thesis <https://www-i6.informatik.rwth-aachen.de/publications/download/1059/Nolden--2017.pdf>`_.

State tying (CART, LUT)
^^^^^^^^^^^^^^^^^^^^^^^
TODO

Stream
^^^^^^

**Binary formats**

Binary I/O of large objects is best organized by means of BinaryStream classes, that wrap and mimic ``std::ostream`` and ``std::istream``: 

* BinaryStream
* BinaryOutputStream
* BinaryInputStream

Classes support binary output using streaming operators.
Read and write functions support the io of blocks of data.

BinaryStreams support endianess swapping.  The byte order of the
file can be set in the constructor.  Swapping is carried out if the
native byte order of the architecture is different from one in the
file.  By convention RASR stores binary data in **little endian files**.  Therefore you should specify the byte order in the
constructor only when dealing with non-RASR file formats.

Classes for which binary persistence is required should implement
methods void read() and write() methods accepting ``BinaryInputStream``
or ``BinaryOutputStream``, or global operators for BinaryStreams:

* ``BinaryOutputStream& operator<<(BinaryOutputStream& o, const Class &c)``;
* ``BinaryInputStream& operator>>(BinaryInputStream& i, Class &c)``;


**Text formats**

``Core::XmlWriter`` can be considered an ``std::ostream`` for writing XML.

``Core::TextInputStream`` and ``Core::TextOutputStream`` augment ``std::ostream`` with some convenience features like character set, indentation and word wrapping. A text stream can perform compression on the fly using `zlib <http://zlib.net/>`_, e.g. reading

.. code-block :: cpp

    Core::CompressedInputStream* cis = new Core::CompressedInputStream(gzfilename.c_str());
    Core::TextInputStream        is(cis);
    std::string line;
    while (!std::getline(is, line).eof()) { ... }

or writing

.. code-block :: cpp

    Core::CompressedOutputStream fout(gzfilename);
    if (!fout) fout << "hello world\n";


**Format specifiers**

The file names have only few rules (gzipped input files must end in ".gz", bundle files must end in ".bundle"). A module can specify a ``Core::FormatSet`` and register certain **prefixes** ("format qualifiers") such as "bin", "xml" or "ascii". The prefixes then can be used to specify the input/output format in the configuration, e.g.

.. code-block :: ini

    *.neural-network.parameters-old = bin:/path/to/params
    *.lda.file                      = xml:my.matrix

The corresponding instances of  

* ``Core::CompressedBinaryFormat``
* ``Core::CompressedPlainTextFormat``
* ``Core::BinaryFormat``
* ``Core::PlainTextFormat``
* ``Core::XmlFormat``

wrap the input and output streams discussed in this section, offering read() and write() functions. A user can then switch flexibly between different formats by accessing e.g. ``Application::us()->formatSet()->read("xml:foo.xml", foo)`` and reading transparently from different formats.

XML
^^^

RASR relies a lot on XML format for many plain text resources (lexicon, corpus, Flow networks, small math objects like CMLLR matrices/vectors, logs, CART etc.)

We use `libxml2 <http://xmlsoft.org/ libxml2>`_ to read XML documents with a SAX-style parser. In particular, the [[#Corpus visitor]] is a straight-forward implementation of a SAX handler. The interface is wrapped in ``src/Core/XmlParser.hh``. See example usage in ``src/Bliss/CorpusParser.cc``, ``src/Bliss/LexiconParser.hh`` or ``src/Core/MatrixParser.hh``.

Writing XML can be done through a ``Core::XmlWriter`` which inherits the convenient interface of a ``std::ostream`` including ``operator<<()`` for many different types. Many classes provide
``Core::XmlWriter& operator<<(Core::XmlWriter& os, const T &obj)``
enabling simple serialization to XML (see also ref:`Stream`).

