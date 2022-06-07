Common component configuration
==============================

In RASR many components appear again and again in different tools / context. The configuration parameters for these components are documented here:

Acoustic Model Configuration
----------------------------

Within this section, all common parameters which are part of the acoustic modeling are presented. These parameters are configured using the standard :doc:`configuration` mechanism, e.g. a parameter concerning hidden markov models is configured ``[*.acoustic-model.hmm]``.

Allophone configuration
^^^^^^^^^^^^^^^^^^^^^^^

| ``add-all`` (boolean) : All possible allophones w.r.t the phoneme list from the lexicon are generated and added. 
| ``add-from-lexicon`` (boolean) :  Only those allophones, which actually occur within the given lexicon are generated and added.
| ``store-to-file`` (string) : Write all created allophones to the given file.
| ``add-from-file`` (string) : Read allophones from the given file.

An example for a common allophone configuration would be:

.. code-block:: ini

    [*.allophones]
    add-all = false
    add-from-lexicon = true

State-tying configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

| ``type`` (enum): Specify the type of state tying used (some settings have multiple names) :
| ``none`` / ``no-tying`` : no state tying
| ``monophone`` : no phoneme context
| ``lut`` / ``lookup`` : use a [[state tying file formats#Lookup_Table|lookup table]]
| ``cart`` / ``decision-tree`` : use CART in [[CART_file#New_CART_file_format|new file format]].
| ``file`` (string): state tying definition file

HMM configuration
^^^^^^^^^^^^^^^^^

The configuration of the Hidden Markov Models is divided in two parts: a more general ``hmm`` part, where the structure of the HMM is defined and a ``tdp`` part, where the time distortion penalties for the individual transitions can be modified.

* ``hmm``

    | ``states-per-phone`` (int): We usually model each phone with three states, one for start, middle and end of the phone. The number of states per phone can be set with this parameter
    | ``state-repetitions`` (int): Each state of the phone may be repeated several times. With this parameter, the number of repetitions per state of the phone is set. Usually, the value ``2`` is chosen. With ``3`` states per phone, this results in the commonly used 6 state HMM.
    | ``across-word-model`` (bool): With this parameter, you can chose between across-word modeling (``yes``) or within-word modeling (``no``).
    | ``early-recombination`` (bool) : if you want to have a smaller state tree, you may chose ``yes`` for early recombination. But this may result in worse performance, so ``no`` is the recommended parameter choice.

* ``tdp``
    | ``scale`` (float): The global time distortion penalty scale for the score calculation is set with this parameter.
    | ``*.loop`` (float): The tdp for a loop transition between two phone states. For the silence phoneme, this penalty can be given separately via ``silence.loop``. The penalty for the transition into the first state of a word can be set via the ``entry-m1`` parameter. For compatibility reasons, there also is the parameter ``entry-m2``, which sets the tdp for an incoming transition into the repetition of the first state of a word.
    | ``*.forward`` (float): The tdp for a forward transition between two phone states. For the silence phoneme, this penalty can be given separately via ``silence.forward``. 
    | ``*.skip`` (float): The tdp for a skip transition between two phone states. For the silence phoneme, this penalty can be given separately via ``silence.skip``. 
    | ``*.exit`` (float): For the end of a word, a special exit transition penalty may be given with this parameter. It controls the average length of the recognized words. For the silence phoneme, this penalty can be given separately via ``silence.exit``.


An example HMM configuration may look like this:

.. code-block:: ini

    [*.acoustic-model.hmm]
    states-per-phone                = 3
    state-repetitions               = 2
    across-word-model               = yes
    early-recombination             = no
    
    [*.acoustic-model.tdp]
    scale                                   = 1.0
    
    *.loop                                  = 3.0
    *.forward                               = 0.0
    *.skip                                  = 3.0
    *.exit                                  = 0.0
    
    entry-m1.loop                           = infinity
    entry-m2.loop                           = infinity
    
    silence.loop                            = 0.0
    silence.forward                         = 3.0
    silence.skip                            = infinity
    silence.exit                            = 6.0


Mixture Set
^^^^^^^^^^^

| ``feature-scorer-type`` (enum): There are several feature scorers available, depending on the used RWTH ASR version. Default is the ``SIMD-diagonal-maximum`` feature scorer, which uses a maximum approximation and a diagonal covariance matrix to calculate the scores. Additionally, the Single Instruction Multiple Data (SIMD) technique is used to achieve data level parallelism and thus improve the computing performance.
| ``file`` (string): The location of the acoustic model is given with this parameter. See also :ref:`Mixture File`
| ``scale`` (float): The acoustic model scale is set with this parameter. It is usually set to ``1``.
| ``covariance-tying`` (enum): Type of covariance matrix is used, it can be "pooled-covariance" (default) or "mixture-specific-covariance".
| ``minimum-variance`` (float): Floor variance value (it can be necessary if the number of mixture is very high).
| ``reduced-mixture-set-dimension`` (int): Clip the mean and variance vectors to the given dimension.

An example configuration:

.. code-block:: ini

    [*.mixture-set]
    file                            = mixture.file
    feature-scorer-type             = SIMD-diagonal-maximum
    scale                           = 1.0
    #covariance-tying                = pooled-covariance
    #minimum-variance                = 0.01

Channel Configuration
---------------------
see :doc:`channel`

Corpus Configuration
--------------------

See also :ref:`Bliss Corpus`

| ``audio-dir`` (string): Usually, the path entries within the corpus to the audio-files are relative to a certain path. The main directory to the audio files is given with this parameter.
| ``capitalize-transcriptions`` (bool): It is possible to have the transcriptions mapped to only upper case letters, if you want to perform a case-insensitive recognition. Then, the value of this parameter should be ``yes``, otherwise it should be ``no``.
| ``gemenize-transcriptions`` (bool): Convert transcription to lower case (overwrites capitalize-transcriptions).
| ``file`` (string): The location of the BLISS :ref:`Bliss Corpus` file which should be recognized
| ``partition`` (int) : Divide corpus into partitions with (approximately) equal number of segments.
| ``recording-based-partition`` (bool) : create corpus partitions based on recordings instead of segments
| ``segments.file`` (string) : include only segments in this file (``#`` can be used to comment out segments to skip)
| ``segment-order`` (string): file defining the order of processed segments (one segment identifier per line).
| ``segment-order-look-up-short-name`` (bool): use short names in segment-order file (segment name only)
| ``segments-to-skip`` (string list) : exclude segments in this list (space separated)
| ``select-partition`` (int) : Select partition of the corpus
| ``skip-first-segments`` (int) : skip the first N segments (counted after partitioning)
| ``warn-about-unexpected-elements`` (bool): By default a warning is issued whenever an unknown XML element is ignored or flattened. Use this flag to turn this off.


Note: segment format in ``segments.file`` or ``segments-to-skip`` should contain the full-segment names, i.e. `` <corpusname>/<recordingname>/<segmentname> ``

Example:

.. code-block:: ini

    [*.corpus]
    file                            = /home/corpus/test.corpus.gz
    audio-dir                       = /home/audio
    warn-about-unexpected-elements  = no
    capitalize-transcriptions       = no
    #segments.file                  = segments.debug
    #segments-to-skip               = coretex-de/tagesthemen/1 coretex-de/tagesthemen/2 coretex-de/tagesthemen/3
    </pre>
    
    Example for ``segments.debug`` file:
    <pre>
    coretex-de/tagesthemen/1
    coretex-de/tagesthemen/2
    coretex-de/tagesthemen/3
    # coretex-de/tagesthemen/4


Lexicon Configuration
---------------------

| ``file`` (string): The location of the XML :ref:`Bliss Lexicon` to be used for recognition.
| ``normalize-pronunciation`` (bool) : If there are no pronunciation weights given, a uniform distribution of the weights among all pronunciations of a lemma is enforced by setting this to ``true`` (default).

A typical lexicon configuration:
 
.. code-block:: ini

    [*.lexicon]
    file                            = $(LEXICON)
    normalize-pronunciation         = false


