Costa
=====

Costa provides statistics about corpora, lexicons, and language models.

**Parameters**:

| ``evaluate-recordings`` : feed the audio files specified in the corpus to the flow network specified by feature-extraction. This is useful if you want to check that all audio files are present and readable. 
| ``lexical-statistics``  : print statistics about the lexicon 
| ``lm-statistics``       : print statistics about the language model 
| ``feature-extraction``  : see feature extraction 
| ``corpus``              : see :ref:`Common component configuration <Corpus Configuration>`
| ``lexicon``             : see :ref:`Common component configuration <Lexicon Configuration>`
| ``lm``                  : see :ref:`SpeechRecognizer`

**Channels**:

| ``recordings``   : recording names
| ``lexicon.dump`` : dump the lexicon 

**Example Configuration** ::

    [*]
    evaluate-recordings             = yes
    lexical-statistics              = yes
    lm-statistics                   = no
    
    # define the lexicon used
    [*.lexicon]
    file                            = ../lexicon/training.lexicon.gz
    
    # do not load the language model, just simple zerogram statistics
    [*.lm]
    type                            = zerogram
    
    # define the corpus
    [*.corpus]
    file                            = corpus.gz
    audio-dir                       = /path/to/audio/files
    
    # Flow file with a single source node to open and scan the audio file
    [*.feature-extraction]
    file                            = config/test-audio.flow
    
    # Channels
    [*]
    log.channel                     = log-channel
    warning.channel                 = log-channel, stderr
    error.channel                   = log-channel, stderr
    statistics.channel              = log-channel
    configuration.channel           = log-channel
    unbuffered                      = true
    log-channel.file                = $(log-file)

