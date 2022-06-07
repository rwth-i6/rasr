LM Util
=======

The lm-util binary performns various language model related tasks. The action to perform is passed via the ``action`` parameter.
Valid actions are:

* ``load-lm`` : Loads and initializes a language model. Can be usefull for LMs where initializing the model has side-effects,
  like the ARPA LM, which creates an image file.

  * Sub-components:
     | ``lexicon``     : lexicon for the languagemodel
     | ``lm``          : configuration for the :doc:`../language_model`

* ``compute-perplexity-from-text-file`` : Compute the perplexity of a LM on a text file.

  * Parameters:
     | ``file``        : input file
     | ``encoding``    : the encoding of the input file
     | ``score-file``  : output path for word scores
     | ``batch-size``  : number of sequences to process in one batch
     | ``renormalize`` : wether to renormalize the word probabiliies
  * Sub-components:
     | ``lexicon``     : lexicon for the languagemodel
     | ``lm``          : configuration for the :doc:`../language_model`

**Example call**::

    lm-util \
    --*.action=compute-perplexity-from-text-file    \
    --*.lm.type=ARPA \
    --*.lm.image=models/lm.image \
    --*.lexicon.file=models/recognition.lex.xml.gz \
    --file=my-text-file \
    --score-file=scores.txt
