Tokenizer
=========

Tokenizers convert between text and sequences of subword tokens, e.g. to turn a prompt or a list of words into
label sequences or a traceback back into text. Tokens are available both as strings (e.g. to look them up as lemmas
in a lexicon) and as ids of the tokenizer's own vocabulary.

Common parameters:

type (enum):
    type of the tokenizer, see below. Further types can be registered in the tokenizer factory, e.g. from python.

Encoding out-of-vocabulary tokens (where not mapped to an unknown token) and decoding invalid ids raise an error,
which terminates the program unless configured otherwise via ``on-error``.

BPE
---

tokenizer type: ``bpe``

Byte-pair encoding as applied by `subword-nmt <https://github.com/rsennrich/subword-nmt>`_. Text is split at
whitespace (blanks, tabs and newlines) and all subwords except the last one of a word get the ``separator`` suffix,
e.g. ``lower`` -> ``lo@@ wer``. Detokenizing joins words with single blanks.

**Configuration**

codes-file (string):
    merge operations as written by ``subword-nmt learn-bpe`` (versions 0.1 and 0.2)
vocab-file (string):
    optional vocabulary defining the token ids, either one token per line (id = line index, further columns such as
    the counts written by ``subword-nmt get-vocab`` are ignored) or a RETURNN-style python dict literal as written by
    ``create-py-vocab.py`` of `rwth-i6/subword-nmt <https://github.com/rwth-i6/subword-nmt>`_
separator (string):
    suffix of non-final subwords, default ``@@``
unknown-token (string):
    token that out-of-vocabulary tokens are mapped to when encoding. If not set, encoding them raises an error.

**Example**

.. code-block :: ini

    [*.tokenizer]
    type          = bpe
    codes-file    = bpe.codes
    vocab-file    = bpe.vocab
    unknown-token = <unk>

SentencePiece
-------------

tokenizer type: ``sentencepiece``

Uses a `SentencePiece <https://github.com/google/sentencepiece>`__ model. Requires building with
``-DMODULE_SENTENCEPIECE=ON`` (optionally with ``-DSENTENCEPIECE_ROOT=<installation prefix>``).
Pieces not in the model are encoded as the model's unknown id.

**Configuration**

model-file (string):
    SentencePiece model file, e.g. as written by ``spm_train``

HuggingFace
-----------

tokenizer type: ``huggingface``

Loads a ``tokenizer.json`` file as shipped with most LLMs on the HuggingFace hub. It runs the python package
`tokenizers <https://github.com/huggingface/tokenizers>`_ (``pip install tokenizers``), so it requires
``MODULE_PYTHON`` but works in all RASR tools, not only via ``librasr``.

**Configuration**

tokenizer-file (string):
    HuggingFace ``tokenizer.json`` file
add-special-tokens (bool):
    whether to add special tokens as defined by the tokenizer's post-processor (e.g. BOS) when encoding, default false
skip-special-tokens (bool):
    whether to drop special tokens when decoding, default true

Usage from python
-----------------

``librasr`` can create any configured tokenizer and lets you implement tokenizers in python.

.. code-block :: python

    import librasr

    config = librasr.Configuration()
    config.set("*.tokenizer.type", "bpe")
    config.set("*.tokenizer.codes-file", "bpe.codes")
    config.set("*.tokenizer.vocab-file", "bpe.vocab")

    tokenizer = librasr.create_tokenizer(config)
    tokens = tokenizer.tokenize("the lower tower")  # e.g. ["the", "lo@@", "wer", "tower"]
    ids = tokenizer.encode("the lower tower")
    text = tokenizer.decode(ids)

* ``librasr.create_tokenizer(config)`` : create the tokenizer configured under ``*.tokenizer.*``. To configure
  several tokenizers, pass sub-configs, e.g. ``librasr.create_tokenizer(librasr.Configuration(config, "prompt"))``
  for ``*.prompt.tokenizer.*``.
* ``librasr.Tokenizer`` methods: ``tokenize(text)`` / ``detokenize(tokens)`` convert between text and token strings,
  ``encode(text)`` / ``decode(ids)`` between text and token ids. Further ``vocabulary_size()``, ``token_id(token)``
  (``librasr.Tokenizer.INVALID_TOKEN_ID`` if not in the vocabulary) and ``token(id)``.

To implement a tokenizer in python, subclass ``librasr.Tokenizer`` and register it under a type name. It can then be
configured like the built-in types, also for components that create their tokenizer in C++:

.. code-block :: python

    class CharTokenizer(librasr.Tokenizer):
        def __init__(self, config):
            super().__init__(config)
            self.vocab = list("abcdefghijklmnopqrstuvwxyz ")

        def tokenize(self, text):
            return list(text)

        def detokenize(self, tokens):
            return "".join(tokens)

        def vocabulary_size(self):
            return len(self.vocab)

        def token_id(self, token):
            return self.vocab.index(token) if token in self.vocab else librasr.Tokenizer.INVALID_TOKEN_ID

        def token(self, id):
            return self.vocab[id]

    librasr.register_tokenizer_type("chars", CharTokenizer)

``tokenize``, ``detokenize``, ``vocabulary_size``, ``token_id`` and ``token`` must be implemented. ``encode`` and
``decode`` can be overridden, by default they are based on the other methods. Register the type before creating a
tokenizer that uses it.
