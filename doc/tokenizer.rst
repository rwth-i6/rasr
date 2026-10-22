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

Byte-pair encoding as applied by `subword-nmt <https://github.com/rwth-i6/subword-nmt>`_. Text is split at
whitespace (as python's ``str.split()``, e.g. blanks, tabs, newlines and non-breaking spaces) and all subwords except
the last one of a word get the ``separator`` suffix, e.g. ``lower`` -> ``lo@@ wer``. Detokenizing joins words with
single blanks.

**Configuration**

codes-file (string):
    merge operations as written by ``learn_bpe.py`` (versions 0.1 and 0.2)
vocab-file (string):
    optional vocabulary defining the token ids, either one token per line (id = line index, further columns such as
    the counts written by ``get_vocab.py`` are ignored) or a RETURNN-style python dict literal as written by
    ``create-py-vocab.py``
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
``MODULE_PYTHON``.

**Configuration**

tokenizer-file (string):
    HuggingFace ``tokenizer.json`` file
add-special-tokens-on-encode (bool):
    whether ``tokenize`` and ``encode`` insert the special tokens defined by the post-processor of the tokenizer,
    e.g. ``"hello"`` -> ``["<s>", "hello"]`` for a tokenizer that prepends a BOS token. Default false.
remove-special-tokens-on-decode (bool):
    whether ``decode`` and ``detokenize`` drop special tokens from the text, e.g. ``["<s>", "hello", "</s>"]`` ->
    ``"hello"`` instead of ``"<s>hello</s>"``. Default true.

Usage from python
-----------------

``librasr`` can create any configured tokenizer and lets you implement custom tokenizers in python.

.. code-block :: python

    import librasr

    config = librasr.Configuration()
    config.set("lib-rasr.tokenizer.type", "bpe")
    config.set("lib-rasr.tokenizer.codes-file", "bpe.codes")
    config.set("lib-rasr.tokenizer.vocab-file", "bpe.vocab")

    tokenizer: librasr.Tokenizer = librasr.create_tokenizer(config)
    tokens = tokenizer.tokenize("the lower tower")  # e.g. ["the", "lo@@", "wer", "tower"]

* ``librasr.create_tokenizer(config)`` : create the tokenizer configured under ``lib-rasr.tokenizer``.
* ``librasr.Tokenizer`` methods: ``tokenize(text)`` / ``detokenize(tokens)`` convert between text and token strings,
  ``encode(text)`` / ``decode(ids)`` between text and token ids. Further ``vocabulary_size()``, ``token_str_to_id(token)``
  (``librasr.Tokenizer.INVALID_TOKEN_ID`` if not in the vocabulary) and ``token_id_to_str(id)``.

To implement a tokenizer in python, subclass ``librasr.Tokenizer`` and register it under a type name. It can then be
configured like the built-in types, also for components that create their tokenizer in C++:

.. code-block :: python

    import librasr

    class CharTokenizer(librasr.Tokenizer):
        def __init__(self, config: librasr._BaseConfig) -> None:
            super().__init__(config)
            self.vocab = list("abcdefghijklmnopqrstuvwxyz ")

        def tokenize(self, text: str) -> list[str]:
            return list(text)

        def detokenize(self, tokens: list[str]) -> str:
            return "".join(tokens)

        def vocabulary_size(self) -> int:
            return len(self.vocab)

        def token_str_to_id(self, token: str) -> int:
            return self.vocab.index(token) if token in self.vocab else librasr.Tokenizer.INVALID_TOKEN_ID

        def token_id_to_str(self, id: int) -> str:
            return self.vocab[id]

    librasr.register_tokenizer_type("chars", CharTokenizer)

    config = librasr.Configuration()
    config.set("lib-rasr.tokenizer.type", "chars")

    tokenizer = librasr.create_tokenizer(config)  # an instance of CharTokenizer

    tokens = tokenizer.tokenize("hello world")  # ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
    text = detokenize(tokens)  # "hello world"

    ids = tokenizer.encode("hello world")  # [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]
    text = tokenizer.decode(ids)  # "hello world"

``tokenize``, ``detokenize``, ``vocabulary_size``, ``token_str_to_id`` and ``token_id_to_str`` must be implemented.
``encode`` and ``decode`` can be overridden, by default they are based on the other methods. Register the type before
creating a tokenizer that uses it.
