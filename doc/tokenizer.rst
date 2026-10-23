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

Word boundaries
---------------

Each token tells whether it marks a word boundary before and after itself. A word boundary lies between two
consecutive tokens if the first one marks it after itself or the second one before itself. The start and end of a
token sequence are always word boundaries. Most tokenizers only mark one side, so a token that does not mark a boundary
does not mean that there is none:

* BPE: tokens without the ``separator`` suffix mark a boundary after themselves, e.g. ``wer`` in ``lo@@ wer``.
* SentencePiece: pieces with a leading ``▁`` mark a boundary before themselves. Control pieces such as ``<s>`` and the
  unknown piece are words of their own.
* HuggingFace: determined from the decoder in ``tokenizer.json``. Tokens with a leading ``Ġ`` (byte-level BPE, also
  other whitespace such as newlines) or ``▁`` (``Metaspace``) and tokens without the ``##`` prefix of ``WordPiece``
  mark a boundary before themselves, tokens with the ``</w>`` suffix of a ``BPEDecoder`` one after themselves. Special
  tokens are words of their own. Other decoders are not supported.

With byte-level BPE, a word on its own is tokenized differently than inside a text, e.g. ``the`` and ``Ġthe``.

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
    RETURNN-style vocabulary defining the token ids, i.e. a python dict literal ``{'<s>': 0, 'lo@@': 1, ...}``
    as written by ``create-py-vocab.py``
separator (string):
    suffix of non-final subwords, default ``@@``
unknown-token (string):
    token that out-of-vocabulary tokens are mapped to when encoding. If not set, encoding them raises an error.

**Example**

.. code-block :: ini

    [*.tokenizer]
    type          = bpe
    codes-file    = bpe.codes
    vocab-file    = bpe.vocab.py
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
    config.set("lib-rasr.tokenizer.vocab-file", "bpe.vocab.py")

    tokenizer: librasr.Tokenizer = librasr.create_tokenizer(config)
    tokens = tokenizer.tokenize("the lower tower")  # e.g. ["the", "lo@@", "wer", "tower"]

* ``librasr.create_tokenizer(config)`` : create the tokenizer configured under ``lib-rasr.tokenizer``.
* ``librasr.Tokenizer`` methods: ``tokenize(text)`` / ``detokenize(tokens)`` convert between text and token strings,
  ``encode(text)`` / ``decode(ids)`` between text and token ids. Further ``vocabulary_size()``, ``token_str_to_id(token)``
  (``librasr.Tokenizer.INVALID_TOKEN_ID`` if not in the vocabulary), ``token_id_to_str(id)`` and ``word_boundary_marks(id)``,
  which returns a ``librasr.Tokenizer.WordBoundaryMarks`` with the attributes ``before`` and ``after``.

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

        def word_boundary_marks(self, id: int) -> librasr.Tokenizer.WordBoundaryMarks:
            blank = self.vocab[id] == " "  # blanks are words of their own
            return librasr.Tokenizer.WordBoundaryMarks(before=blank, after=blank)

    librasr.register_tokenizer_type("chars", CharTokenizer)

    config = librasr.Configuration()
    config.set("lib-rasr.tokenizer.type", "chars")

    tokenizer = librasr.create_tokenizer(config)  # an instance of CharTokenizer

    tokens = tokenizer.tokenize("hello world")  # ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
    text = tokenizer.detokenize(tokens)  # "hello world"

    ids = tokenizer.encode("hello world")  # [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3]
    text = tokenizer.decode(ids)  # "hello world"

``tokenize``, ``detokenize``, ``vocabulary_size``, ``token_str_to_id``, ``token_id_to_str`` and ``word_boundary_marks`` must
be implemented.
``encode`` and ``decode`` can be overridden, by default they are based on the other methods. Register the type before
creating a tokenizer that uses it.
