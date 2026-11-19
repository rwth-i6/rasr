"""
Tests for the tokenizer bindings of librasr. Requires librasr to be importable, e.g. via PYTHONPATH.

Run with: python3 src/Test/LibRASR_Tokenizer.py
"""

import importlib.util
import unittest
from pathlib import Path

import librasr

DATA_DIR = Path(__file__).resolve().parent / "data" / "tokenizer"
TEXT = "the newest lower tower is wider than straße xy"


def make_config(**params: str) -> librasr.Configuration:
    config = librasr.Configuration()
    config.set("*.log.channel", "nil")
    for key, value in params.items():
        config.set(key, value)
    return config


class CharTokenizer(librasr.Tokenizer):
    """Character-level tokenizer which only implements the abstract methods."""

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

    # Blanks are words of their own
    def word_boundary_marks(self, id: int) -> librasr.Tokenizer.WordBoundaryMarks:
        blank = self.vocab[id] == " "
        return librasr.Tokenizer.WordBoundaryMarks(before=blank, after=blank)


class UpperCaseTokenizer(CharTokenizer):
    """Additionally overrides `decode`."""

    def decode(self, ids: list[int]) -> str:
        return super().decode(ids).upper()


librasr.register_tokenizer_type("test-chars", CharTokenizer)
librasr.register_tokenizer_type("test-upper-case", UpperCaseTokenizer)


class ConfiguredTokenizerTest(unittest.TestCase):
    """Tokenizers configured in RASR and created via the tokenizer factory."""

    def test_bpe(self) -> None:
        config = make_config(
            **{
                "lib-rasr.tokenizer.type": "bpe",
                "lib-rasr.tokenizer.codes-file": str(DATA_DIR / "bpe.codes"),
                "lib-rasr.tokenizer.vocab-file": str(DATA_DIR / "bpe.vocab.py"),
                "lib-rasr.tokenizer.unknown-token": "<unk>",
            }
        )
        tokenizer = librasr.create_tokenizer(config)
        tokens = tokenizer.tokenize(TEXT)
        self.assertEqual(" ".join(tokens), "the newest lower tower is wide@@ r than s@@ t@@ r@@ a@@ ß@@ e x@@ y")
        self.assertEqual(tokenizer.detokenize(tokens), TEXT)
        self.assertEqual(tokenizer.vocabulary_size(), 34)
        self.assertEqual(tokenizer.token_str_to_id("tower"), 3)
        self.assertEqual(tokenizer.token_str_to_id("x@@"), librasr.Tokenizer.INVALID_TOKEN_ID)
        self.assertEqual(tokenizer.token_id_to_str(3), "tower")
        self.assertEqual(tokenizer.encode(TEXT), [2, 8, 7, 3, 4, 30, 5, 9, 15, 29, 28, 19, 31, 24, 1, 1])
        boundaries = tokenizer.word_boundary_marks(tokenizer.token_str_to_id("wide@@"))
        self.assertEqual((boundaries.before, boundaries.after), (False, False))
        self.assertTrue(tokenizer.word_boundary_marks(tokenizer.token_str_to_id("r")).after)
        self.assertEqual(tokenizer.decode([2, 8, 3, 1]), "the newest tower <unk>")

    def test_sub_config(self) -> None:
        config = make_config(
            **{
                "lib-rasr.prompt.tokenizer.type": "bpe",
                "lib-rasr.prompt.tokenizer.codes-file": str(DATA_DIR / "bpe.codes"),
                "lib-rasr.prompt.tokenizer.vocab-file": str(DATA_DIR / "bpe.vocab.py"),
            }
        )
        tokenizer = librasr.create_tokenizer(librasr.Configuration(config, "prompt"))
        self.assertEqual(tokenizer.tokenize("wider"), ["wide@@", "r"])

    @unittest.skipIf(importlib.util.find_spec("tokenizers") is None, "python package 'tokenizers' is not available")
    def test_huggingface(self) -> None:
        config = make_config(
            **{
                "lib-rasr.tokenizer.type": "huggingface",
                "lib-rasr.tokenizer.tokenizer-file": str(DATA_DIR / "tokenizer.json"),
                "lib-rasr.tokenizer.add-special-tokens-on-encode": "true",
            }
        )
        tokenizer = librasr.create_tokenizer(config)
        ids = tokenizer.encode("the tower")
        self.assertEqual(ids, [0, 277, 266])
        self.assertEqual(tokenizer.decode(ids), "the tower")
        self.assertEqual(tokenizer.tokenize("the tower"), ["<s>", "the", "Ġtower"])


class PythonTokenizerTest(unittest.TestCase):
    """Tokenizers implemented in python, used directly and via the tokenizer factory."""

    def setUp(self) -> None:
        self.config = make_config()

    def test_direct(self) -> None:
        tokenizer = CharTokenizer(self.config)
        self.assertEqual(tokenizer.vocabulary_size(), 27)
        self.assertEqual(tokenizer.tokenize("ab c"), ["a", "b", " ", "c"])

    def test_factory(self) -> None:
        self.config.set("lib-rasr.tokenizer.type", "test-chars")
        tokenizer = librasr.create_tokenizer(self.config)
        self.assertIsInstance(tokenizer, CharTokenizer)
        # The default implementations of `encode` and `decode` in C++ call the python methods
        ids = tokenizer.encode("hello world")
        self.assertEqual(ids, [7, 4, 11, 11, 14, 26, 22, 14, 17, 11, 3])
        self.assertEqual(tokenizer.decode(ids), "hello world")
        # Calling the C++ method goes through RASR to the python implementation
        boundaries = librasr.Tokenizer.word_boundary_marks(tokenizer, tokenizer.token_str_to_id(" "))
        self.assertEqual(repr(boundaries), "WordBoundaryMarks(before=True, after=True)")

    def test_override_default_method(self) -> None:
        self.config.set("lib-rasr.tokenizer.type", "test-upper-case")
        tokenizer = librasr.create_tokenizer(self.config)
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "HELLO")

    def test_register_again(self) -> None:
        self.config.set("lib-rasr.tokenizer.type", "test-replaced")
        librasr.register_tokenizer_type("test-replaced", CharTokenizer)
        librasr.register_tokenizer_type("test-replaced", UpperCaseTokenizer)
        self.assertIsInstance(librasr.create_tokenizer(self.config), UpperCaseTokenizer)


if __name__ == "__main__":
    unittest.main()
