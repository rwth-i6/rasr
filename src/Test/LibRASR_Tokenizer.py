"""
Tests for the tokenizer bindings of librasr. Requires librasr to be importable, e.g. via PYTHONPATH.

Run with: python3 src/Test/LibRASR_Tokenizer.py
"""

import importlib.util
import subprocess
import sys
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

    # Raises ValueError for characters that are not part of the vocabulary
    def encode(self, text: str) -> list[int]:
        return [self.vocab.index(character) for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[id] for id in ids)

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
    """Additionally overrides `detokenize`."""

    def detokenize(self, tokens: list[str]) -> str:
        return super().detokenize(tokens).upper()


class RaisingTokenizer(CharTokenizer):
    """Raises in the methods that return their result without a status in C++."""

    def vocabulary_size(self) -> int:
        raise KeyError("size")

    def token_str_to_id(self, token: str) -> int:
        raise KeyError(token)


class IncompleteTokenizer(librasr.Tokenizer):
    """Does not implement `encode` and `token_str_to_id`."""

    def vocabulary_size(self) -> int:
        return 0


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

    def test_errors(self) -> None:
        config = make_config(
            **{
                "lib-rasr.tokenizer.type": "bpe",
                "lib-rasr.tokenizer.codes-file": str(DATA_DIR / "bpe.codes"),
                "lib-rasr.tokenizer.vocab-file": str(DATA_DIR / "bpe.vocab.py"),
            }
        )
        tokenizer = librasr.create_tokenizer(config)
        with self.assertRaisesRegex(RuntimeError, "not part of the vocabulary"):
            tokenizer.encode("xy")
        with self.assertRaisesRegex(RuntimeError, "Invalid token id"):
            tokenizer.token_id_to_str(34)
        config.set("lib-rasr.tokenizer.vocab-file", str(DATA_DIR / "does-not-exist.py"))
        with self.assertRaisesRegex(RuntimeError, "does-not-exist.py"):
            librasr.create_tokenizer(config)

    # Handled errors do not terminate the program later
    def test_exit_after_error(self) -> None:
        code = (
            "import librasr\n"
            "config = librasr.Configuration()\n"
            "config.set('*.tokenizer.type', 'bpe')\n"
            f"config.set('*.tokenizer.codes-file', {str(DATA_DIR / 'bpe.codes')!r})\n"
            f"config.set('*.tokenizer.vocab-file', {str(DATA_DIR / 'bpe.vocab.py')!r})\n"
            "librasr.create_tokenizer(config)\n"
            "config.set('*.tokenizer.type', 'does-not-exist')\n"
            "try:\n"
            "    librasr.create_tokenizer(config)\n"
            "except RuntimeError as error:\n"
            "    assert 'No valid tokenizer type' in str(error)\n"
            "else:\n"
            "    raise AssertionError('no error')\n"
        )
        process = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr.decode())

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
        # The default implementations of `tokenize` and `detokenize` in C++ call the python methods
        tokens = tokenizer.tokenize("hello world")
        self.assertEqual(tokens, list("hello world"))
        self.assertEqual(tokenizer.detokenize(tokens), "hello world")
        # Calling the C++ method goes through RASR to the python implementation
        boundaries = librasr.Tokenizer.word_boundary_marks(tokenizer, tokenizer.token_str_to_id(" "))
        self.assertEqual(repr(boundaries), "WordBoundaryMarks(before=True, after=True)")
        # `INVALID_TOKEN_ID` of the python implementation is passed on, the default `detokenize` fails on it
        self.assertEqual(librasr.Tokenizer.token_str_to_id(tokenizer, "A"), librasr.Tokenizer.INVALID_TOKEN_ID)
        with self.assertRaisesRegex(RuntimeError, "not part of the vocabulary"):
            tokenizer.detokenize(["A"])

    def test_override_default_method(self) -> None:
        self.config.set("lib-rasr.tokenizer.type", "test-upper-case")
        tokenizer = librasr.create_tokenizer(self.config)
        self.assertEqual(tokenizer.detokenize(tokenizer.tokenize("hello")), "HELLO")
        # Also when called from C++
        self.assertEqual(librasr.Tokenizer.detokenize(tokenizer, ["h", "i"]), "HI")

    # Exceptions of python implementations are passed on when called from C++
    def test_python_exception(self) -> None:
        tokenizer = CharTokenizer(self.config)
        with self.assertRaisesRegex(RuntimeError, "ValueError"):
            tokenizer.tokenize("Hello")
        tokenizer = RaisingTokenizer(self.config)
        with self.assertRaisesRegex(RuntimeError, "KeyError"):
            librasr.Tokenizer.vocabulary_size(tokenizer)
        with self.assertRaisesRegex(RuntimeError, "KeyError"):
            librasr.Tokenizer.token_str_to_id(tokenizer, "a")
        with self.assertRaisesRegex(RuntimeError, "KeyError"):
            tokenizer.detokenize(["a"])

    def test_missing_method(self) -> None:
        tokenizer = IncompleteTokenizer(self.config)
        with self.assertRaisesRegex(RuntimeError, "encode"):
            tokenizer.tokenize("a")
        with self.assertRaisesRegex(RuntimeError, "not implement `token_str_to_id`"):
            librasr.Tokenizer.token_str_to_id(tokenizer, "a")

    def test_register_again(self) -> None:
        self.config.set("lib-rasr.tokenizer.type", "test-replaced")
        librasr.register_tokenizer_type("test-replaced", CharTokenizer)
        librasr.register_tokenizer_type("test-replaced", UpperCaseTokenizer)
        self.assertIsInstance(librasr.create_tokenizer(self.config), UpperCaseTokenizer)

    # RASR keeps the registered callable until after the interpreter has been finalized
    def test_exit_after_register(self) -> None:
        code = "import librasr; librasr.register_tokenizer_type('test-function', lambda config: None)"
        process = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr.decode())


if __name__ == "__main__":
    unittest.main()
