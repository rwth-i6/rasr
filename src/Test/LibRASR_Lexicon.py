"""
Tests for the lexicon bindings of librasr. Requires librasr to be importable, e.g. via PYTHONPATH.

Run with: python3 src/Test/LibRASR_Lexicon.py
"""

import tempfile
import unittest
from pathlib import Path

import librasr

LEXICON = """<?xml version="1.0" encoding="utf-8"?>
<lexicon>
  <phoneme-inventory>
    <phoneme><symbol>a</symbol></phoneme>
    <phoneme><symbol>b</symbol></phoneme>
  </phoneme-inventory>
  <lemma><orth>ab</orth><phon>a b</phon></lemma>
  <lemma><orth>ba</orth><phon>b a</phon></lemma>
</lexicon>
"""


class LexiconTest(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.xml"
            path.write_text(LEXICON)
            config = librasr.Configuration()
            config.set("*.log.channel", "nil")
            config.set("lexicon.file", str(path))
            self.lexicon = librasr.Lexicon.create(config=librasr.Configuration(config, "lexicon"))

    # Comparisons with other types are False instead of raising a TypeError, equal objects have equal hashes
    def test_symbol_equality(self) -> None:
        first, second = [lemma.symbol() for lemma in self.lexicon.lemmas()]
        self.assertEqual(first, next(iter(self.lexicon.lemmas())).symbol())
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, "ab")
        self.assertEqual(len({first, second, next(iter(self.lexicon.lemmas())).symbol()}), 2)

    def test_symbol_conversions(self) -> None:
        symbol = next(iter(self.lexicon.lemmas())).symbol()
        self.assertTrue(symbol)
        self.assertEqual(len(symbol), 2)
        self.assertEqual(symbol.to_string(), "ab")
        self.assertEqual(repr(symbol), 'Symbol("ab")')

        # An empty symbol has no string
        empty = librasr.Symbol()
        self.assertFalse(empty)
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.to_string(), "")
        self.assertIsNone(empty.to_cstring())
        self.assertEqual(repr(empty), "Symbol()")

    def test_iterator_equality(self) -> None:
        alphabet = self.lexicon.lemma_alphabet()
        iterator = librasr.Alphabet.const_iterator(alphabet, 1)
        self.assertEqual(iterator, librasr.Alphabet.const_iterator(alphabet, 1))
        self.assertNotEqual(iterator, librasr.Alphabet.const_iterator(alphabet, 2))
        self.assertEqual(hash(iterator), hash(librasr.Alphabet.const_iterator(alphabet, 1)))
        self.assertEqual(librasr.Pronunciation.LemmaIterator(), librasr.Pronunciation.LemmaIterator())
        self.assertNotEqual(librasr.Pronunciation.LemmaIterator(), 0)

    def test_add_pronunciation(self) -> None:
        lemma = self.lexicon.new_lemma(name="new")
        first, second = self.lexicon.pronunciations()
        self.lexicon.add_pronunciation(lemma, first)
        self.lexicon.add_pronunciation(lemma=lemma, pronunciation=second, weight=2.0)
        self.assertEqual(lemma.num_pronunciations(), 2)


if __name__ == "__main__":
    unittest.main()
