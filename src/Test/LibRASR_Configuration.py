"""
Tests for the configuration bindings of librasr. Requires librasr to be importable, e.g. via PYTHONPATH.

Run with: python3 src/Test/LibRASR_Configuration.py
"""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import librasr
import numpy as np

LEXICON = """<?xml version="1.0" encoding="utf-8"?>
<lexicon>
  <phoneme-inventory>
    <phoneme><symbol>a</symbol></phoneme>
  </phoneme-inventory>
  <lemma><orth>a</orth></lemma>
</lexicon>
"""


class ConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = librasr.Configuration()
        self.config.set("*.log.channel", "nil")

    def test_default_selection(self) -> None:
        self.assertEqual(self.config.get_selection(), "lib-rasr")

    # Names are relative to the selection of the config
    def test_set_and_get(self) -> None:
        self.config.set("value", "1")
        self.config.set("component.value", "2")
        self.config.set("flag")
        self.assertEqual(self.config["value"], "1")
        self.assertEqual(self.config["component.value"], "2")
        self.assertEqual(self.config["flag"], "true")
        self.assertIsNone(self.config["unset"])

        self.config.set("value", "3")
        self.assertEqual(self.config["value"], "3")

    # A wildcard matches any number of names, including none
    def test_wildcards(self) -> None:
        self.config.set("*.value", "1")
        self.config.set("component.*", "2")
        self.assertEqual(self.config["value"], "1")
        self.assertEqual(self.config["a.b.value"], "1")
        self.assertEqual(self.config["component.other"], "2")

    def test_sub_config(self) -> None:
        self.config.set("component.nested.value", "1")
        sub_config = librasr.Configuration(self.config, "component")
        self.assertEqual(sub_config.get_selection(), "lib-rasr.component")
        self.assertEqual(sub_config["nested.value"], "1")
        self.assertEqual(librasr.Configuration(self.config, "component.nested").get_selection(), "lib-rasr.component.nested")

        # Setting values in a sub-config is relative to its selection as well
        sub_config.set("other", "2")
        self.assertEqual(self.config["component.other"], "2")

    # Copies and sub-configs share their values with the original config
    def test_shared_values(self) -> None:
        librasr.Configuration(self.config).set("from-copy", "1")
        librasr.Configuration(self.config, "component").set("from-sub-config", "2")
        self.assertEqual(self.config["from-copy"], "1")
        self.assertEqual(self.config["component.from-sub-config"], "2")

    def test_set_selection(self) -> None:
        self.config.set("component.value", "1")
        self.config.set_selection("lib-rasr.component")
        self.assertEqual(self.config.get_selection(), "lib-rasr.component")
        self.assertEqual(self.config.get_name(), "component")
        self.assertEqual(self.config["value"], "1")

        # Setting values is relative to the new selection as well
        self.config.set("other", "2")
        self.config.set_selection("lib-rasr")
        self.assertEqual(self.config["component.other"], "2")

    # Config files use absolute names as in all RASR tools
    def test_set_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.config"
            path.write_text("[*.component]\nvalue = 1\npath = $(base)/file\nother = 1\n")
            self.assertTrue(self.config.set_from_file(str(path)))
        self.config.set("base", "/data")
        self.assertEqual(self.config["component.value"], "1")
        self.assertEqual(self.config["component.path"], "/data/file")

        # Values set from python are more specific than wildcard values from files
        self.config.set("component.other", "2")
        self.assertEqual(self.config["component.other"], "2")

        self.assertFalse(self.config.set_from_file("/nonexistent/file.config"))

    # Malformed names raise a ValueError instead of aborting the process
    def test_invalid_names(self) -> None:
        invalid = {
            "read with wildcard": lambda: self.config["*.value"],
            "read empty name": lambda: self.config[""],
            "read empty component": lambda: self.config["component..value"],
            "set empty name": lambda: self.config.set("", "1"),
            "set empty component": lambda: self.config.set("component..value", "1"),
            "set partial wildcard": lambda: self.config.set("value*", "1"),
            "empty sub-config": lambda: librasr.Configuration(self.config, ""),
            "wildcard sub-config": lambda: librasr.Configuration(self.config, "*"),
            "wildcard selection": lambda: self.config.set_selection("lib-rasr.*"),
            "empty selection": lambda: self.config.set_selection(""),
        }
        for case, function in invalid.items():
            with self.subTest(case):
                with self.assertRaises(ValueError):
                    function()


class TypedValueTest(unittest.TestCase):
    """Values are stored as strings, typed setting and reading uses the parsing rules of RASR's parameters."""

    def setUp(self) -> None:
        self.config = librasr.Configuration()
        self.config.set("*.log.channel", "nil")

    def test_set_types(self) -> None:
        self.config.set("bool", True)
        self.config.set("int", 12)
        self.config.set("float", 0.1)
        self.config.set("path", Path("/data/file"))
        self.config.set("string", "value")
        self.config.set("numpy-int", np.int64(3))
        self.config.set("numpy-float", np.float32(0.5))
        self.assertEqual(self.config["bool"], "true")
        self.assertEqual(self.config["int"], "12")
        self.assertEqual(self.config["float"], "0.1")
        self.assertEqual(self.config["path"], "/data/file")
        self.assertEqual(self.config["string"], "value")
        self.assertEqual(self.config["numpy-int"], "3")
        self.assertEqual(self.config["numpy-float"], "0.5")
        with self.assertRaises(TypeError):
            self.config.set("value", None)

    # Floats are stored with enough digits to be read back exactly
    def test_float_round_trip(self) -> None:
        for value in [0.1, 1e-5, 1 / 3, -2.5e10]:
            with self.subTest(value=value):
                self.config.set("value", value)
                self.assertEqual(self.config.get_float("value"), value)

    def test_get_bool(self) -> None:
        for string, expected in [("true", True), ("yes", True), ("on", True), ("1", True), ("false", False), ("no", False), ("off", False), ("0", False)]:
            with self.subTest(string=string):
                self.config.set("value", string)
                self.assertIs(self.config.get_bool("value"), expected)
        self.config.set("value", "True")
        with self.assertRaises(ValueError):
            self.config.get_bool("value")

    def test_get_int(self) -> None:
        for string, expected in [(" 12 ", 12), ("-3", -3), ("infinite", 2**31 - 1), ("-infinite", -(2**31))]:
            with self.subTest(string=string):
                self.config.set("value", string)
                self.assertEqual(self.config.get_int("value"), expected)
        # RASR's int parameters have 32 bits
        for string in ["12abc", "0.5", str(2**31)]:
            with self.subTest(string=string):
                self.config.set("value", string)
                with self.assertRaises(ValueError):
                    self.config.get_int("value")

    def test_get_float(self) -> None:
        self.config.set("value", "0.25")
        self.assertEqual(self.config.get_float("value"), 0.25)
        self.config.set("value", "infinite")
        self.assertEqual(self.config.get_float("value"), sys.float_info.max)
        self.config.set("value", "abc")
        with self.assertRaises(ValueError):
            self.config.get_float("value")

    def test_default(self) -> None:
        self.assertIsNone(self.config.get_str("unset"))
        self.assertIsNone(self.config.get_int("unset"))
        self.assertEqual(self.config.get_int("unset", 5), 5)
        self.assertEqual(self.config.get_bool("unset", True), True)
        self.assertEqual(self.config.get_float_list("unset", [1.0]), [1.0])

    def test_lists(self) -> None:
        self.config.set("ints", [4, 8, 16], delimiter=",")
        self.config.set("strings", ("a", "b c"), delimiter=";")
        self.config.set("floats", [0.5, 1.5])
        self.assertEqual(self.config["ints"], "4,8,16")
        self.assertEqual(self.config["floats"], "0.5 1.5")
        self.assertEqual(self.config.get_int_list("ints", delimiter=","), [4, 8, 16])
        self.assertEqual(self.config.get_str_list("strings", delimiter=";"), ["a", "b c"])
        self.assertEqual(self.config.get_float_list("floats"), [0.5, 1.5])

        self.config.set("empty", "")
        self.assertEqual(self.config.get_int_list("empty"), [])

    # Int and float lists allow ranges like RASR's vector parameters
    def test_list_ranges(self) -> None:
        self.config.set("value", "1:4,10")
        self.assertEqual(self.config.get_int_list("value", delimiter=","), [1, 2, 3, 4, 10])
        self.config.set("value", "0:2")
        self.assertEqual(self.config.get_float_list("value"), [0.0, 1.0, 2.0])

    def test_invalid_lists(self) -> None:
        for string in ["1,x", f"1,{2**31}", f"1:{2**31}"]:
            with self.subTest(string=string):
                self.config.set("value", string)
                with self.assertRaises(ValueError):
                    self.config.get_int_list("value", delimiter=",")
        with self.assertRaises(ValueError):
            self.config.get_int_list("value", delimiter="")
        with self.assertRaises(ValueError):
            self.config.set("value", [1, 2], delimiter="")
        # The element would otherwise be read back as two elements
        with self.assertRaises(ValueError):
            self.config.set("value", ["a b", "c"])


class RecordingLabelScorer(librasr.LabelScorer):
    """Label scorer which records its config when created by RASR."""

    config: Optional[librasr.Configuration] = None

    def __init__(self, config: librasr.Configuration) -> None:
        super().__init__(config)
        RecordingLabelScorer.config = config

    def reset(self) -> None:
        pass

    def signal_no_more_features(self) -> None:
        pass

    def get_initial_scoring_context(self) -> int:
        return 0

    def allowed_transition_types(self) -> list[librasr.TransitionType]:
        return [librasr.TransitionType.LABEL_TO_LABEL]

    def extended_scoring_context(self, context: int, next_token: int, transition_type: librasr.TransitionType) -> int:
        return context

    def add_inputs(self, inputs: np.ndarray) -> None:
        pass

    def compute_scores_with_times(self, contexts: list[int]) -> list[Optional[tuple[list[float], int]]]:
        return [None for _ in contexts]


librasr.register_label_scorer_type("test-recording", RecordingLabelScorer)


class ConfigurationFromRasrTest(unittest.TestCase):
    """Configs that RASR passes to python components are full `librasr.Configuration` objects."""

    def test_label_scorer_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lexicon = Path(directory) / "lexicon.xml"
            lexicon.write_text(LEXICON)
            config = librasr.Configuration()
            config.set("*.log.channel", "nil")
            config.set("lexicon.file", str(lexicon))
            config.set("search-algorithm.type", "lexiconfree-timesync-beam-search")
            config.set("search-algorithm.max-beam-size", "4")
            config.set("label-scorer.type", "test-recording")
            config.set("label-scorer.value", "1")
            # Fails if the C++ components do not find the values set relative to the root selection
            librasr.SearchAlgorithm(config)

        scorer_config = RecordingLabelScorer.config
        self.assertIsInstance(scorer_config, librasr.Configuration)
        self.assertEqual(scorer_config.get_selection(), "lib-rasr.label-scorer")
        self.assertEqual(scorer_config["value"], "1")

        # Sub-configs for nested components and setting values work as well
        sub_config = librasr.Configuration(scorer_config, "nested")
        sub_config.set("value", "2")
        self.assertEqual(sub_config.get_selection(), "lib-rasr.label-scorer.nested")
        self.assertEqual(scorer_config["nested.value"], "2")


if __name__ == "__main__":
    unittest.main()
