"""
Tests for the label scorer bindings of librasr. Requires librasr to be importable, e.g. via PYTHONPATH.

Run with: python3 src/Test/LibRASR_LabelScorer.py
"""

import subprocess
import sys
import unittest


class RegisterLabelScorerTest(unittest.TestCase):
    # RASR keeps the registered callable until after the interpreter has been finalized
    def test_exit_after_register(self) -> None:
        code = "import librasr; librasr.register_label_scorer_type('test-function', lambda config: None)"
        process = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr.decode())


if __name__ == "__main__":
    unittest.main()
