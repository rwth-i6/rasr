import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from librasr import (
    Configuration,
    Corpus,
    CorpusDescription,
    Data,
    Feature,
    FeatureDescription,
    FeatureExtractor,
    FeatureVector,
    FlowDataSource,
    InputNode,
    MmFeature,
    Network,
    Recording,
    SpeechCorpusVisitor,
    SpeechSegment,
    Timestamp,
)


SETUP_ROOT = Path(__file__).resolve().parents[5]
TEST_DATA_DIR = Path(os.environ.get("LIBRASR_TEST_DATA_DIR", SETUP_ROOT / "test"))
FLOW_FILE = Path(os.environ.get("LIBRASR_TEST_FLOW_FILE", SETUP_ROOT / "feature.flow.small"))
CORPUS_FILE = Path(os.environ.get("LIBRASR_TEST_CORPUS_FILE", TEST_DATA_DIR / "corpus.xml.gz"))
AUDIO_FILE = Path(os.environ.get("LIBRASR_TEST_AUDIO_FILE", TEST_DATA_DIR / "8288-274162-0066.wav"))


@contextmanager
def temporary_working_directory():
    previous_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="librasr-test-") as directory:
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous_directory)


def make_config():
    config = Configuration()
    config.set_selection("lib-rasr.feature-extractor")
    config.set("feature-extraction.file", str(FLOW_FILE))
    config.set_selection("lib-rasr.corpus")
    config.set("file", str(CORPUS_FILE))
    config.set("audio-dir", str(TEST_DATA_DIR))
    config.set("allow-empty-whitelist", "true")
    return config


class TrackingCorpusVisitor(SpeechCorpusVisitor):
    """Verify that C++ corpus traversal reaches snake_case Python overrides."""

    def __init__(self, config):
        super().__init__(config)
        self.corpora = []
        self.recordings = []
        self.speech_segments = []

    def enter_corpus(self, corpus):
        assert isinstance(corpus, Corpus)
        self.corpora.append(corpus.name())

    def enter_recording(self, recording):
        assert isinstance(recording, Recording)
        self.recordings.append((recording.name(), recording.audio()))

    def visit_speech_segment(self, segment):
        assert isinstance(segment, SpeechSegment)
        self.speech_segments.append(
            {
                "name": segment.name(),
                "orth": segment.orth(),
                "start": segment.start(),
                "end": segment.end(),
            }
        )


class PythonFeatureExtractor(FeatureExtractor):
    """Exercise FeatureExtractor's protected virtual callbacks from Python."""

    def __init__(self, config):
        super().__init__(config, True)
        self.description_num_streams = None
        self.num_features = 0
        self.first_feature_size = None
        self.first_timestamp = None

    def set_feature_description(self, description):
        assert isinstance(description, FeatureDescription)
        assert description.verify_number_of_streams(1)
        self.description_num_streams = description.num_streams()

    def process_feature(self, feature):
        assert isinstance(feature, Feature)
        assert feature.num_streams() == 1

        if self.num_features == 0:
            stream = feature.main_stream()
            timestamp = feature.timestamp()
            assert isinstance(stream, FeatureVector)
            assert isinstance(timestamp, Timestamp)
            self.first_feature_size = len(stream)
            self.first_timestamp = (timestamp.start_time(), timestamp.end_time())

        self.num_features += 1


class LibRasrFeatureExtractorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [path for path in (FLOW_FILE, CORPUS_FILE, AUDIO_FILE) if not path.is_file()]
        if missing:
            raise unittest.SkipTest(f"Missing LibRASR integration-test data: {missing}")

    def test_flow_value_types(self):
        timestamp = Timestamp(1.25, 2.5)
        self.assertTrue(timestamp.is_valid_timestamp())
        self.assertTrue(timestamp.contains(2.0))
        self.assertEqual(timestamp.start_time(), 1.25)
        self.assertEqual(timestamp.end_time(), 2.5)

        vector = FeatureVector(3)
        vector[0] = 1.0
        vector[1] = 2.0
        vector[2] = 3.0
        self.assertEqual(list(vector), [1.0, 2.0, 3.0])

        feature = MmFeature(1)
        feature.set(0, vector)
        self.assertEqual(feature.num_streams(), 1)
        self.assertEqual(list(feature.main_stream()), [1.0, 2.0, 3.0])

    def test_flow_network_introspection_and_node_ownership(self):
        config = Configuration()
        config.set("file", str(FLOW_FILE))
        with temporary_working_directory():
            source = FlowDataSource(config, True)

            self.assertEqual(source.outputs(), [(0, "features")])
            self.assertEqual(source.output_name(0), "features")
            self.assertIsInstance(source.get_node("samples"), InputNode)

            source.set_parameter("input-file", str(AUDIO_FILE))
            source.set_parameter("start-time", "0.0")
            source.set_parameter("end-time", "0.1")
            source.set_parameter("track", "0")
            source.configure()
            source.reset()
            source.configure_all()
            self.assertIsInstance(source.get_data(0), Data)

        network_config = Configuration()
        network_config.set_selection("lib-rasr.test-network")
        network = Network(network_config, False)
        node_config = Configuration(network_config, "input")
        node = network.add_input_node(node_config)
        self.assertIsInstance(node, InputNode)
        self.assertIs(network.get_node("input"), node)

    def test_snake_case_corpus_callbacks(self):
        config = make_config()
        config.set_selection("lib-rasr.corpus")
        corpus = CorpusDescription(config)
        visitor = TrackingCorpusVisitor(config)

        with temporary_working_directory():
            corpus.accept(visitor)

        self.assertEqual(visitor.corpora, ["dev-other"])
        self.assertEqual(len(visitor.recordings), 1)
        self.assertEqual(visitor.recordings[0][0], "8288-274162-0066")
        self.assertTrue(visitor.recordings[0][1].endswith("8288-274162-0066.wav"))
        self.assertEqual(len(visitor.speech_segments), 1)
        self.assertEqual(visitor.speech_segments[0]["name"], "8288-274162-0066")
        self.assertIn("SHE SAID", visitor.speech_segments[0]["orth"])
        self.assertEqual(visitor.speech_segments[0]["start"], 0.0)
        self.assertGreater(visitor.speech_segments[0]["end"], 0.0)

    def test_feature_extractor_callbacks(self):
        config = make_config()
        config.set_selection("lib-rasr.feature-extractor")
        feature_extractor = PythonFeatureExtractor(config)

        config.set_selection("lib-rasr.corpus")
        corpus = CorpusDescription(config)
        visitor = SpeechCorpusVisitor(config)
        feature_extractor.sign_on(visitor)

        with temporary_working_directory():
            corpus.accept(visitor)

        self.assertEqual(feature_extractor.description_num_streams, 1)
        self.assertGreater(feature_extractor.num_features, 0)
        self.assertGreater(feature_extractor.first_feature_size, 0)
        self.assertLessEqual(feature_extractor.first_timestamp[0], feature_extractor.first_timestamp[1])


if __name__ == "__main__":
    unittest.main()
