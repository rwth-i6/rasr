from librasr import Configuration, CorpusDescription, FeatureDescription, FeatureExtractor, SpeechCorpusVisitor


class PythonFeatureExtractor(FeatureExtractor):
    def __init__(self, config):
        super().__init__(config, True)
        self.description = None
        self.num_features = 0

    def set_feature_description(self, description):
        assert isinstance(description, FeatureDescription)
        self.description = description

    def process_feature(self, feature):
        self.num_features += 1

conf = Configuration()
conf.set_from_file("./feature_extractor.config")

# 1 FeatureExtractor performs feature extraction on segments from corpus
conf.set_selection("lib-rasr.feature-extractor")
feature_extractor = PythonFeatureExtractor(conf)

conf.set_selection("lib-rasr.corpus")
corpus_d = CorpusDescription(conf)

v = SpeechCorpusVisitor(conf)

feature_extractor.sign_on(v)
corpus_d.accept(v)

assert feature_extractor.description is not None
assert feature_extractor.num_features > 0
