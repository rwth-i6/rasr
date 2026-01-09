from librasr import Configuration, FeatureExtractor, CorpusDescription, SpeechCorpusVisitor

conf = Configuration()
conf.set_from_file("./feature_extractor.config")

# 1 FeatureExtractor performs feature extraction on segments from corpus
conf.set_selection("lib-rasr.feature-extractor")
feature_extractor = FeatureExtractor(conf, True)

conf.set_selection("lib-rasr.corpus")
corpus_d = CorpusDescription(conf)

v = SpeechCorpusVisitor(conf)

feature_extractor.sign_on(v)
corpus_d.accept(v)
