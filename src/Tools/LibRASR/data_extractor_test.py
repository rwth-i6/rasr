from librasr import Configuration, DataExtractor, CorpusDescription, SpeechCorpusVisitor

conf = Configuration()
conf.set_from_file("./data_extractor.config")

# 1 DataExtractor performs feature extraction on segments from corpus
conf.set_selection("lib-rasr.data-extractor")
data_extractor = DataExtractor(conf, True)

conf.set_selection("lib-rasr.corpus")
corpus_d = CorpusDescription(conf)

v = SpeechCorpusVisitor(conf)

data_extractor.sign_on(v)
corpus_d.accept(v)
