from src.Tools.LibRASR import Configuration
from src.Tools.LibRASR import FeatureExtractor
from src.Tools.LibRASR import CorpusVisitor
from src.Tools.LibRASR import CorpusDescription

corpus_config = Configuration()
corpus_config.set_from_file("corpus.config")
feature_config = Configuration()
feature_config.set_from_file("feature.config")

v = CorpusVisitor(corpus_config)
p = FeatureExtractor(feature_config, True)
p.sign_on(v)
d = CorpusDescription(corpus_config)
d.accept(v)
