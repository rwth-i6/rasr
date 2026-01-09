from librasr import Configuration, DataExtractor, CorpusDescription, SpeechCorpusVisitor, FeatureExtractor  

conf = Configuration()
conf.set_from_file("/nas/models/asr/hyoshimochi/setups/2024-08-06-nour-internship/test/data_extractor.config")

#void DataExtractor::processSegment(Bliss::Segment* segment) {
#    while (dataSource()->getData())
#        ;
#}
#
#// ====================================================================================================
#void FeatureExtractor::processSegment(Bliss::Segment* segment) {
#    Core::Ref<Feature> feature;
#    bool               firstFeature = true;
#    while (dataSource()->getData(feature)) {
#        if (firstFeature) {  // try to check the dimension only once for each segment
#            setFeatureDescription(Mm::FeatureDescription(*this, *feature));
#            firstFeature = false;
#        }
#        processFeature(feature);
#    }
#}

# 1 DataExtractor is enough to do feature extraction on segments from corpus
conf.set_selection("lib-rasr.data-extractor")
data_extractor = DataExtractor(conf, True)

# 2
conf.set_selection("lib-rasr.data-extractor")
feat_extractor = FeatureExtractor(conf, True)

conf.set_selection("lib-rasr.corpus")
corpus_d = CorpusDescription(conf)

v = SpeechCorpusVisitor(conf)

data_extractor.sign_on(v)
corpus_d.accept(v)
