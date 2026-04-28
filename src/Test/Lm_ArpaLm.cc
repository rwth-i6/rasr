#include <Bliss/Lexicon.hh>
#include <Lm/Module.hh>
#include <Test/UnitTest.hh>

class TestArpaLm : public Test::Fixture {
public:
    void setUp();

protected:
    Core::Configuration base_lex_config_;
    Core::Configuration shuffle_lex_config_;
    Core::Configuration lm_config_;
    Bliss::LexiconRef   base_lex_;
    Bliss::LexiconRef   shuffle_lex_;
};

void TestArpaLm::setUp() {
    base_lex_config_.set("*.lexicon.file", "data/arpa_lm/base.xml.gz");
    base_lex_ = Bliss::Lexicon::create(Core::Configuration(base_lex_config_, "lexicon"));

    shuffle_lex_config_.set("*.lexicon.file", "data/arpa_lm/shuffle.xml.gz");
    shuffle_lex_ = Bliss::Lexicon::create(Core::Configuration(shuffle_lex_config_, "lexicon"));

    lm_config_.set("*.lm.type", "ARPA");
    lm_config_.set("*.lm.file", "data/arpa_lm/unigram.arpa.gz");
    lm_config_.set("*.lm.image", "");
}

TEST_F(Test, TestArpaLm, TestShuffle) {
    auto m = Lm::Module::instance();
    lm_config_.set("*.lm.image", "data/arpa_lm/unigram.image");
    Core::Ref<Lm::LanguageModel> base_lm    = m.createLanguageModel(Core::Configuration(lm_config_, "lm"), base_lex_);
    Core::Ref<Lm::LanguageModel> shuffle_lm = m.createLanguageModel(Core::Configuration(lm_config_, "lm"), shuffle_lex_);

    Lm::History base_h    = base_lm->startHistory();
    Lm::History shuffle_h = shuffle_lm->startHistory();
}
