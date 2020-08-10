/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include <Bliss/Lexicon.hh>
#include <Core/Application.hh>
#include <Core/CompressedStream.hh>
#include <Math/Utilities.hh>

#include <Flf/Module.hh>
#include <Flow/Module.hh>
#include <Lm/Module.hh>
#include <Math/Module.hh>
#include <Mc/Module.hh>
#include <Me/Module.hh>
#include <Mm/Module.hh>
#include <Nn/Module.hh>
#include <Signal/Module.hh>
#include <Speech/Module.hh>
#ifdef MODULE_TENSORFLOW
#include <Tensorflow/Module.hh>
#endif

namespace {

using LanguageModelRef = Core::Ref<const Lm::LanguageModel>;

struct LMRequest {
    std::string                  word;
    Bliss::Lemma const*          lemma;
    Bliss::SyntacticToken const* token;
    Lm::History                  history;
    Lm::Score                    score;
};

void computeAllScores(std::vector<LMRequest>& requests, LanguageModelRef lm, bool renormalize) {
    for (auto& r : requests) {
        Lm::Score normalization = 0.0f;
        if (renormalize) {
            normalization = std::numeric_limits<Lm::Score>::infinity();
            auto iters    = lm->lexicon()->syntacticTokens();
            for (auto tok = iters.first; tok != iters.second; ++tok) {
                Lm::Score s   = lm->score(r.history, *tok);
                normalization = Math::scoreSum(normalization, s);
            }
        }
        r.score = lm->score(r.history, r.token) - normalization;
    }
}

}  // namespace

class LmUtilityTool : public Core::Application {
public:
    enum Action {
        actionNotGiven,
        actionComputePerplexityFromTextFile
    };

    static const Core::Choice          choiceAction;
    static const Core::ParameterChoice paramAction;
    static const Core::ParameterString paramFile;
    static const Core::ParameterString paramEncoding;
    static const Core::ParameterString paramScoreFile;
    static const Core::ParameterInt    paramBatchSize;
    static const Core::ParameterBool   paramRenormalize;

    LmUtilityTool();
    virtual ~LmUtilityTool() = default;

    int main(std::vector<std::string> const& arguments);

private:
    void computePerplexityFromTextFile();
};

APPLICATION(LmUtilityTool)

// ---------- Implementations ----------

const Core::Choice          LmUtilityTool::choiceAction("compute-perplexity-from-text-file", actionComputePerplexityFromTextFile, Core::Choice::endMark());
const Core::ParameterChoice LmUtilityTool::paramAction("action", &choiceAction, "action to perform", actionNotGiven);
const Core::ParameterString LmUtilityTool::paramFile("file", "input file");
const Core::ParameterString LmUtilityTool::paramEncoding("encoding", "the encoding of the input file", "utf8");
const Core::ParameterString LmUtilityTool::paramScoreFile("score-file", "output path for word scores", "");
const Core::ParameterInt    LmUtilityTool::paramBatchSize("batch-size", "number of sequences to process in one batch", 100);
const Core::ParameterBool   LmUtilityTool::paramRenormalize("renormalize", "wether to renormalize the word probabiliies", false);

LmUtilityTool::LmUtilityTool()
        : Core::Application() {
    INIT_MODULE(Lm);
    INIT_MODULE(Mm);
    INIT_MODULE(Mc);
    INIT_MODULE(Flf);
    INIT_MODULE(Flow);
    INIT_MODULE(Math);
    INIT_MODULE(Signal);
    INIT_MODULE(Speech);
    INIT_MODULE(Nn);
    INIT_MODULE(Me);
#ifdef MODULE_TENSORFLOW
    INIT_MODULE(Tensorflow);
#endif

    setTitle("lm-util");
}

int LmUtilityTool::main(std::vector<std::string> const& arguments) {
    switch (paramAction(config)) {
        case actionComputePerplexityFromTextFile: computePerplexityFromTextFile(); break;
        default:
        case actionNotGiven: error("no action given");
    }
    return EXIT_SUCCESS;
}

void LmUtilityTool::computePerplexityFromTextFile() {
    bool                   renormalize = paramRenormalize(config);
    size_t                 batch_size  = paramBatchSize(config);
    Bliss::LexiconRef      lexicon(Bliss::Lexicon::create(select("lexicon")));
    LanguageModelRef       lm(Lm::Module::instance().createLanguageModel(select("lm"), lexicon));
    Core::TextInputStream  tis(new Core::CompressedInputStream(paramFile(config)));
    Core::TextOutputStream out;

    tis.setEncoding(paramEncoding(config));
    std::string out_file = paramScoreFile(config);
    if (not out_file.empty()) {
        out.open(out_file);
    }

    std::vector<LMRequest> requests;
    size_t                 num_tokens   = 0;
    Lm::Score              corpus_score = 0.0;

    do {
        std::string line;
        std::getline(tis, line);
        if (tis.good()) {
            std::stringstream ss(line);
            Lm::History       h = lm->startHistory();
            while (ss.good()) {
                std::string word;
                ss >> word;
                Bliss::Lemma const* lemma = lexicon->lemma(word);
                if (lemma == nullptr) {
                    lemma = lexicon->specialLemma("unknown");
                }
                auto const tokens = lemma->syntacticTokenSequence();
                for (auto const& t : tokens) {
                    requests.emplace_back(LMRequest({word, lemma, t, h, 0.0f}));
                    h = lm->extendedHistory(h, t);
                }
            }
            Bliss::Lemma const* lemma  = lexicon->specialLemma("sentence-end");
            auto const          tokens = lemma->syntacticTokenSequence();
            for (auto const& t : tokens) {
                requests.emplace_back(LMRequest({"\\n", lemma, t, h, 0.0f}));
                h = lm->extendedHistory(h, t);
            }
        }

        if (not tis.good() or requests.size() >= batch_size) {
            computeAllScores(requests, lm, renormalize);
            for (auto const& r : requests) {
                corpus_score += r.score;
                num_tokens += 1ul;
                if (out.good()) {
                    out << r.word << " " << r.lemma->preferredOrthographicForm().str() << " " << r.score << std::endl;
                }
            }
            requests.clear();
        }
    } while (tis.good());

    Lm::Score ppl = std::exp(corpus_score / num_tokens);

    log() << Core::XmlOpen("corpus-score") << corpus_score << Core::XmlClose("corpus-score")
          << Core::XmlOpen("num-tokens") << num_tokens << Core::XmlClose("num-tokens")
          << Core::XmlOpen("perplexity") << ppl << Core::XmlClose("perplexity");
}
