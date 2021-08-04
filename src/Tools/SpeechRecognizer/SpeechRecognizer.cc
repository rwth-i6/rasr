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
#include <Am/Module.hh>
#include <Audio/Module.hh>
#include <Bliss/CorpusDescription.hh>
#include <Core/Application.hh>
#include <Flow/Module.hh>
#include <Lm/Module.hh>
#include <Math/Module.hh>
#include <Mm/Module.hh>
#include <Modules.hh>
#include <Search/Module.hh>
#include <Signal/Module.hh>
#include <Speech/CorpusVisitor.hh>
#include <Speech/Module.hh>
#include <Speech/Recognizer.hh>
#ifdef MODULE_NN
#include <Nn/Module.hh>
#endif
#ifdef MODULE_TENSORFLOW
#include <Tensorflow/Module.hh>
#endif

class SpeechRecognizer : public Core::Application {
public:
    virtual std::string getUsage() const {
        return "off-line (i.e. corpus driven) speech recognizer";
    }

    SpeechRecognizer() {
        INIT_MODULE(Flow);
        INIT_MODULE(Am);
        INIT_MODULE(Audio);
        INIT_MODULE(Lm);
        INIT_MODULE(Math);
        INIT_MODULE(Mm);
        INIT_MODULE(Search);
        INIT_MODULE(Signal);
        INIT_MODULE(Speech);
#ifdef MODULE_NN
        INIT_MODULE(Nn);
#endif
#ifdef MODULE_TENSORFLOW
        INIT_MODULE(Tensorflow);
#endif

        setTitle("speech-recognizer");
    }

    enum RecognitionMode {
        offlineRecognition,
        offlineConstrainedRecognition,
        initOnlyRecognition,
    };
    static const Core::Choice          recognitionModeChoice;
    static const Core::ParameterChoice paramRecognitionMode;

public:
    int main(const std::vector<std::string>& arguments);
};

APPLICATION(SpeechRecognizer)

const Core::Choice SpeechRecognizer::recognitionModeChoice(
        "offline", offlineRecognition,
        "constrained", offlineConstrainedRecognition,
        "init-only", initOnlyRecognition,
        Core::Choice::endMark());
const Core::ParameterChoice SpeechRecognizer::paramRecognitionMode(
        "recognition-mode", &recognitionModeChoice,
        "operation mode: corpus-base (offline) or online",
        offlineRecognition);

int SpeechRecognizer::main(const std::vector<std::string>& arguments) {
    switch (paramRecognitionMode(config)) {
        case offlineRecognition:
        case offlineConstrainedRecognition: {
            Speech::CorpusProcessor* processor = 0;
            switch (paramRecognitionMode(config)) {
                case offlineRecognition:
                    processor = new Speech::OfflineRecognizer(config);
                    break;
                case offlineConstrainedRecognition:
                    processor = new Speech::ConstrainedOfflineRecognizer(config);
                    break;
                default: defect();
            }
            verify(processor);
            Speech::CorpusVisitor corpusVisitor(config);
            processor->signOn(corpusVisitor);
            Bliss::CorpusDescription corpusDescription(select("corpus"));
            corpusDescription.accept(&corpusVisitor);
            delete processor;
        } break;
        case initOnlyRecognition: {
            auto recognizer = Search::Module::instance().createRecognizer(static_cast<Search::SearchType>(Speech::Recognizer::paramSearch(config)), select("recognizer"));
            auto modelCombinationRef = Speech::ModelCombinationRef(new Speech::ModelCombination(select("model-combination"), recognizer->modelCombinationNeeded(), Am::AcousticModel::noEmissions));
            modelCombinationRef->load();
            recognizer->setModelCombination(*modelCombinationRef);
            recognizer->init();
            delete recognizer;
        } break;
        default: defect();
    }
    return 0;
}
