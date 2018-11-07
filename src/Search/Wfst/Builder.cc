/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include <Core/Debug.hh>
#include <Core/Directory.hh>
#include <Core/Application.hh>
#include <OpenFst/Types.hh>
#include <OpenFst/Count.hh>
#include <Speech/ModelCombination.hh>
#include <Lm/Module.hh>
#include <Am/Module.hh>
#include <Search/Wfst/Builder.hh>

using namespace Search::Wfst::Builder;

Resources::Resources(const Core::Configuration &c) :
    Core::Component(c)
{
    models_ = new Speech::ModelCombination(select("model-combination"),
                    Speech::ModelCombination::useLexicon, Am::AcousticModel::noEmissions);
}

Resources::~Resources()
{
    delete models_;
}

Core::Ref<const Bliss::Lexicon> Resources::lexicon() const
{
    return models_->lexicon();
}

Core::Ref<const Lm::ScaledLanguageModel> Resources::languageModel() const
{
    if (!models_->languageModel()) {
        models_->setLanguageModel(Lm::Module::instance().createScaledLanguageModel(
                Core::Configuration(models_->getConfiguration(), "lm"), lexicon()));
    }
    return models_->languageModel();
}

void Resources::deleteLanguageModel() const
{
    models_->setLanguageModel(Core::Ref<Lm::ScaledLanguageModel>());
}

Core::Ref<const Am::AcousticModel> Resources::acousticModel() const
{
    if (!models_->acousticModel()) {
        models_->setAcousticModel(Am::Module::instance().createAcousticModel(
                Core::Configuration(models_->getConfiguration(), "acoustic-model"), lexicon(), Am::AcousticModel::noEmissions));
    }
    return models_->acousticModel();
}

Mm::Score Resources::pronunciationScale() const
{
    return models_->pronunciationScale();
}

// ============================================================================
const int Automaton::InvalidIntAttribute = Core::Type<int>::max;
const char* Automaton::attrNumDisambiguators = "disambiguators";

Automaton* Automaton::cloneWithAttributes() const
{
    Automaton *r = clone();
    copyAttributes(r);
    return r;
}

void Automaton::copyAttributes(Automaton *dest) const
{
    for (Attributes::const_iterator a = attributes_.begin();
            a != attributes_.end(); ++a)
        dest->setAttribute(a->first, a->second);
}

void Automaton::copyAttribute(Automaton *dest, const std::string &name) const
{
    dest->setAttribute(name, getAttribute(name));
}

std::string Automaton::getAttribute(const std::string &name) const
{
    Attributes::const_iterator i = attributes_.find(name);
    if (i != attributes_.end())
        return i->second;
    else {
        Core::Application::us()->warning("attribute %s not found", name.c_str());
        return "";
    }
}

int Automaton::getIntAttribute(const std::string &name) const
{
    int value;
    std::string strValue = getAttribute(name);
    if (strValue.empty() || !Core::strconv(strValue, value))
        return InvalidIntAttribute;
    else
        return value;
}

void Automaton::setAttribute(const std::string &name, const std::string &value)
{
    attributes_.insert(std::make_pair(name, value));
}

void Automaton::setAttribute(const std::string &name, int value)
{
    setAttribute(name, Core::form("%d", value));
}

Operation::AutomatonRef Operation::getResult()
{
    if (!precondition()) return 0;
    bool measureTime = timerChannel_.isOpen();
    if (measureTime) timer_.start();
    AutomatonRef result = process();
    if (measureTime) {
        timer_.stop();
        timer_.write(timerChannel_);
    }
    return result;
}


const Core::Choice OutputTypeDependent::choiceOutputType(
        "lemma-pronunciations", outputLemmaPronunciations,
        "lemmas", outputLemmas,
        "syntactic-tokens",  outputSyntacticTokens,
        Core::Choice::endMark());

const Core::ParameterChoice OutputTypeDependent::paramOutputType(
        "output-type", &choiceOutputType, "type of output", outputLemmaPronunciations);


const Core::Choice SemiringDependent::choiceSemiring(
        "tropical", tropicalSemiring,
        "log", logSemiring,
        Core::Choice::endMark());

const Core::ParameterChoice SemiringDependent::paramSemiring(
        "semiring", &choiceSemiring, "semiring used", tropicalSemiring);

const Core::Choice LabelTypeDependent::choiceLabel(
        "input", LabelTypeDependent::Input,
        "output", LabelTypeDependent::Output,
        Core::Choice::endMark());

const Core::ParameterChoice LabelTypeDependent::paramLabel(
        "label", &choiceLabel, "input or output label", LabelTypeDependent::Input);




const Core::ParameterInt DisambiguatorDependentOperation::paramDisambiguators(
        "disambiguators", "number of disambiguators in the alphabet", -1);

u32 DisambiguatorDependentOperation::nInputAutomata() const
{
    if (nDisambiguators_ < 0)
        return 1;
    else
        return 0;
}

bool DisambiguatorDependentOperation::addInput(AutomatonRef f)
{
    s32 nDisambiguators = f->getIntAttribute(Automaton::attrNumDisambiguators);
    if (nDisambiguators == Automaton::InvalidIntAttribute) {
        log("automaton has no disambiguator count");
        return false;
    } else {
        nDisambiguators_ = nDisambiguators;
        return true;
    }
}
