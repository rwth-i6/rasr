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
#include <Core/Application.hh>
#include <OpenFst/Scale.hh>
#include <Search/Wfst/ComposeFst.hh>
#include <Search/Wfst/ComposedNetwork.hh>
#include <Search/Wfst/GrammarFst.hh>
#include <Search/Wfst/LexiconFst.hh>
// #include <OpenFst/Scale.hh>

namespace Search {
namespace Wfst {

const Core::ParameterString ComposedNetwork::paramNetworkLeft_(
        "file-left", "left automaton for composition", "");
const Core::ParameterString ComposedNetwork::paramNetworkRight_(
        "file-right", "right automaton for composition", "");
const Core::ParameterInt ComposedNetwork::paramStateCache_(
        "state-cache", "number of bytes used for state caching", Core::Type<s32>::max, 0);
const Core::ParameterInt ComposedNetwork::paramResetInterval_(
        "reset-interval", "number of segments to process before resetting the ComposeFst", 0);
const Core::Choice ComposedNetwork::choiceGrammarType_(
        "vector", AbstractGrammarFst::TypeVector,
        "const", AbstractGrammarFst::TypeConst,
        "compact", AbstractGrammarFst::TypeCompact,
        "combine", AbstractGrammarFst::TypeCombine,
        "composed", AbstractGrammarFst::TypeCompose,
        "dynamic", AbstractGrammarFst::TypeDynamic,
        "fail-arc", AbstractGrammarFst::TypeFailArc,
        Core::Choice::endMark());
const Core::ParameterChoice ComposedNetwork::paramGrammarType_(
        "grammar-type", &choiceGrammarType_, "type of the right automaton",
        AbstractGrammarFst::TypeVector);

ComposedNetwork::ComposedNetwork(const Core::Configuration& c)
        : Precursor(c), l_(0), r_(0), stateTable_(0), resetCount_(0), resetInterval_(paramResetInterval_(config)), cacheSize_(paramStateCache_(config)) {
}

ComposedNetwork::~ComposedNetwork() {
    delete r_;
    delete l_;
    if (stateTable_)
        log("visited states in compose fst:") << stateTable_->size();
    delete stateTable_;
    // fst_ is deleted by destructor of FstNetwork
}

bool ComposedNetwork::init() {
    logMemoryUsage();
    createG();
    logMemoryUsage();
    createL();
    logMemoryUsage();
    // make sure reset() builds the composefst
    resetCount_ = resetInterval_;
    return r_ && l_;
}

void ComposedNetwork::createG() {
    AbstractGrammarFst::GrammarType t        = static_cast<AbstractGrammarFst::GrammarType>(paramGrammarType_(config));
    const std::string&              mainFile = paramNetworkRight_(config);
    r_                                       = AbstractGrammarFst::create(t, select("grammar-fst"));
    ensure(r_);
    r_->setLexicon(lexicon_);
    Core::Application::us()->log("reading G: %s", mainFile.c_str());
    if (!r_->load(mainFile)) {
        Core::Application::us()->criticalError("cannot load grammar fst");
    }
}

void ComposedNetwork::createL() {
    const std::string filename = paramNetworkLeft_(config);
    log("reading CL: %s", filename.c_str());
    AbstractGrammarFst::GrammarType gtype = static_cast<AbstractGrammarFst::GrammarType>(paramGrammarType_(config));
    l_                                    = LexicalFstFactory(select("lexicon-fst")).load(filename, gtype, r_);
}

void ComposedNetwork::reset() {
    if (++resetCount_ < resetInterval_) {
        return;
    }
    resetCount_ = 0;
    r_->reset();
    if (stateTable_)
        log("visited states in compose fst:") << stateTable_->size();
    delete stateTable_;
    delete f_;
    log("creating composed fst");
    logMemoryUsage();
    f_ = l_->compose(*r_, cacheSize_, &stateTable_);
    log("composed fst. cache=%zd", cacheSize_);
    logMemoryUsage();
}

}  // namespace Wfst
}  // namespace Search
