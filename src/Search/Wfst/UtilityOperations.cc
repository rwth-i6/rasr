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
#include <Am/ClassicAcousticModel.hh>
#include <Am/ClassicStateModel.hh>
#include <OpenFst/SymbolTable.hh>
#include <Search/Wfst/UtilityOperations.hh>
#include <Search/Wfst/StateSequence.hh>
#include <fst/arc-map.h>

using namespace Search::Wfst;
using namespace Search::Wfst::Builder;

Operation::AutomatonRef RemoveHmmDisambiguators::process()
{
    log("removing hmm disambiguators");
    FstLib::ArcMap(input_, HmmDisambiguatorRemoveMapper<OpenFst::Arc>());
    return input_;
}

Operation::AutomatonRef Info::process()
{
    log("# states: %d", input_->NumStates());
    u32 nArcs = 0;
    for (OpenFst::StateIterator siter(*input_); !siter.Done(); siter.Next())
        nArcs += input_->NumArcs(siter.Value());
    log("# arcs: %d", nArcs);
    return input_;
}

const Core::ParameterString Count::paramStateSequences(
    "state-sequences", "state sequences file", "");

Operation::AutomatonRef Count::process()
{
    log("counting");
    std::string stateSequences = paramStateSequences(config);
    log("loading state sequences: %s", stateSequences.c_str());
    StateSequenceList ssl;
    if (!ssl.read(stateSequences))
        criticalError("cannot read state sequences");
    u32 nStates = 0, nEffectiveStates = 0;
    u32 nArcs = 0, nEffectiveArcs = 0;
    u32 nEpsilonArcs = 0;
    u32 nSilenceArcs = 0;
    for (OpenFst::StateIterator siter(*input_); !siter.Done(); siter.Next()) {
        ++nStates; ++nEffectiveStates;
        for (OpenFst::ArcIterator aiter(*input_, siter.Value()); !aiter.Done(); aiter.Next()) {
            ++nArcs; ++nEffectiveArcs;
            const OpenFst::Arc &arc = aiter.Value();
            if (arc.ilabel != OpenFst::Epsilon) {
                const StateSequence &ss = ssl[arc.ilabel - 1];
                nEffectiveStates += ss.nStates() - 1;
                nEffectiveArcs += ss.nStates() - 1;
                if (ss.nStates() == 1)
                    ++nSilenceArcs;
            } else {
                ++nEpsilonArcs;
            }
        }
    }
    log("# states: %d", nStates);
    log("# expanded states: %d", nEffectiveStates);
    log("# arcs: %d", nArcs);
    log("# expanded arcs: %d", nEffectiveArcs);
    log("# silence acs: %d", nSilenceArcs);
    log("# epsilon arcs: %d", nEpsilonArcs);
    return input_;
}



const Core::ParameterString CreateStateSequenceSymbols::paramStateSequences(
    "state-sequences", "state sequences file", "");

const Core::ParameterBool CreateStateSequenceSymbols::paramShortSymbols(
    "short-symbols", "use abbreviated symbols", false);


Operation::AutomatonRef CreateStateSequenceSymbols::process()
{
    std::string stateSequences = paramStateSequences(config);
    log("loading state sequences: %s", stateSequences.c_str());
    StateSequenceList ssl;
    if (!ssl.read(stateSequences))
        criticalError("cannot read state sequences");
    OpenFst::SymbolTable *symbols = createSymbols(ssl);
    input_->SetInputSymbols(symbols);
    return input_;
}

OpenFst::SymbolTable* CreateStateSequenceSymbols::createSymbols(const StateSequenceList &ss) const
{
    const bool shortSymbols = paramShortSymbols(config);
    Core::Ref<const Am::AcousticModel> am = resources_.acousticModel();
    Core::Ref<const Am::AllophoneStateAlphabet> alphabet = am->allophoneStateAlphabet();
    u32 nMixtures = am->nEmissions();
    std::vector<std::string> stateTying(nMixtures, "");
    std::pair<Am::AllophoneStateIterator, Am::AllophoneStateIterator> iter = alphabet->allophoneStates();
    for (Am::AllophoneStateIterator s = iter.first; s != iter.second; ++s) {
        Am::AllophoneStateIndex ai = s.id();
        u32 m = am->emissionIndex(ai);
        std::string symbol = alphabet->toString(s.allophoneState());
        if (shortSymbols) {
            if (stateTying[m].empty()) {
                stateTying[m] = Core::form("%s:%d", symbol.substr(0, symbol.find("{")).c_str(), m);
            }
        } else {
            stateTying[m] += symbol + "_";
        }
    }
    if (!shortSymbols) {
        for (std::vector<std::string>::iterator s = stateTying.begin(); s != stateTying.end(); ++s)
            *s = s->substr(0, s->size() - 1);
    }
    OpenFst::SymbolTable *symbols = new OpenFst::SymbolTable("state-sequences");
    symbols->AddSymbol("eps", 0);
    for (u32 s = 0; s < ss.size(); ++s) {
        const StateSequence &seq = ss[s];
        std::string symbol = "";
        for (u32 i = 0; i < seq.nStates(); ++i) {
            symbol += Core::form("%s:%d+", stateTying[seq.state(i).emission_].c_str(), seq.state(i).transition_);
        }
        symbol = symbol.substr(0, symbol.size() - 1);
        if (seq.isInitial()) symbol += "@i";
        if (seq.isFinal()) symbol += "@f";
        if (symbol.empty()) symbol = Core::form("#%d", s);
        symbols->AddSymbol(symbol, OpenFst::convertLabelFromFsa(s));
    }
    return symbols;
}
