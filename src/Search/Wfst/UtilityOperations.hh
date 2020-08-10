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
#ifndef _SEARCH_UTILITY_OPERATIONS_HH
#define _SEARCH_UTILITY_OPERATIONS_HH

#include <Search/Wfst/Builder.hh>

namespace Search {
namespace Wfst {

class StateSequence;
class StateSequenceList;
class TiedStateSequenceMap;

namespace Builder {

/**
 * replace all (input) HMM disambiguation symbols by epsilon.
 */
class RemoveHmmDisambiguators : public SleeveOperation {
public:
    RemoveHmmDisambiguators(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "remove-hmm-disambiguators";
    }
};

/**
 * logs number of states and arcs
 */
class Info : public SleeveOperation {
public:
    Info(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "info";
    }
};

class Count : public SleeveOperation {
    static const Core::ParameterString paramStateSequences;

public:
    Count(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "count";
    }
};

class CreateStateSequenceSymbols : public SleeveOperation {
    static const Core::ParameterString paramStateSequences;
    static const Core::ParameterBool   paramShortSymbols;

public:
    CreateStateSequenceSymbols(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r) {}

protected:
    virtual AutomatonRef process();

private:
    OpenFst::SymbolTable* createSymbols(const StateSequenceList& ss) const;

public:
    static std::string name() {
        return "state-sequence-symbols";
    }
};

}  // namespace Builder
}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_UTILITY_OPERATIONS_HH
