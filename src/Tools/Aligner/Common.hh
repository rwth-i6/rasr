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
#ifndef ALIGNER_COMMON_HH
#define ALIGNER_COMMON_HH

#include <string>
#include <vector>

#include <Fsa/Automaton.hh>

std::string              numberTokens(const std::string& sentence);
std::vector<std::string> numberTokensVector(const std::vector<std::string>& sentence);
std::vector<std::string> numberTokensVector(const std::string& sentence);

namespace Fsa {
class ChangePropertiesAutomaton : public SlaveAutomaton {
public:
    ChangePropertiesAutomaton(ConstAutomatonRef f, Property properties)
            : SlaveAutomaton(f) {
        setProperties(properties);
    }
    virtual std::string describe() const {
        return fsa_->describe();
    }
};

void writeBiLang(ConstAutomatonRef f, std::ostream& o);
void writeAachen(ConstAutomatonRef f, std::ostream& o, size_t sentenceNumber, bool oneToOne = false, double threshold = 0);

}  // namespace Fsa

#endif
