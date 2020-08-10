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
#include "Common.hh"

#include <Core/StringUtilities.hh>

std::string numberTokens(const std::string& sentence) {
    std::string              numberedString = "";
    std::vector<std::string> list           = Core::split(sentence, " ");
    u32                      counter        = 0;
    for (u32 i = 0; i < list.size(); ++i) {
        if (list[i].length() > 0) {
            numberedString = numberedString + Core::form("%s#%i ", list[i].c_str(), counter);
            counter++;
        }
    }
    return numberedString;
}

std::vector<std::string> numberTokensVector(const std::vector<std::string>& sentence) {
    std::vector<std::string> result;
    u32                      counter = 0;
    for (std::vector<std::string>::const_iterator i = sentence.begin(); i != sentence.end(); ++i) {
        if ((*i).length() > 0) {
            result.push_back(Core::form("%s#%i", (*i).c_str(), counter));
            counter++;
        }
    }
    return result;
}

std::vector<std::string> numberTokensVector(const std::string& sentence) {
    std::string              numberedString = "";
    std::vector<std::string> list           = Core::split(sentence, " ");
    u32                      counter        = 0;
    for (u32 i = 0; i < list.size(); ++i) {
        if (list[i].length() > 0) {
            list[i] = Core::form("%s#%i ", list[i].c_str(), counter);
            counter++;
        }
    }
    return list;
}

std::string cleanBiLangSymbol(const std::string& s) {
    // change *EPS* to $
    std::string result;
    if (s == "*EPS*") {
        result = "$";
    }
    else {
        // remove word index from symbol
        result = s.substr(0, s.rfind("#"));
        // escape strange characters [+~|$]
        // + -> ++
        // _ -> +
        // $ -> $$
        // ~ -> ~~
        // | -> ~
    }
    return result;
}

namespace Fsa {
void writeBiLang(ConstAutomatonRef f, std::ostream& o) {
    Fsa::ConstStateRef    currentState(f->getState(f->initialStateId()));
    Fsa::ConstAlphabetRef inputAlphabet(f->getInputAlphabet());
    Fsa::ConstAlphabetRef outputAlphabet(f->getOutputAlphabet());
    while (currentState->hasArcs()) {
        o << cleanBiLangSymbol(inputAlphabet->symbol(currentState->begin()->input())) << "|" << cleanBiLangSymbol(outputAlphabet->symbol(currentState->begin()->output())) << " ";
        currentState = f->getState(currentState->begin()->target());
    }
    o << std::endl;
}

void writeAachen(ConstAutomatonRef f, std::ostream& o, size_t sentenceNumber, bool oneToOne, double threshold) {
    Fsa::ConstStateRef currentState(f->getState(f->initialStateId()));
    o << "SENT: " << sentenceNumber << std::endl;
    LabelId iPrev(0);
    LabelId jPrev(0);  // There is an error here: if reordering is used. the previous j at sentence beginning might not be "0"
    while (currentState->hasArcs()) {
        LabelId j(currentState->begin()->input());
        LabelId i(currentState->begin()->output());
        bool    isEpsilon(i == Epsilon || j == Epsilon);
        double  w(currentState->begin()->weight());

        if (!oneToOne) {
            i = (i != Epsilon ? i : iPrev);
            j = (j != Epsilon ? j : jPrev);
        }

        if (!(oneToOne && isEpsilon) && !(threshold > 0 && w > threshold)) {
            o << "S "
              << j << " "
              << i
              << std::endl;
        }
        iPrev = i;
        jPrev = j;

        currentState = f->getState(currentState->begin()->target());
    }
    o << std::endl;
}
}  // namespace Fsa
