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
#include "ConditionalLexiconPlain.hh"
#include <Core/StringUtilities.hh>

namespace Translation {

Core::ParameterString ConditionalLexiconPlain::paramFilename_(
        "file", "lexicon file", "");
Core::ParameterFloat ConditionalLexiconPlain::paramFloor_(
        "floor", "lexicon floor value", 99);

void ConditionalLexiconPlain::read() {
    Core::CompressedInputStream is(lexiconFilename_);
    std::cerr << "lexicon.file = " << lexiconFilename_ << std::endl;
    if (is)
        this->read(is);
}

void ConditionalLexiconPlain::read(std::istream& is) {
    std::string line;
    unsigned    nEntries = 0;

    while (std::getline(is, line)) {
        std::vector<std::string> splitLine = Core::split(line, " ");
        size_t                   lexiconIndex;
        Core::strconv(splitLine[0], lexiconIndex);

        if (lexica_.size() < lexiconIndex + 1) {
            std::cerr << "resizing lexicon to " << lexiconIndex + 1 << std::endl;

            //initialize new lexica
            size_t oldSize = lexica_.size();
            lexica_.resize(lexiconIndex + 1);
            for (size_t i = oldSize; i <= lexiconIndex; ++i) {
                lexica_[i] = LexiconRef(new Lexicon);
            }
            std::cerr << "lexica_.size()=" << lexica_.size() << std::endl;
        }

        Cost value;
        Core::strconv(splitLine[1], value);

        std::vector<Fsa::LabelId> key;
        for (size_t i = 2; i < splitLine.size(); ++i) {
            Fsa::LabelId symbol;
            if (splitLine[i] != "NULL")
                symbol = tokens_->addSymbol(splitLine[i]);
            else
                symbol = Fsa::Epsilon;
            key.push_back(symbol);
        }

        lexica_[lexiconIndex]->store(key, value);
        ++nEntries;
    }
    std::cerr << "read " << nEntries << " entries" << std::endl;
}

}  // namespace Translation
