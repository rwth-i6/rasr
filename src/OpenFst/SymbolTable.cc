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
#include "SymbolTable.hh"

namespace OpenFst {

SymbolTable* convertAlphabet(Fsa::ConstAlphabetRef alphabet, const std::string& name, s32 keyOffset) {
    SymbolTable* symbols = new SymbolTable(name);
    symbols->AddSymbol(alphabet->specialSymbol(Fsa::Epsilon), Epsilon);
    for (Fsa::Alphabet::const_iterator i = alphabet->begin(); i != alphabet->end(); ++i) {
        symbols->AddSymbol(*i, convertLabelFromFsa(Fsa::LabelId(i)) + keyOffset);
    }
    return symbols;
}

Fsa::ConstAlphabetRef convertAlphabet(const SymbolTable* symbolTable) {
    if (!symbolTable)
        return Fsa::ConstAlphabetRef();

    Fsa::StaticAlphabet* alphabet = new Fsa::StaticAlphabet();
    for (FstLib::SymbolTableIterator i(*symbolTable); !i.Done(); i.Next()) {
        alphabet->addIndexedSymbol(i.Symbol(), convertLabelToFsa(i.Value()));
    }
    return Fsa::ConstAlphabetRef(alphabet);
}

}  // namespace OpenFst
