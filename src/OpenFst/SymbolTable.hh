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
#ifndef _OPEN_FST_SYMBOL_TABLE_HH
#define _OPEN_FST_SYMBOL_TABLE_HH

#include <Fsa/Alphabet.hh>
#include "Types.hh"

namespace OpenFst {
/**
 * convert label id from Fsa to OpenFst.
 * required because Fsa::Epsilon != OpenFst::Epsilon
 */
inline Label convertLabelFromFsa(Fsa::LabelId l) {
    return l + 1;
}

/**
 * convert label id from OpenFst to Fsa.
 * required because Fsa::Epsilon != OpenFst::Epsilon
 */
inline Fsa::LabelId convertLabelToFsa(Label l) {
    return l - 1;
}

/**
 * convert a Fsa::Alphabet to a OpenFst::SymbolTable
 */
SymbolTable* convertAlphabet(Fsa::ConstAlphabetRef alphabet, const std::string& name, s32 keyOffset = 0);

/**
 * convert an OpenFst::SymbolTable to a Fsa::Alphabet
 */
Fsa::ConstAlphabetRef convertAlphabet(const SymbolTable* symbolTable);
}  // namespace OpenFst

#endif /* _OPEN_FST_SYMBOL_TABLE_HH */
