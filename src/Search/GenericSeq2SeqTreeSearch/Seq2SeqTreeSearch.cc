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

#include "Seq2SeqTreeSearch.hh"

using namespace Search;


Seq2SeqTreeSearchManager::Seq2SeqTreeSearchManager(const Core::Configuration& c)
        : Core::Component(c),
          SearchAlgorithm(c) {
}


bool Seq2SeqTreeSearchManager::setModelCombination(const Speech::ModelCombination& modelCombination) {
}


void Seq2SeqTreeSearchManager::setGrammar(Fsa::ConstAutomatonRef g) {
}

void Seq2SeqTreeSearchManager::resetStatistics() {
}

void Seq2SeqTreeSearchManager::logStatistics() const {
}

void Seq2SeqTreeSearchManager::restart() { 
}


void Seq2SeqTreeSearchManager::getCurrentBestSentence(Traceback &result) const {
}

Core::Ref<const LatticeAdaptor> Seq2SeqTreeSearchManager::getCurrentWordLattice() const {
}

