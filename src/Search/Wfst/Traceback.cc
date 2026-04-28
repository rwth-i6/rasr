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
#include <OpenFst/LabelMap.hh>
#include <OpenFst/SymbolTable.hh>
#include <Search/Traceback.hh>
#include <Search/Wfst/Traceback.hh>

using namespace Search::Wfst;
using OpenFst::Epsilon;

void BestPath::getTraceback(Bliss::LexiconRef lexicon, OutputType outputType,
                            const OpenFst::LabelMap* olabelMap, Traceback* result) const {
    result->clear();
    result->push_back(TracebackItem(0, 0, ScoreVector(0, 0), TracebackItem::Transit()));
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> alphabet = lexicon->lemmaPronunciationAlphabet();
    Core::Ref<const Bliss::LemmaAlphabet>              lemmas   = lexicon->lemmaAlphabet();
    Core::Ref<const Bliss::SyntacticTokenAlphabet>     synt     = lexicon->syntacticTokenAlphabet();
    for (ConstIterator i = begin(); i != end(); ++i) {
        const Bliss::LemmaPronunciation* p     = 0;
        const Bliss::Lemma*              lemma = 0;
        if (i->word != Epsilon) {
            Fsa::LabelId output = OpenFst::convertLabelToFsa(olabelMap ? olabelMap->mapLabel(i->word) : i->word);

            if (outputType == OutputSyntacticToken)
                lemma = *synt->syntacticToken(output)->lemmas().first;
            else if (outputType == OutputLemma)
                lemma = lemmas->lemma(output);

            if (outputType == OutputLemmaPronunciation)
                p = alphabet->lemmaPronunciation(output);
            else if (lemma)
                p = lemma->pronunciations().first;
        }
        result->push_back(TracebackItem(p, i->time, i->score, TracebackItem::Transit()));
    }
}
