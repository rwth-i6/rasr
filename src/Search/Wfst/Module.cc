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
#include <Search/Wfst/CreateOperations.hh>
#include <Search/Wfst/FstOperations.hh>
#include <Search/Wfst/IoOperations.hh>
#include <Search/Wfst/Module.hh>
#include <Search/Wfst/UtilityOperations.hh>

namespace Search {
namespace Wfst {

Search::Wfst::Module_::Module_() {
    registerBuilderOperation<Builder::AddNonWordTokens>();
    registerBuilderOperation<Builder::AddPronunciationWeight>();
    registerBuilderOperation<Builder::ArcInputSort>();
    registerBuilderOperation<Builder::ArcOutputSort>();
    registerBuilderOperation<Builder::BuildGrammar>();
    registerBuilderOperation<Builder::BuildLexicon>();
    registerBuilderOperation<Builder::BuildOldLexicon>();
    registerBuilderOperation<Builder::BuildStateTree>();
    registerBuilderOperation<Builder::CheckLabels>();
    registerBuilderOperation<Builder::CloseLexicon>();
    registerBuilderOperation<Builder::Compose>();
    registerBuilderOperation<Builder::Compress>();
    registerBuilderOperation<Builder::ContextBuilder>();
    registerBuilderOperation<Builder::ConvertStateSequences>();
    registerBuilderOperation<Builder::Count>();
    registerBuilderOperation<Builder::CreateLookahead>();
    registerBuilderOperation<Builder::CreateStateSequences>();
    registerBuilderOperation<Builder::CreateStateSequenceSymbols>();
    registerBuilderOperation<Builder::CreateSubwordGrammar>();
    registerBuilderOperation<Builder::Determinize>();
    registerBuilderOperation<Builder::ExpandStates>();
    registerBuilderOperation<Builder::Factorize>();
    registerBuilderOperation<Builder::HmmBuilder>();
    registerBuilderOperation<Builder::Info>();
    registerBuilderOperation<Builder::Invert>();
    registerBuilderOperation<Builder::LabelDecode>();
    registerBuilderOperation<Builder::LabelEncode>();
    registerBuilderOperation<Builder::LemmaMapping>();
    registerBuilderOperation<Builder::Minimize>();
    registerBuilderOperation<Builder::NormalizeEpsilon>();
    registerBuilderOperation<Builder::Pop>();
    registerBuilderOperation<Builder::Project>();
    registerBuilderOperation<Builder::PushLabels>();
    registerBuilderOperation<Builder::PushOutputLabels>();
    registerBuilderOperation<Builder::PushWeights>();
    registerBuilderOperation<Builder::ReachableCompose>();
    registerBuilderOperation<Builder::ReadFsa>();
    registerBuilderOperation<Builder::ReadFst>();
    registerBuilderOperation<Builder::Relabel>();
    registerBuilderOperation<Builder::RemoveEmptyPath>();
    registerBuilderOperation<Builder::RemoveEpsilon>();
    registerBuilderOperation<Builder::RemoveHmmDisambiguators>();
    registerBuilderOperation<Builder::RemovePhoneDisambiguators>();
    registerBuilderOperation<Builder::RemoveWeights>();
    registerBuilderOperation<Builder::RestoreOutputSymbols>();
    registerBuilderOperation<Builder::ScaleLabelWeights>();
    registerBuilderOperation<Builder::ScaleWeights>();
    registerBuilderOperation<Builder::Synchronize>();
    registerBuilderOperation<Builder::WeightEncode>();
    registerBuilderOperation<Builder::WriteFsa>();
    registerBuilderOperation<Builder::WriteFst>();
}

}  // namespace Wfst
}  // namespace Search
