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
#ifndef SPEECH_RECOGNIZER_V2_HH
#define SPEECH_RECOGNIZER_V2_HH

#include <Flf/FlfCore/Lattice.hh>
#include <Search/Module.hh>
#include <Search/SearchV2.hh>
#include <Speech/Module.hh>
#include "Network.hh"
#include "SegmentwiseSpeechProcessor.hh"
#include "Speech/ModelCombination.hh"

namespace Flf {

NodeRef createRecognizerNodeV2(const std::string& name, const Core::Configuration& config);

class RecognizerNodeV2 : public Node {
public:
    RecognizerNodeV2(const std::string& name, const Core::Configuration& config);

    virtual ~RecognizerNodeV2() {
        delete searchAlgorithm_;
    }

    void recognizeSegment(const Bliss::SpeechSegment* segment);

    void work();

    virtual void init(const std::vector<std::string>& arguments) override;
    virtual void sync() override;
    virtual void finalize() override;

    virtual ConstSegmentRef sendSegment(Port to) override;
    virtual ConstLatticeRef sendLattice(Port to) override;

private:
    ConstLatticeRef buildLattice(Core::Ref<const Search::LatticeAdaptor> la, std::string segmentName);

    std::pair<ConstLatticeRef, ConstSegmentRef> resultBuffer_;

    Search::SearchAlgorithmV2*     searchAlgorithm_;
    Speech::ModelCombination       modelCombination_;
    SegmentwiseFeatureExtractorRef featureExtractor_;
};

}  // namespace Flf

#endif  // SPEECH_RECOGNIZER_V2_HH
