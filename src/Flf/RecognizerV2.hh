/** Copyright 2025 RWTH Aachen University. All rights reserved.
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
#ifndef RECOGNIZER_V2_HH
#define RECOGNIZER_V2_HH

#include <Flf/FlfCore/Lattice.hh>
#include <Search/Module.hh>
#include <Search/SearchV2.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Module.hh>
#include "LatticeHandler.hh"
#include "Network.hh"
#include "SegmentwiseSpeechProcessor.hh"

namespace Flf {

NodeRef createRecognizerNodeV2(std::string const& name, Core::Configuration const& config);

/*
 * Node to run recognition on speech segments using a `SearchAlgorithmV2` internally.
 */
class RecognizerNodeV2 : public Node {
public:
    RecognizerNodeV2(std::string const& name, Core::Configuration const& config);

    // Inherited methods
    virtual void init(std::vector<std::string> const& arguments) override;
    virtual void sync() override;
    virtual void finalize() override;

    virtual ConstSegmentRef sendSegment(Port to) override;
    virtual ConstLatticeRef sendLattice(Port to) override;

private:
    /*
     * Perform recognition of `segment` using `searchAlgorithm_` and store the result in `resultBuffer_`
     */
    void recognizeSegment(const Bliss::SpeechSegment* segment);

    /*
     * Requests input segment and runs recognition on it
     */
    void work();

    ConstLatticeRef latticeResultBuffer_;
    ConstSegmentRef segmentResultBuffer_;

    std::unique_ptr<Flf::LatticeHandler>       latticeHandler_;
    std::unique_ptr<Search::SearchAlgorithmV2> searchAlgorithm_;
    Core::Ref<Speech::ModelCombination>        modelCombination_;
    SegmentwiseFeatureExtractorRef             featureExtractor_;
};

/*
 * Convert an output lattice from `searchAlgorithm_` to an Flf lattice
 */
ConstLatticeRef convertSearchLatticeToFlf(Core::Ref<const Search::LatticeAdaptor> latticeAdaptor, Flf::LatticeHandler const* handler, std::string segmentName, f32 lmScale);

}  // namespace Flf

#endif  // RECOGNIZER_V2_HH
