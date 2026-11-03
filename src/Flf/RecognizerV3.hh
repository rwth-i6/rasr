/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef RECOGNIZER_V3_HH
#define RECOGNIZER_V3_HH

#include <chrono>
#include <deque>

#include <Flf/FlfCore/Lattice.hh>
#include <Search/Module.hh>
#include <Search/SearchV3.hh>
#include <Speech/ModelCombination.hh>

#include "LatticeHandler.hh"
#include "Network.hh"
#include "Segment.hh"
#include "SegmentwiseSpeechProcessor.hh"

namespace Flf {

NodeRef createRecognizerNodeV3(std::string const& name, Core::Configuration const& config);

/*
 * Node to run batched recognition of several speech segments at the same time using a `SearchAlgorithmV3`.
 *
 * In contrast to the other nodes, this node doesn't pull its input segments from the network. Instead it is driven by
 * the `BatchedCorpusProcessor` (which is used automatically if the network contains a `recognizer-v3` node):
 *  - For each corpus segment, the processor calls `addSegment`, which extracts the features of the segment and passes
 *    them to the search. Since this happens while the corpus visitor is at the segment, the feature extraction works
 *    as for the other recognizer nodes.
 *  - The processor calls `decodeStep` to decode all active segments together until the oldest segment (in corpus order)
 *    is finished. Then `prepareNextResult` makes its result the output of this node and the network is processed once
 *    for this segment. The `speech-segment` node of the network provides a copy of the segment information.
 *  - At most `max-active-segments` segments are decoded at the same time.
 * This way, all nodes behind the recognizer (e.g. dumping CTMs or writing lattices) work unchanged and receive the
 * results in corpus order.
 */
class RecognizerNodeV3 : public Node {
public:
    RecognizerNodeV3(std::string const& name, Core::Configuration const& config);

    // Inherited methods
    virtual void init(std::vector<std::string> const& arguments) override;
    virtual void sync() override;
    virtual void finalize() override;

    virtual ConstSegmentRef sendSegment(Port to) override;
    virtual ConstLatticeRef sendLattice(Port to) override;

    /*
     * Interface for `BatchedCorpusProcessor`
     */

    // Maximum number of segments that are decoded at the same time
    size_t maxActiveSegments() const;

    // Number of segments that are not finished yet
    size_t numActiveSegments() const;

    // Extract the features of the segment and add them to the search
    void addSegment(Bliss::SpeechSegment const* segment);

    // Whether there are segments whose result hasn't been sent yet
    bool hasPendingSegments() const;

    // Whether the oldest pending segment is finished, i.e. its result can be sent
    bool nextResultReady() const;

    // Decode one step for all active segments and collect the results of finished segments
    void decodeStep();

    // Make the result of the oldest pending segment the output of this node and return the segment information
    ConstSegmentRef prepareNextResult();

private:
    /*
     * A segment whose result hasn't been sent yet
     */
    struct PendingSegment {
        Search::SearchAlgorithmV3::SegmentHandle handle;
        ConstSegmentRef                          segment;
        std::string                              name;
        std::string                              referenceOrth;
        bool                                     finished;
        ConstLatticeRef                          lattice;
        Core::Ref<Search::Traceback const>       traceback;
        std::chrono::steady_clock::time_point    startTime;
        double                                   recognitionTime;  // Time from adding the segment until it was finished in ms
        double                                   signalDuration;   // In ms
    };

    // Collect the results of all segments that are finished and remove them from the search
    void collectFinishedSegments();

    // Log the result of a segment
    void logResult(PendingSegment const& pendingSegment);

    std::deque<PendingSegment> pendingSegments_;

    ConstLatticeRef latticeResultBuffer_;
    ConstSegmentRef segmentResultBuffer_;

    std::unique_ptr<Flf::LatticeHandler>       latticeHandler_;
    std::unique_ptr<Search::SearchAlgorithmV3> searchAlgorithm_;
    Core::Ref<Speech::ModelCombination>        modelCombination_;
    SegmentwiseFeatureExtractorRef             featureExtractor_;

    double totalRecognitionTime_;  // Wall time spent in feature extraction and decoding in ms
    double totalSignalDuration_;   // In ms
};

// Copy of the segment information that doesn't refer to the Bliss segment (which is only valid while the corpus
// visitor is at the segment)
ConstSegmentRef detachedSegment(Bliss::SpeechSegment const* blissSegment);

}  // namespace Flf

#endif  // RECOGNIZER_V3_HH
