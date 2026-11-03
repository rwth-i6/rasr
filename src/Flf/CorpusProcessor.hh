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
#ifndef _FLF_CORPUS_PROCESSOR_HH
#define _FLF_CORPUS_PROCESSOR_HH

#include <Speech/CorpusProcessor.hh>
#include <Speech/CorpusVisitor.hh>

#include "Processor.hh"
#include "Segment.hh"

namespace Flf {

class SpeechSegmentNode;
class RecognizerNodeV3;

class CorpusProcessor : public Processor, public Speech::CorpusProcessor {
private:
    typedef std::vector<SpeechSegmentNode*> SpeechSegmentNodeList;

private:
    Speech::CorpusVisitor* corpusVisitor_;
    SpeechSegmentNodeList* speechSegNodes_;
    bool                   good_;

private:
    CorpusProcessor(const Core::Configuration& config,
                    Network* network, SpeechSegmentNodeList* speechSegNodes);

public:
    virtual ~CorpusProcessor();
    virtual bool init(const std::vector<std::string>& arguments);
    virtual void run();
    void         processSpeechSegment(Bliss::SpeechSegment* segment);

    static Processor* create(const Core::Configuration& config, Network* network);
};

/*
 * Corpus processor for networks with a `recognizer-v3` node, which decodes several segments at the same time.
 *
 * For each corpus segment, the features are extracted and passed to the recognizer node. Up to `max-active-segments`
 * segments are decoded together; whenever the oldest segment is finished, the network is processed once for this
 * segment: the `speech-segment` nodes send a copy of its segment information (the Bliss segment itself is not
 * available anymore) and the recognizer node sends its lattice. So the results are processed in corpus order.
 */
class BatchedCorpusProcessor : public Processor, public Speech::CorpusProcessor {
private:
    typedef std::vector<SpeechSegmentNode*> SpeechSegmentNodeList;

private:
    Speech::CorpusVisitor* corpusVisitor_;
    SpeechSegmentNodeList  speechSegNodes_;
    RecognizerNodeV3*      recognizer_;
    bool                   good_;

    BatchedCorpusProcessor(const Core::Configuration& config, Network* network, SpeechSegmentNodeList const& speechSegNodes, RecognizerNodeV3* recognizer);

    // Decode and process the network for all finished segments in corpus order.
    // If `all` is false, decode only as long as the maximum number of active segments is reached.
    void processResults(bool all);

public:
    virtual ~BatchedCorpusProcessor();
    virtual bool init(const std::vector<std::string>& arguments);
    virtual void run();
    void         processSpeechSegment(Bliss::SpeechSegment* segment);

    // Returns a processor if the network contains a `recognizer-v3` node
    static Processor* create(const Core::Configuration& config, Network* network);
};

/*
 * Port 0: ConstSegmentRef
 * Port 1: void* (const Bliss::SpeechSegment*)
 */
class SpeechSegmentNode : public Node {
private:
    const Bliss::SpeechSegment* blissSpeechSegment_;
    ConstSegmentRef             segment_;

public:
    SpeechSegmentNode(const std::string& name, const Core::Configuration& config);
    virtual void init(const std::vector<std::string>& arguments);
    virtual void sync();

    bool synced() const {
        return !blissSpeechSegment_;
    }

    virtual bool good() {
        return true;
    }

    void setSpeechSegment(Bliss::SpeechSegment* blissSpeechSegment) {
        blissSpeechSegment_ = blissSpeechSegment;
    }

    // Set the segment information without a Bliss segment (used for batched recognition, where the Bliss segment
    // isn't available anymore when the network is processed for a segment)
    void setSegment(ConstSegmentRef segment) {
        blissSpeechSegment_ = 0;
        segment_            = segment;
    }

    virtual ConstSegmentRef sendSegment(Port to);
    virtual const void*     sendData(Port to);
};
NodeRef createSpeechSegmentNode(const std::string& name, const Core::Configuration& config);

}  // namespace Flf

#endif  // _FLF_CORPUS_PROCESSOR_HH
