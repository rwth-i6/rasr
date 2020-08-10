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
#ifndef _BLISS_SEGMENT_ORDERING_HH
#define _BLISS_SEGMENT_ORDERING_HH

#include <Bliss/CorpusDescription.hh>
#include <random>

namespace Bliss {

/**
 * Changes the order of processed segments according to a given
 * segment id list (full segment names).
 * This visitor needs to make a copy of each sub-corpus, recording, and
 * segment object, because they are immediately deleted by
 * the CorpusDescriptionParser
 */
class SegmentOrderingVisitor : public CorpusVisitor {
public:
    SegmentOrderingVisitor();
    SegmentOrderingVisitor(const SegmentOrderingVisitor&);
    virtual ~SegmentOrderingVisitor();
    virtual SegmentOrderingVisitor* copy();

    void setVisitor(CorpusVisitor* v) {
        visitor_ = v;
    }
    virtual void setShortNameLookup(bool enabled) {
        shortNameLookup_ = enabled;
    }
    virtual void setAutoShuffle(bool enabled) {
        autoShuffle_ = enabled;
    }
    void shuffleRandomSeed(u32 seed) {
        shuffleRandomEngine_.seed(seed);
    }

    virtual void setSegmentList(const std::string& filename);
    void         setSortByTimeLength(bool enabled, long chunkSize);

    virtual void enterRecording(Recording* r);
    virtual void enterCorpus(Corpus* c);
    virtual void leaveCorpus(Corpus* corpus);
    virtual void visitSegment(Segment* s) {
        curSegment_      = s;
        Segment* segment = new Segment(*s);
        addSegment(segment);
    }
    virtual void visitSpeechSegment(SpeechSegment* s);

private:
    class CorpusCopy;
    class RecordingCopy;

    void addSegment(Segment* segment);
    template<class T, class C>
    const T* updateSegmentData(Segment* segment, const T* entry, C& map);
    void     updateCondition(Segment* segment);
    void     updateSpeaker(SpeechSegment* segment);
    template<class T>
    const std::string _getName(const T* entry);

protected:
    Segment* getSegmentByName(const std::string& name);
    void     prepareSegmentLoop();
    void     finishSegmentLoop();

    struct CustomCorpusGuide {
        SegmentOrderingVisitor* parent_;
        Corpus*                 rootCorpus_;
        Corpus*                 curCorpus_;
        Recording*              curRecording_;

        CustomCorpusGuide(SegmentOrderingVisitor* parent, Corpus* rootCorpus);
        ~CustomCorpusGuide();
        void showSegment(Segment* segment);
        void showSegmentByName(const std::string& segmentName);
    };

    typedef Core::StringHashMap<Segment*>            SegmentMap;
    typedef Core::StringHashMap<Speaker*>            SpeakerMap;
    typedef Core::StringHashMap<AcousticCondition*>  ConditionMap;
    typedef std::vector<std::pair<Corpus*, Corpus*>> CorpusMap;

    CorpusVisitor*           visitor_;
    std::vector<Recording*>  recordings_;
    std::vector<Corpus*>     corpus_;  // own copies of sub-corpora. kept list for cleanup.
    CorpusMap                curCorpus_;
    SpeakerMap               speakers_;
    ConditionMap             conditions_;
    const Segment*           curSegment_;
    const Recording*         curRecording_;
    SegmentMap               segments_;
    std::vector<std::string> segmentList_;
    bool                     shortNameLookup_;
    bool                     autoShuffle_;
    std::mt19937             shuffleRandomEngine_;
    bool                     sortByTimeLength_;
    long                     sortByTimeLengthChunkSize_;
    bool                     predefinedOrder_;  // set via setSegmentList
};

}  // namespace Bliss

#endif  // _BLISS_SEGMENT_ORDERING_HH
