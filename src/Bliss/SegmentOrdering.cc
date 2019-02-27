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
#include <Bliss/SegmentOrdering.hh>
#include <Core/Application.hh>
#include <Core/CompressedStream.hh>
#include <Math/Random.hh>

using namespace Bliss;

class SegmentOrderingVisitor::CorpusCopy : public Corpus {
public:
    CorpusCopy(const Corpus& c)
            : Corpus(c) {}
    void setParentCorpus(Corpus* corpus) {
        setParent(corpus);
    }
};

class SegmentOrderingVisitor::RecordingCopy : public Recording {
public:
    RecordingCopy(const Recording& r)
            : Recording(r) {}
    void setParentCorpus(Corpus* corpus) {
        setParent(corpus);
    }
};

SegmentOrderingVisitor::SegmentOrderingVisitor()
        : visitor_(0),
          curSegment_(0),
          curRecording_(0),
          shortNameLookup_(false),
          autoShuffle_(false),
          sortByTimeLength_(false),
          sortByTimeLengthChunkSize_(0),
          predefinedOrder_(false) {}

SegmentOrderingVisitor::SegmentOrderingVisitor(const SegmentOrderingVisitor& other)
        : visitor_(other.visitor_),
          curSegment_(0),
          curRecording_(0),
          segmentList_(other.segmentList_),
          shortNameLookup_(other.shortNameLookup_),
          autoShuffle_(other.autoShuffle_),
          shuffleRandomEngine_(other.shuffleRandomEngine_),
          predefinedOrder_(other.predefinedOrder_) {
    // Note: Really copying this is complicated because we would need to do a deep copy.
    // Otherwise, our destructor would double-free elements.
    // However, the only use-case of copying this is when we calculate the total
    // segment count in CorpusDescription. We actually intend that this is in
    // its initial state.
}

SegmentOrderingVisitor* SegmentOrderingVisitor::copy() {
    return new SegmentOrderingVisitor(*this);
}

SegmentOrderingVisitor::~SegmentOrderingVisitor() {
    for (SpeakerMap::const_iterator s = speakers_.begin(); s != speakers_.end(); ++s)
        delete s->second;
    for (ConditionMap::const_iterator c = conditions_.begin(); c != conditions_.end(); ++c)
        delete c->second;
    for (SegmentMap::const_iterator s = segments_.begin(); s != segments_.end(); ++s)
        delete s->second;
    for (std::vector<Recording*>::const_iterator r = recordings_.begin(); r != recordings_.end(); ++r)
        delete *r;
    for (std::vector<Corpus*>::const_iterator c = corpus_.begin(); c != corpus_.end(); ++c)
        delete *c;
}

template<class T>
inline const std::string SegmentOrderingVisitor::_getName(const T* entry) {
    if (shortNameLookup_) {
        return entry->name();
    }
    else {
        return entry->fullName();
    }
}

void SegmentOrderingVisitor::addSegment(Segment* segment) {
    verify(!recordings_.empty());
    segment->setRecording(recordings_.back());
    if (segment->condition())
        updateCondition(segment);
    std::string name = _getName(segment);
    if (segments_.find(name) != segments_.end()) {
        Core::Application::us()->error("can not add segment, because it is already present in segment list: ") << name;
    }
    segments_.insert(SegmentMap::value_type(name, segment));
    if (!predefinedOrder_)
        segmentList_.push_back(name);
}

template<class T, class C>
const T* SegmentOrderingVisitor::updateSegmentData(Segment* segment, const T* entry, C& map) {
    // Not needed for the root corpus.
    if (entry->parent() != curCorpus_.front().first) {
        typename C::const_iterator c = map.find(_getName(entry));
        if (c != map.end()) {
            entry = c->second;
        }
        else {
            T* add = new T(*entry);
            if (entry->parent() == curRecording_) {
                add->setParent(recordings_.back());
            }
            else if (entry->parent() == curSegment_) {
                add->setParent(segment);
            }
            else {
                for (CorpusMap::const_reverse_iterator i = curCorpus_.rbegin();
                     i != curCorpus_.rend(); ++i) {
                    if (entry->parent() == i->first) {
                        add->setParent(i->second);
                        break;
                    }
                }
            }
            map.insert(std::make_pair(_getName(entry), add));
            entry = add;
        }
    }
    return entry;
}

void SegmentOrderingVisitor::updateCondition(Segment* segment) {
    const AcousticCondition* cond = segment->condition();
    verify(cond);
    cond = updateSegmentData(segment, cond, conditions_);
    segment->setCondition(cond);
}

void SegmentOrderingVisitor::updateSpeaker(SpeechSegment* segment) {
    const Speaker* speaker = segment->speaker();
    verify(speaker);
    speaker = updateSegmentData(segment, speaker, speakers_);
    segment->setSpeaker(speaker);
}

Segment* SegmentOrderingVisitor::getSegmentByName(const std::string& name) {
    SegmentMap::const_iterator i = segments_.find(name);
    if (i == segments_.end())
        return NULL;
    return i->second;
}

void SegmentOrderingVisitor::setSegmentList(const std::string& filename) {
    verify(!autoShuffle_);  // we will automatically fill segmentList_
    verify(!filename.empty());
    require(!predefinedOrder_);  // not yet loaded

    Core::CompressedInputStream* cis = new Core::CompressedInputStream(filename.c_str());
    Core::TextInputStream        is(cis);
    if (!is)
        Core::Application::us()->criticalError("Failed to open segment list \"%s\".", filename.c_str());
    std::string s;
    while (Core::getline(is, s) != EOF) {
        if ((s.size() == 0) || (s.at(0) == '#'))
            continue;
        Core::stripWhitespace(s);
        segmentList_.push_back(s);
    }

    predefinedOrder_ = true;
}

void SegmentOrderingVisitor::setSortByTimeLength(bool enabled, long chunkSize) {
    sortByTimeLength_          = enabled;
    sortByTimeLengthChunkSize_ = chunkSize;
}

void SegmentOrderingVisitor::visitSpeechSegment(SpeechSegment* s) {
    curSegment_            = s;
    SpeechSegment* segment = new SpeechSegment(*s);
    if (s->speaker())
        updateSpeaker(segment);
    addSegment(segment);
}

void SegmentOrderingVisitor::enterRecording(Recording* r) {
    RecordingCopy* recording = new RecordingCopy(*r);
    recording->setParentCorpus(curCorpus_.back().second);
    recordings_.push_back(recording);
    curRecording_ = r;
}

void SegmentOrderingVisitor::enterCorpus(Corpus* c) {
    if (!curCorpus_.empty()) {
        // subcorpus
        CorpusCopy* corpus = new CorpusCopy(*c);
        corpus->setParentCorpus(curCorpus_.back().second);
        corpus_.push_back(corpus);
        curCorpus_.push_back(std::make_pair(c, corpus));
    }
    else {
        // root corpus
        curCorpus_.push_back(std::make_pair(c, c));
    }
}

void SegmentOrderingVisitor::prepareSegmentLoop() {
    if (autoShuffle_)
        std::shuffle(segmentList_.begin(), segmentList_.end(), shuffleRandomEngine_);

    if (sortByTimeLength_) {
        size_t n0 = 0;
        while (n0 < segmentList_.size()) {
            // Sort remaining segments - or chunk of it.
            size_t n = segmentList_.size() - n0;
            if (sortByTimeLengthChunkSize_ > 0)
                n = std::min(n, (size_t)sortByTimeLengthChunkSize_);

            // Stable sort so that we keep the order deterministic.
            std::stable_sort(
                    &segmentList_[n0], &segmentList_[n0] + n,
                    [this](const std::string& sn0, const std::string& sn1) {
                        Bliss::Segment* s0 = this->getSegmentByName(sn0);
                        Bliss::Segment* s1 = this->getSegmentByName(sn1);
                        require(s0 && s1);
                        Bliss::Time st0 = s0->end() - s0->start();
                        Bliss::Time st1 = s1->end() - s1->start();
                        return st0 < st1;
                    });

            n0 += n;
        }
    }
}

void SegmentOrderingVisitor::finishSegmentLoop() {
    if (!predefinedOrder_)
        // We will add them again when we iterate through the corpus.
        segmentList_.clear();
}

void SegmentOrderingVisitor::leaveCorpus(Corpus* corpus) {
    curCorpus_.pop_back();
    if (!curCorpus_.empty()) {
        // not the root corpus
        return;
    }
    // corpus is the root corpus. We don't have our own copy of this one.

    prepareSegmentLoop();

    CustomCorpusGuide corpusGuide(this, /* root */ corpus);
    for (std::vector<std::string>::const_iterator name = segmentList_.begin(); name != segmentList_.end(); ++name)
        corpusGuide.showSegmentByName(*name);

    finishSegmentLoop();
}

SegmentOrderingVisitor::CustomCorpusGuide::CustomCorpusGuide(SegmentOrderingVisitor* parent, Corpus* rootCorpus)
        : parent_(parent), rootCorpus_(rootCorpus), curCorpus_(rootCorpus_), curRecording_(0) {
    // Enter the root corpus.
    parent_->visitor_->enterCorpus(rootCorpus_);
}

SegmentOrderingVisitor::CustomCorpusGuide::~CustomCorpusGuide() {
    if (curRecording_)
        parent_->visitor_->leaveRecording(curRecording_);
    if (curCorpus_ != rootCorpus_)
        // Leave the sub-corpus.
        parent_->visitor_->leaveCorpus(curCorpus_);

    // Leave the root corpus.
    parent_->visitor_->leaveCorpus(rootCorpus_);
}

void SegmentOrderingVisitor::CustomCorpusGuide::showSegmentByName(const std::string& segmentName) {
    Segment* segment = parent_->getSegmentByName(segmentName);
    if (!segment) {
        Core::Application::us()->error("segment '%s' not found", segmentName.c_str());
        return;
    }
    showSegment(segment);
}

void SegmentOrderingVisitor::CustomCorpusGuide::showSegment(Segment* segment) {
    if (segment->recording() != curRecording_) {
        if (curRecording_)
            parent_->visitor_->leaveRecording(curRecording_);
        curRecording_     = segment->recording();
        Corpus* recCorpus = static_cast<Corpus*>(curRecording_->parent());
        if (recCorpus != curCorpus_) {
            if (curCorpus_ != rootCorpus_) {
                // Leave the sub-corpus.
                parent_->visitor_->leaveCorpus(curCorpus_);
            }
            if (recCorpus != rootCorpus_) {
                // This recCorpus is a sub-corpus.
                parent_->visitor_->enterCorpus(recCorpus);
            }
            curCorpus_ = recCorpus;
        }
        parent_->visitor_->enterRecording(segment->recording());
    }

    segment->accept(parent_->visitor_);
}
