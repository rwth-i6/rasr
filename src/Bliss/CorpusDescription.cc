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
#include "CorpusDescription.hh"
#include "CorpusParser.hh"
#include "SegmentOrdering.hh"

#include <Core/Application.hh>
#include <Core/CompressedStream.hh>
#include <Core/Hash.hh>
#include <Core/Parameter.hh>
#include <Core/ProgressIndicator.hh>
#include <Core/StringUtilities.hh>
#include <Core/TextStream.hh>
#include <Modules.hh>

#ifdef MODULE_THEANO_INTERFACE
#include "TheanoSegmentOrderingVisitor.hh"
#endif
#ifdef MODULE_PYTHON
#include "PythonSegmentOrdering.hh"
#endif

using namespace Bliss;

// ========================================================================
// class NamedCorpusEntity

// initialization of static members:
const char* const NamedCorpusEntity::anonymous = "ANONYMOUS";

NamedCorpusEntity::NamedCorpusEntity(ParentEntity* _parent)
        : parent_(_parent),
          name_(anonymous),
          removePrefix_(std::string()) {}

std::string NamedCorpusEntity::fullName() const {
    std::string new_name;
    if (parent())
        new_name = parent()->fullName() + "/" + name();
    else
        new_name = name();
    if (!removePrefix_.empty()) {
        auto res = std::mismatch(removePrefix_.begin(), removePrefix_.end(), new_name.begin());
        if (res.first == removePrefix_.end()){
            new_name = std::string(res.second, new_name.end());
        }
    }
    return new_name;
}

// ========================================================================
// class Speaker

const char* Speaker::genderId[] = {
        "unknown",
        "male",
        "female"};

Speaker::Speaker(ParentEntity* _parent)
        : NamedCorpusEntity(_parent),
          gender_(unknown) {}

// ========================================================================
// class AcousticCondition

AcousticCondition::AcousticCondition(ParentEntity* _parent)
        : NamedCorpusEntity(_parent) {}

// ========================================================================
// class CorpusSection

CorpusSection::CorpusSection(CorpusSection* _parent)
        : ParentEntity(_parent),
          level_((_parent) ? (_parent->level() + 1) : 0),
          defaultCondition_(0),
          defaultSpeaker_(0) {}

const Speaker* CorpusSection::defaultSpeaker() const {
    if (defaultSpeaker_) {
        return defaultSpeaker_;
    }
    else if (parent()) {
        return parent()->defaultSpeaker();
    }
    else {
        return 0;
    }
}

const AcousticCondition* CorpusSection::defaultCondition() const {
    if (defaultCondition_) {
        return defaultCondition_;
    }
    else if (parent()) {
        return parent()->defaultCondition();
    }
    else {
        return 0;
    }
}

// ========================================================================
// class Corpus
Corpus::Corpus(Corpus* parentCorpus)
        : CorpusSection(parentCorpus) {}

// ========================================================================
// class Recording

Recording::Recording(Corpus* corpus)
        : CorpusSection(corpus),
          duration_(0) {
    require(corpus);
}

// ========================================================================
// class Segment

const char* Segment::typeId[] = {
        "speech",
        "other"};

Segment::Segment(Type _type, Recording* _recording)
        : ParentEntity(_recording),
          recording_(_recording),
          type_(_type),
          start_(0),
          end_(0),
          track_(0),
          condition_(0) {
    require(_recording);
}

void Segment::accept(SegmentVisitor* v) {
    v->visitSegment(this);
}

// ========================================================================
// class SpeechSegment

SpeechSegment::SpeechSegment(Recording* _recording)
        : Segment(typeSpeech, _recording),
          speaker_(0) {}

void SpeechSegment::accept(SegmentVisitor* v) {
    v->visitSpeechSegment(this);
}

// ========================================================================
// class CorpusDescription

const Core::ParameterString CorpusDescription::paramFilename(
        "file",
        "file name for segment whitelist",
        "");
const Core::ParameterBool CorpusDescription::paramAllowEmptyWhitelist(
        "allow-empty-whitelist",
        "allow empty segment whitelist. otherwise we would error if the list is empty.",
        false);
const Core::ParameterString CorpusDescription::paramEncoding(
        "encoding",
        "encoding",
        "utf-8");
const Core::ParameterInt CorpusDescription::paramPartition(
        "partition",
        "divide corpus into partitions with (approximately) equal number of segments",
        0, 0);
const Core::ParameterInt CorpusDescription::paramPartitionSelection(
        "select-partition",
        "select a partition of the corpus",
        0, 0);
const Core::ParameterInt CorpusDescription::paramSkipFirstSegments(
        "skip-first-segments",
        "skip the first N segments (counted after partitioning)",
        0, 0);
const Core::ParameterStringVector CorpusDescription::paramSegmentsToSkip(
        "segments-to-skip",
        "skip the segments in this list");
const Core::ParameterBool CorpusDescription::paramRecordingBasedPartition(
        "recording-based-partition",
        "create corpus partitions based on recordings instead of segments",
        false);
const Core::ParameterBool CorpusDescription::paramProgressReportingSegmentOrth(
        "report-segment-orth",
        "output also segment orth in progress report",
        false);
const Core::ParameterString CorpusDescription::paramSegmentOrder(
        "segment-order",
        "file defining the order of processed segments", "");
const Core::ParameterBool CorpusDescription::paramSegmentOrderLookupName(
        "segment-order-look-up-short-name",
        "Look up using full or short name (segment only)", false);
const Core::ParameterBool CorpusDescription::paramSegmentOrderShuffle(
        "segment-order-shuffle",
        "Automatically shuffle segment list.", false);
const Core::ParameterInt CorpusDescription::paramSegmentOrderShuffleSeed(
        "segment-order-shuffle-seed",
        "Use this seed for the random engine for auto-shuffle.", -1);
const Core::ParameterBool CorpusDescription::paramSegmentOrderSortByTimeLength(
        "segment-order-sort-by-time-length",
        "Sort segment list by time-length of each segment.", false);
const Core::ParameterInt CorpusDescription::paramSegmentOrderSortByTimeLengthChunkSize(
        "segment-order-sort-by-time-length-chunk-size",
        "Only sort each such chunk of segments. (-1 = disabled)", -1);
const Core::ParameterBool CorpusDescription::paramTheanoSegmentOrder(
        "theano-segment-order",
        "use theano to specify the order of segments over shared memory",
        false);
const Core::ParameterBool CorpusDescription::paramPythonSegmentOrder(
        "python-segment-order",
        "use Python to specify the order of segments",
        false);
const Core::ParameterString CorpusDescription::paramPythonSegmentOrderModPath(
        "python-segment-order-pymod-path",
        "the path where the Python module is (added to sys.path)",
        "");
const Core::ParameterString CorpusDescription::paramPythonSegmentOrderModName(
        "python-segment-order-pymod-name",
        "the Python module name. does `import <modname>`",
        "");
const Core::ParameterString CorpusDescription::paramPythonSegmentOrderConfig(
        "python-segment-order-config",
        "config string, passed to the Python module init",
        "");

// ---------------------------------------------------------------------------
class CorpusDescription::SegmentPartitionVisitorAdaptor : public CorpusVisitor {
private:
    u32                 segmentIndex_, recordingIndex_, nPartitions_, selectedPartition_, nSkippedSegments_;
    Core::StringHashSet segmentsToSkip_;  // blacklist
    Core::StringHashSet segmentsToKeep_;  // whitelist
    Recording*          currentRecording_;
    bool                isVisitorInCurrentRecording_;
    bool                recordingBasedPartitions_;
    CorpusVisitor*      visitor_;

    bool shouldVisit(Segment* s) {
        u32  index                    = recordingBasedPartitions_ ? recordingIndex_ : segmentIndex_;
        bool isSelectedPartition      = (index % nPartitions_) == selectedPartition_;
        bool hasSkippedEnoughSegments = ((segmentIndex_ / nPartitions_) >= nSkippedSegments_);
        bool shouldNotSkip            = segmentsToSkip_.empty() || (segmentsToSkip_.find(s->fullName()) == segmentsToSkip_.end());
        bool shouldKeep               = segmentsToKeep_.empty() || (segmentsToKeep_.find(s->fullName()) != segmentsToKeep_.end()) || (segmentsToKeep_.find(s->name()) != segmentsToKeep_.end());
        ++segmentIndex_;
        if (isSelectedPartition && hasSkippedEnoughSegments && shouldNotSkip && shouldKeep) {
            if (!isVisitorInCurrentRecording_) {
                visitor_->enterRecording(currentRecording_);
                isVisitorInCurrentRecording_ = true;
            }
            return true;
        }
        else {
            return false;
        }
    }

public:
    SegmentPartitionVisitorAdaptor()
            : visitor_(0) {
        nPartitions_              = 1;
        selectedPartition_        = 0;
        recordingBasedPartitions_ = false;
        nSkippedSegments_         = 0;
    }
    void                       loadSegmentList(const std::string& filename, const std::string& encoding);
    const Core::StringHashSet& segmentsToKeep() const {
        return segmentsToKeep_;
    }
    void setPartitioning(u32 nPartitions, u32 selectedPartition, bool recordingBased = false) {
        require(selectedPartition < nPartitions);
        nPartitions_              = nPartitions;
        selectedPartition_        = selectedPartition;
        recordingBasedPartitions_ = recordingBased;
    }
    void setSkippedSegments(u32 nSkippedSegments) {
        nSkippedSegments_ = nSkippedSegments;
    }
    void setSegmentsToSkip(const Core::StringHashSet& segmentsToSkip) {
        segmentsToSkip_ = segmentsToSkip;
    }
    void setVisitor(CorpusVisitor* v) {
        visitor_ = v;
    }
    virtual void visitSegment(Segment* s) {
        if (shouldVisit(s))
            visitor_->visitSegment(s);
    }
    virtual void visitSpeechSegment(SpeechSegment* s) {
        if (shouldVisit(s))
            visitor_->visitSpeechSegment(s);
    }
    virtual void enterRecording(Recording* r) {
        currentRecording_ = r;
        ++recordingIndex_;
        isVisitorInCurrentRecording_ = false;
    }
    virtual void leaveRecording(Recording* r) {
        if (isVisitorInCurrentRecording_)
            visitor_->leaveRecording(r);
        currentRecording_ = 0;
    }
    virtual void enterCorpus(Corpus* c) {
        if (!c->level()) {
            segmentIndex_   = 0;
            recordingIndex_ = 0;
        }
        visitor_->enterCorpus(c);
    }
    virtual void leaveCorpus(Corpus* c) {
        visitor_->leaveCorpus(c);
    }
};

void CorpusDescription::SegmentPartitionVisitorAdaptor::loadSegmentList(const std::string& filename, const std::string& encoding) {
    if (!filename.empty()) {
        Core::CompressedInputStream* cis = new Core::CompressedInputStream(filename.c_str());
        Core::TextInputStream        is(cis);
        is.setEncoding(encoding);
        if (!is)
            Core::Application::us()->criticalError("Failed to open segment list file \"%s\".", filename.c_str());
        std::string s;
        while (Core::getline(is, s) != EOF) {
            if ((s.size() == 0) || (s.at(0) == '#'))
                continue;
            Core::stripWhitespace(s);
            segmentsToKeep_.insert(s);
        }
    }
}

// ---------------------------------------------------------------------------
ProgressReportingVisitorAdaptor::ProgressReportingVisitorAdaptor(Core::XmlChannel& ch, bool reportOrth)
        : visitor_(0), channel_(ch), reportSegmentOrth_(reportOrth) {}

void ProgressReportingVisitorAdaptor::enterCorpus(Corpus* c) {
    channel_ << Core::XmlOpen((c->level()) ? "subcorpus" : "corpus") + Core::XmlAttribute("name", c->name()) + Core::XmlAttribute("full-name", c->fullName());
    visitor_->enterCorpus(c);
}

void ProgressReportingVisitorAdaptor::leaveCorpus(Corpus* c) {
    visitor_->leaveCorpus(c);
    channel_ << Core::XmlClose((c->level()) ? "subcorpus" : "corpus");
}

void ProgressReportingVisitorAdaptor::enterRecording(Recording* r) {
    Core::XmlOpen open("recording");
    open + Core::XmlAttribute("name", r->name());
    open + Core::XmlAttribute("full-name", r->fullName());
    if (!r->audio().empty())
        open + Core::XmlAttribute("audio", r->audio());
    if (!r->video().empty())
        open + Core::XmlAttribute("video", r->video());
    channel_ << open;
    visitor_->enterRecording(r);
}

void ProgressReportingVisitorAdaptor::leaveRecording(Bliss::Recording* r) {
    visitor_->leaveRecording(r);
    channel_ << Core::XmlClose("recording");
}

void ProgressReportingVisitorAdaptor::openSegment(Segment* s) {
    channel_ << Core::XmlOpen("segment") + Core::XmlAttribute("name", s->name()) + Core::XmlAttribute("full-name", s->fullName()) + Core::XmlAttribute("track", s->track()) + Core::XmlAttribute("start", s->start()) + Core::XmlAttribute("end", s->end());
    if (s->condition()) {
        channel_ << Core::XmlEmpty("condition") + Core::XmlAttribute("name", s->condition()->name());
    }
}

void ProgressReportingVisitorAdaptor::closeSegment(Segment* s) {
    channel_ << Core::XmlClose("segment");
}

void ProgressReportingVisitorAdaptor::visitSegment(Segment* s) {
    openSegment(s);
    visitor_->visitSegment(s);
    closeSegment(s);
}
void ProgressReportingVisitorAdaptor::visitSpeechSegment(SpeechSegment* s) {
    openSegment(s);
    if (s->speaker()) {
        channel_ << Core::XmlEmpty("speaker") + Core::XmlAttribute("name", s->speaker()->name()) + Core::XmlAttribute("gender", Bliss::Speaker::genderId[s->speaker()->gender()]);
    }
    if ((s->orth() != "") && (reportSegmentOrth_)) {
        channel_ << Core::XmlOpen("orth")
                 << s->orth()
                 << Core::XmlClose("orth");
    }
    visitor_->visitSpeechSegment(s);
    closeSegment(s);
}

// ---------------------------------------------------------------------------
const Core::Choice CorpusDescription::progressIndicationChoice(
        "none", noProgress,
        "local", localProgress,
        "global", globalProgress,
        Core::Choice::endMark());

const Core::ParameterChoice CorpusDescription::paramProgressIndication(
        "progress-indication",
        &progressIndicationChoice,
        "how to display progress in processing the corpus",
        noProgress);

class CorpusDescription::SegmentCountingVisitor : public CorpusVisitor {
private:
    u32 nSegments_;

public:
    void reset() {
        nSegments_ = 0;
    }
    u32 nSegments() const {
        return nSegments_;
    }
    virtual void visitSegment(Segment*) {
        ++nSegments_;
    }
};

class CorpusDescription::ProgressIndicationVisitorAdaptor : public CorpusVisitor {
private:
    u32                     nSegments_;
    CorpusVisitor*          visitor_;
    Core::ProgressIndicator pi_;

public:
    ProgressIndicationVisitorAdaptor()
            : nSegments_(0), visitor_(0), pi_("traversing corpus", "segments") {}
    void setVisitor(CorpusVisitor* v) {
        visitor_ = v;
    }
    void setTotal(u32 n) {
        nSegments_ = n;
    }
    virtual void visitSegment(Segment* s) {
        visitor_->visitSegment(s);
        pi_.notify();
    }
    virtual void visitSpeechSegment(SpeechSegment* s) {
        visitor_->visitSpeechSegment(s);
        pi_.notify();
    }
    virtual void enterRecording(Recording* r) {
        visitor_->enterRecording(r);
    }
    virtual void leaveRecording(Recording* r) {
        visitor_->leaveRecording(r);
    }
    virtual void enterCorpus(Corpus* c) {
        pi_.setTask(c->fullName());
        if (!c->level())
            pi_.start(nSegments_);
        visitor_->enterCorpus(c);
    }
    virtual void leaveCorpus(Corpus* c) {
        if (!c->level())
            pi_.finish();
        visitor_->leaveCorpus(c);
    }
};

// ---------------------------------------------------------------------------
CorpusDescription::CorpusDescription(const Core::Configuration& c)
        : Component(c),
          selector_(0),
          progressChannel_(c, "progress"),
          reporter_(0),
          indicator_(0),
          ordering_(0) {
    filename_ = paramFilename(config);

    s32                            partitioning      = paramPartition(config);
    u32                            skipFirstSegments = paramSkipFirstSegments(config);
    const std::vector<std::string> segmentFullNames  = paramSegmentsToSkip(config);
    std::string                    segmentsFilename  = paramFilename(select("segments"));
    if (partitioning || skipFirstSegments || !segmentFullNames.empty() || !segmentsFilename.empty()) {
        selector_ = new SegmentPartitionVisitorAdaptor;
    }

    if (partitioning) {
        s32  selectedPartition       = paramPartitionSelection(config);
        bool recordingBasedPartition = paramRecordingBasedPartition(config);
        if (selectedPartition == partitioning)
            selectedPartition = 0;  // This convention is useful for SGE array jobs
        else if (selectedPartition > partitioning)
            error("Invalid partition %d (should be 0 - %d).", selectedPartition, partitioning);
        selector_->setPartitioning(partitioning, selectedPartition, recordingBasedPartition);
    }
    if (skipFirstSegments) {
        selector_->setSkippedSegments(skipFirstSegments);
    }
    if (!segmentFullNames.empty()) {
        Core::StringHashSet segmentsToSkip;
        for (u32 i = 0; i < segmentFullNames.size(); ++i) {
            segmentsToSkip.insert(segmentFullNames[i]);
        }
        selector_->setSegmentsToSkip(segmentsToSkip);
    }
    if (!segmentsFilename.empty()) {
        selector_->loadSegmentList(segmentsFilename, paramEncoding(select("segments")));
        if (selector_->segmentsToKeep().empty() && !paramAllowEmptyWhitelist(select("segments")))
            error("Discard segment whitelist, because file is empty or does not exist: ") << segmentsFilename;
        else
            log("Use a segment whitelist with %d entries, keep only listed segments.", u32(selector_->segmentsToKeep().size()));
    }

    // Handle the ordering.
    {
        if (paramTheanoSegmentOrder(config)) {
#ifdef MODULE_THEANO_INTERFACE
            verify(!ordering_);
            ordering_ = new TheanoSegmentOrderingVisitor();
            log("Using Theano segment ordering");
#else
            criticalError("theano-segment-order not possible, MODULE_THEANO_INTERFACE disabled.");
#endif
        }
        if (paramPythonSegmentOrder(config)) {
#ifdef MODULE_PYTHON
            if (ordering_)
                criticalError("python-segment-order not possible, another ordering (theano?) already used");
            std::string pyModPath = paramPythonSegmentOrderModPath(config);
            std::string pyModName = paramPythonSegmentOrderModName(config);
            std::string pyConfig  = paramPythonSegmentOrderConfig(config);
            if (pyModName.empty()) {
                criticalError("python-segment-order: need Python module name (%s)",
                              paramPythonSegmentOrderModPath.name().c_str());
                return;
            }
            ordering_ = new PythonSegmentOrderingVisitor(pyModPath, pyModName, pyConfig, *this);
#else
            criticalError("python-segment-order not possible, MODULE_PYTHON disabled.");
#endif
        }
        std::string segmentOrder = paramSegmentOrder(config);
        if (!segmentOrder.empty()) {
            if (!ordering_)
                ordering_ = new SegmentOrderingVisitor();
            log("Using segment order list '%s'", segmentOrder.c_str());
            ordering_->setSegmentList(segmentOrder);
        }
        bool segmentOrderShuffle = paramSegmentOrderShuffle(config);
        if (segmentOrderShuffle) {
            if (!ordering_)
                ordering_ = new SegmentOrderingVisitor();
            ordering_->setAutoShuffle(true);
            s32 seed = paramSegmentOrderShuffleSeed(config);
            if (seed == -1)
                // Fallback to other seed config setting which is used e.g. by NnTrainer and others.
                seed = Core::ParameterInt("seed", "seed", -1)(config);
            if (seed == -1)
                seed = 0;
            ordering_->shuffleRandomSeed((u32)seed);
            log("Using segment order shuffling with seed %i", seed);
        }
        if (paramSegmentOrderSortByTimeLength(config)) {
            if (!ordering_)
                ordering_ = new SegmentOrderingVisitor();
            s32 chunkSize = paramSegmentOrderSortByTimeLengthChunkSize(config);
            ordering_->setSortByTimeLength(true, chunkSize);
            log("Using segment order sort-by-time-length with chunk-size %i", chunkSize);
        }

        if (ordering_) {
            ordering_->setShortNameLookup(paramSegmentOrderLookupName(config));
        }
    }

    progressIndicationMode_ = ProgressIndcationMode(paramProgressIndication(config));
    u32 nSegments           = 0;
    switch (progressIndicationMode_) {
        case globalProgress: {
            nSegments = totalSegmentCount();
        }  // fall through
        case localProgress:
            indicator_ = new ProgressIndicationVisitorAdaptor();
            indicator_->setTotal(nSegments);
            break;
        case noProgress: break;
        default: defect();
    }

    if (progressChannel_.isOpen()) {
        reporter_ = new ProgressReportingVisitorAdaptor(progressChannel_, paramProgressReportingSegmentOrth(config));
    }
}

CorpusDescription::~CorpusDescription() {
    delete ordering_;
    delete selector_;
    delete reporter_;
    delete indicator_;
}

void CorpusDescription::accept(CorpusVisitor* visitor) {
    CorpusDescriptionParser parser(config);
    if (indicator_) {
        indicator_->setVisitor(visitor);
        visitor = indicator_;
    }
    if (reporter_) {
        reporter_->setVisitor(visitor);
        visitor = reporter_;
    }
    if (selector_) {
        selector_->setVisitor(visitor);
        visitor = selector_;
    }
    if (ordering_) {
        ordering_->setVisitor(visitor);
        visitor = ordering_;
    }
    parser.accept(file(), visitor);
}

u32 CorpusDescription::totalSegmentCount() {
    SegmentCountingVisitor* counter = new SegmentCountingVisitor();
    counter->reset();
    // Note: An accept() call is problematic because we don't
    // want certain side-effects like the reporter or the indicator.
    // Also, it would change the internal state of ordering,
    // thus we need to have our own copy here to not change the state.
    SegmentOrderingVisitor* ordering = ordering_ ? ordering_->copy() : 0;
    {
        CorpusDescriptionParser parser(config);
        CorpusVisitor*          visitor = counter;
        if (selector_) {
            selector_->setVisitor(visitor);
            visitor = selector_;
        }
        if (ordering) {
            ordering->setVisitor(visitor);
            visitor = ordering;
        }
        parser.accept(file(), visitor);
    }
    delete ordering;
    u32 nSegments = counter->nSegments();
    delete counter;
    return nSegments;
}
