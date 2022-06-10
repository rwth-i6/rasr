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
#include "AligningFeatureExtractor.hh"

using namespace Speech;

const Core::ParameterString AligningFeatureExtractor::paramAlignmentPortName(
    "alignment-port-name",
    "name of the main data source port",
    "alignments");

const Core::ParameterBool AligningFeatureExtractor::paramEnforceWeightedProcessing(
    "enforce-weighted-processing",
    "enforce weighted processing even for weights=1 etc.",
    false);

const Core::ParameterString AligningFeatureExtractor::paramAlignment2PortName(
    "alignment-2-port-name",
    "name of the second data source port",
    "alignments-2");

const Core::ParameterBool AligningFeatureExtractor::paramPeakyAlignment(
    "peaky-alignment",
    "peaky alignment: label segment of the same labels with one label and blank elsewhere",
    false);

const Core::ParameterFloat AligningFeatureExtractor::paramPeakPosition(
    "peak-position",
    "relative position of peaky alignment in the label segment",
    0.5);

const Core::ParameterBool AligningFeatureExtractor::paramForceSingleState(
    "force-single-state",
    "force the alignment allophone to be single state",
    false);

AligningFeatureExtractor::AligningFeatureExtractor(const Core::Configuration& c,
                                                   AlignedFeatureProcessor&   alignedFeatureProcessor)
        : Core::Component(c),
          Precursor(c),
          alignedFeatureProcessor_(alignedFeatureProcessor),
          alignmentPortId_(Flow::IllegalPortId),
          processWeighted_(paramEnforceWeightedProcessing(config)),
          alignment_(0),
          alignment2PortId_(Flow::IllegalPortId),
          alignment2_(0),
          peakyAlignment_(paramPeakyAlignment(c)),
          peakPos_(paramPeakPosition(c)),
          forceSingleState_(paramForceSingleState(c)) {
    alignedFeatureProcessor_.setDataSource(dataSource());

    const std::string alignmentPortName(paramAlignmentPortName(c));
    alignmentPortId_ = dataSource()->getOutput(alignmentPortName);
    if (alignmentPortId_ == Flow::IllegalPortId)
        criticalError("Flow network does not have an output named \"%s\"", alignmentPortName.c_str());

    const std::string alignment2PortName(paramAlignment2PortName(c));
    alignment2PortId_ = dataSource()->getOutput(alignment2PortName);
    // enforce peaky alignment (mainly for blank-based transducer topology)
    if ( peakyAlignment_ ) {
        verify( alignment2PortId_ == Flow::IllegalPortId ); // not supported
        blankIndex_ = alignedFeatureProcessor_.getSilenceAllophoneStateIndex();
        verify( blankIndex_ != Fsa::InvalidLabelId );
        log() << "apply peaky alignment with relative position " << peakPos_
              << " and blank allophoneStateIndex (silence) " << blankIndex_;
        alignedFeatureProcessor_.setPeakyAlignment();
    } else if ( forceSingleState_ )
        log() << "force the alignemnt to have single allophone state";
}

AligningFeatureExtractor::~AligningFeatureExtractor() {}

void AligningFeatureExtractor::signOn(CorpusVisitor& corpusVisitor) {
    alignedFeatureProcessor_.signOn(corpusVisitor);
    Precursor::signOn(corpusVisitor);
}

void AligningFeatureExtractor::enterCorpus(Bliss::Corpus* corpus) {
    Precursor::enterCorpus(corpus);
    alignedFeatureProcessor_.enterCorpus(corpus);
}

void AligningFeatureExtractor::leaveCorpus(Bliss::Corpus* corpus) {
    alignedFeatureProcessor_.leaveCorpus(corpus);
    Precursor::leaveCorpus(corpus);
}

void AligningFeatureExtractor::enterSegment(Bliss::Segment* segment) {
    Precursor::enterSegment(segment);
    alignedFeatureProcessor_.enterSegment(segment);
}

void AligningFeatureExtractor::leaveSegment(Bliss::Segment* segment) {
    alignedFeatureProcessor_.leaveSegment(segment);
    Precursor::leaveSegment(segment);
}

void AligningFeatureExtractor::enterSpeechSegment(Bliss::SpeechSegment* segment) {
    Precursor::enterSpeechSegment(segment);
    alignedFeatureProcessor_.enterSpeechSegment(segment);
}

void AligningFeatureExtractor::leaveSpeechSegment(Bliss::SpeechSegment* segment) {
    alignedFeatureProcessor_.leaveSpeechSegment(segment);
    Precursor::leaveSpeechSegment(segment);
}

void AligningFeatureExtractor::processSegment(Bliss::Segment* segment) {
    verify(alignmentPortId_ != Flow::IllegalPortId);
    if (initializeAlignment()) {
        Precursor::processSegment(segment);
    }
    else {
        log() << "alignment failed: " << segment->name();
    }
}

void AligningFeatureExtractor::setFeatureDescription(const Mm::FeatureDescription& description) {
    alignedFeatureProcessor_.setFeatureDescription(description);
}

void AligningFeatureExtractor::processFeature(Core::Ref<const Feature> f) {
    verify(alignment_ && !alignment_->empty());
    if ( currentAlignmentItem_ == alignment_->end() )
    {   // allow already sub-sampled alignment input: just process extra features
        if ( alignedFeatureProcessor_.needReducedAlignment() )
            alignedFeatureProcessor_.processExtraFeature(f, alignment_->size());
        else
            warning("Alignment (size=%zd) shorter than the feature stream (current=%zd)", alignment_->size(), (size_t)currentFeatureId_);
    } else { // possible gap in the alignment: ensure processing
        while( currentFeatureId_ < currentAlignmentItem_->time )
            ++currentFeatureId_;
        verify( currentFeatureId_ == currentAlignmentItem_->time );
        if ( alignment2_ )
            binaryProcessFeature(f);
        else
            unaryProcessFeature(f);
    }
    ++currentFeatureId_;
}

void AligningFeatureExtractor::unaryProcessFeature(Core::Ref<const Feature> f) {
    while (currentAlignmentItem_ != alignment_->end() && currentAlignmentItem_->time == currentFeatureId_) {
        verify(currentAlignmentItem_ == alignment_->begin() ? true : (currentAlignmentItem_ - 1)->time <= currentAlignmentItem_->time);
        if ( processWeighted_ )
            alignedFeatureProcessor_.processAlignedFeature(f, currentAlignmentItem_->emission, currentAlignmentItem_->weight);
        else
            alignedFeatureProcessor_.processAlignedFeature(f, currentAlignmentItem_->emission);
        ++currentAlignmentItem_;
    }
}

void AligningFeatureExtractor::binaryProcessFeature(Core::Ref<const Feature> f) {
    verify(alignment2_ && !alignment2_->empty());
    while (currentAlignmentItem_ != alignment_->end() && currentAlignmentItem_->time == currentFeatureId_) {
        verify(currentAlignmentItem_ == alignment_->begin() ? true : (currentAlignmentItem_ - 1)->time <= currentAlignmentItem_->time);
        verify(currentAlignmentItem_->time == currentAlignment2Item_->time);
        verify(currentAlignmentItem_->emission == currentAlignment2Item_->emission);
        if ( processWeighted_ ) {
            Mm::Weight w1 = currentAlignmentItem_->weight;
            Mm::Weight w2 = currentAlignment2Item_->weight;
            alignedFeatureProcessor_.processAlignedFeature(f, currentAlignmentItem_->emission, w1, w2);
        } else
            alignedFeatureProcessor_.processAlignedFeature(f, currentAlignmentItem_->emission);
        ++currentAlignmentItem_;
        ++currentAlignment2Item_;
    }
}

bool AligningFeatureExtractor::initializeAlignment() {
    alignment_  = 0;
    alignment2_ = 0;

    if (!dataSource()->getData(alignmentPortId_, alignmentRef_)) {
        error("Failed to extract alignment.");
        return false;
    }
    if ( peakyAlignment_ )
        makePeakyAlignment(&alignmentRef_->data());
    else if ( forceSingleState_ )
        makeSingleStateAlignment(&alignmentRef_->data());
    alignment_            = &alignmentRef_->data();
    currentFeatureId_     = 0;
    currentAlignmentItem_ = alignment_->begin();
    if (alignment_->empty()) {
        warning("Segment has been discarded because of empty alignment.");
        return false;
    }

    // fixed behavior: per alignment decision
    processWeighted_ = paramEnforceWeightedProcessing(config) || alignment_->hasWeights() || currentAlignmentItem_->time != 0;

    if (alignment2PortId_ != Flow::IllegalPortId) {
        if (!dataSource()->getData(alignment2PortId_, alignment2Ref_)) {
            error("Failed to extract alignment.");
            return false;
        }
        alignment2_            = &alignment2Ref_->data();
        currentAlignment2Item_ = alignment2_->begin();
        if (alignment2_->empty()) {
            warning("Segment has been discarded because of empty alignment.");
            return false;
        }
        if (alignment_->size() != alignment2_->size()) {
            error("Mismatch in size of alignments.");
            return false;
        }
        if ( !processWeighted_ )
            processWeighted_ = alignment2_->hasWeights();
    }

    return true;
}

// always single state only
void AligningFeatureExtractor::makePeakyAlignment(Alignment* align)
{
    if ( align->empty() )
        return;
    std::vector<std::pair<u32, Fsa::LabelId> > peaks;
    u32 start = 0, end = 0; s16 state = 0;
    Fsa::LabelId currentAlloIdx = align->begin()->emission & Am::AllophoneAlphabet::IdMask;
    Fsa::LabelId currentEmission = currentAlloIdx | state << 26;
    for (u32 idx = 0; idx < align->size(); ++idx) 
    {
        Fsa::LabelId alloIdx = align->at(idx).emission & Am::AllophoneAlphabet::IdMask;
        if ( alloIdx != currentAlloIdx )
        {
            u32 pos = peakPos_ * (end - start) + start;
            verify( pos >= start && pos <= end );
            peaks.push_back( std::make_pair(pos, currentEmission) );
            start = idx; end = idx;
            currentAlloIdx = alloIdx;
            currentEmission = currentAlloIdx | state << 26;
        }
        align->at(idx).emission = blankIndex_;
        end = idx; // inclusive
    }
    // also the last segment
    u32 pos = peakPos_ * (end - start) + start;
    verify( pos >= start && pos <= end );
    peaks.push_back( std::make_pair(pos, currentEmission) );
    // assign peaks
    verify( !peaks.empty() );
    for (std::vector<std::pair<u32, Fsa::LabelId> >::const_iterator iter=peaks.begin(); iter!=peaks.end(); ++iter)
        align->at(iter->first).emission = iter->second;
}

// silence is anyway single state
void AligningFeatureExtractor::makeSingleStateAlignment(Alignment* align)
{
    if ( align->empty() )
        return;
    s16 state = 0;
    for (u32 idx = 0; idx < align->size(); ++idx)
    {
        Fsa::LabelId alloIdx = align->at(idx).emission & Am::AllophoneAlphabet::IdMask;
        align->at(idx).emission = alloIdx | state << 26;
    }
}

