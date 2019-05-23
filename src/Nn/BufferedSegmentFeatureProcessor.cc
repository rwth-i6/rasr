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
#include "BufferedSegmentFeatureProcessor.hh"

namespace Nn {

template<typename FloatT>
BufferedSegmentFeatureProcessor<FloatT>::BufferedSegmentFeatureProcessor(const Core::Configuration& config)
        : Core::Component(config),
          BufferedFeatureExtractor<FloatT>(config) {
    if (Precursor::bufferType_ != Precursor::BufferType::utterance)
        this->error("underlying BufferedFeatureExtractor must be of type 'utterance'"
                    " (buffer-type = utterance)");
    if (Precursor::shuffle_)
        this->error("underlying BufferedFeatureExtractor not be shuffled (shuffle = false)");
}

template<typename FloatT>
BufferedSegmentFeatureProcessor<FloatT>::~BufferedSegmentFeatureProcessor() {
}

// We don't use the BufferedFeatureExtractor::processBuffer because it just forwards
// the segment features ^= mini batch to the NN trainer.
// The NN must also know about the segment transcription, though.
template<typename FloatT>
void BufferedSegmentFeatureProcessor<FloatT>::processBuffer() {
    typedef typename Math::FastMatrix<FloatT> FastMatrix;
    typedef typename Precursor::NnMatrix      NnMatrix;

    // We expect that the underlying BufferedFeatureProcessor
    // is in utterance buffer mode and this is only called indirectly
    // from leaveSpeechSegment().
    verify(Precursor::nBufferedFeatures_ > 0);
    auto* currentSegment = dynamic_cast<Bliss::SpeechSegment*>(Precursor::getCurSegment());
    require(currentSegment);

    Precursor::prepareProcessBuffer();

    // This is the number of time frames of the current segment.
    verify_eq(Precursor::batchSize_, Precursor::nBufferedFeatures_);

    std::vector<NnMatrix> miniBatch;
    Precursor::generateMiniBatch(miniBatch, Precursor::batchSize_);
    verify(!miniBatch.empty());
    verify_eq(Precursor::batchSize_, miniBatch.at(0).nColumns());

    // Every buffer is exactly one mini-batch.
    verify_eq(Precursor::nProcessedMiniBatches_, 0);

    Precursor::log("Process segment ")
            << "with " << miniBatch.at(0).nColumns() << " features.";

    if (!Precursor::trainer_->isInitialized()) {
        std::vector<u32> streamSizes;
        for (u32 stream = 0; stream < miniBatch.size(); stream++)
            streamSizes.push_back(miniBatch.at(stream).nRows());
        Precursor::trainer_->initializeTrainer(Precursor::batchSize_, streamSizes);
    }
    else
        Precursor::trainer_->setBatchSize(Precursor::batchSize_);

    // Process mini batch.
    Precursor::trainer_->processBatch_feedInput(miniBatch, NULL, currentSegment);
    Precursor::trainer_->processBatch_finishWithSpeechSegment(*currentSegment);

    Precursor::nProcessedMiniBatches_++;
    Precursor::nProcessedFeatures_ += Precursor::batchSize_;

    Precursor::finalizeProcessBuffer();
}

template<typename FloatT>
NeuralNetworkTrainer<FloatT>* BufferedSegmentFeatureProcessor<FloatT>::createTrainer(const Core::Configuration& config) {
    // We need a generic trainer which supports the processBatch_finishWithSpeechSegment() function.
    return NeuralNetworkTrainer<FloatT>::createSupervisedTrainer(config);
}

//=============================================================================
// explicit template instantiation
template class BufferedSegmentFeatureProcessor<f32>;
template class BufferedSegmentFeatureProcessor<f64>;

}  // namespace Nn
