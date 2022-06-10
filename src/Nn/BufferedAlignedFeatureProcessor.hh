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
#ifndef _NN_BUFFERED_ALIGNED_FEATURE_PROCESSOR_HH
#define _NN_BUFFERED_ALIGNED_FEATURE_PROCESSOR_HH

#include <Core/Component.hh>

#include <Speech/AlignedFeatureProcessor.hh>  // supervised training
#include <Speech/CorpusVisitor.hh>
#include "BufferedFeatureExtractor.hh"  // buffered features

#include <Math/CudaMatrix.hh>
#include <Math/CudaVector.hh>
#include <Math/Vector.hh>

#include <Am/AcousticModel.hh>
#include <Fsa/Types.hh>
#include "ClassLabelWrapper.hh"

#include <vector>

namespace Nn {

/**
 *
 *	The class combines the "BufferedFeatureExtractor" and the "AlignedFeatureProcessor"
 *	and provides the buffered processing of the features and alignment .
 *	This should be used by Speech::AligningFeatureExtractor.
 */
template<class T>
class BufferedAlignedFeatureProcessor : protected BufferedFeatureExtractor<T>, public Speech::AlignedFeatureProcessor {
    typedef Core::Component                 Precursor;
    typedef BufferedFeatureExtractor<T>     PrecursorBuffer;
    typedef Speech::AlignedFeatureProcessor PrecursorAligned;

protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

public:
    static const Core::ParameterFloat  paramSilenceWeight;
    static const Core::ParameterString paramClassWeightsFile;
    static const Core::ParameterBool   paramWeightedAlignment;
    static const Core::ParameterInt    paramReduceAlignmentFactor;

protected:
    Core::Ref<const Am::AcousticModel> acousticModel_;
    Fsa::LabelId                       silence_;
    bool                               acousticModelNeedInit_;

    ClassLabelWrapper*      classLabelWrapper_;
    Math::Vector<T>         classWeights_;            // weights for each class
    std::vector<u32>        alignmentBuffer_;         // buffer for alignment indices
    std::vector<Mm::Weight> alignmentWeightsBuffer_;  // buffer for weights from the alignment
    bool                    weightedAlignment_;

    // alignment subsampling
    u32  reduceAlignFactor_; // reduction factor for downsampling alignment
    bool alignmentReduced_; // input alignment is already reduced
    u32  reducedSize_; // size of already reduced input alignment (for verification)

public:
    BufferedAlignedFeatureProcessor(const Core::Configuration& config, bool loadFromFile = true);
    virtual ~BufferedAlignedFeatureProcessor();

protected:
    virtual void initAcousticModel();
    virtual void setClassWeights();
    virtual void initTrainer(const std::vector<NnMatrix>& miniBatch);
    virtual void initBuffer(Core::Ref<const Speech::Feature> f);
    virtual void resetBuffer();
    virtual void processBuffer();
    virtual void generateMiniBatch(std::vector<NnMatrix>& miniBatch, Math::CudaVector<u32>& miniBatchAlignment, std::vector<f64>& miniBatchAlignmentWeights, u32 batchSize);
    virtual void processAlignedFeature(Core::Ref<const Speech::Feature> f, Am::AllophoneStateIndex e);
    virtual void processAlignedFeature(Core::Ref<const Speech::Feature> f, Am::AllophoneStateIndex e, Mm::Weight w);

    virtual bool needReducedAlignment() { return reduceAlignFactor_ > 1; }
    virtual void processExtraFeature(Core::Ref<const Speech::Feature> f, u32 size);

public:
    virtual void signOn(Speech::CorpusVisitor& corpusVisitor) {
        Speech::AlignedFeatureProcessor::signOn(corpusVisitor);
    }
    virtual void enterSegment(Bliss::Segment* segment);
    virtual void leaveSegment(Bliss::Segment* segment);
    virtual void leaveCorpus(Bliss::Corpus* corpus);

protected:
    Mm::EmissionIndex classIndex(Am::AllophoneStateIndex e) const;

public:
    virtual NeuralNetworkTrainer<T>* createTrainer(const Core::Configuration& config);
    Fsa::LabelId getSilenceAllophoneStateIndex();

private:
    void reduceAlignment(u32, u32, std::vector<u32>&);
    void reducePeakyAlignment(u32, u32, std::vector<u32>&);
};

}  // namespace Nn

#endif  // _NN_BUFFERED_ALIGNED_FEATURE_PROCESSOR_HH
