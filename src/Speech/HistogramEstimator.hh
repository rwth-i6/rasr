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
#ifndef _SPEECH_HISTOGRAM_ESTIMATOR_HH
#define _SPEECH_HISTOGRAM_ESTIMATOR_HH

#include <Core/ObjectCache.hh>
#include <Signal/Histogram.hh>
#include "DataExtractor.hh"

namespace Speech {

    /** HistogramEstimator */
    class HistogramEstimator : public FeatureVectorExtractor {
        typedef FeatureVectorExtractor Precursor;
    public:
        typedef f32 Value;
    public:
        static const Core::ParameterFloat paramBucketSize;
    private:
        Core::Ref<Bliss::CorpusKey> corpusKey_;
        typedef Signal::HistogramVector<Value> HistogramVector;
        HistogramVector *currentHistogramVector_;

        size_t featureDimension_;
        Value bucketSize_;
        typedef Core::ObjectCache<Core::MruObjectCacheList<
            std::string, HistogramVector, Core::StringHash, Core::StringEquality> >
        HistogramVectorCache;
        HistogramVectorCache histogramVectorCache_;
    private:
        Signal::HistogramVector<Value>* histogramVector(size_t featureDimension);
    protected:
        virtual void setFeatureVectorDescription(const Mm::FeatureDescription::Stream &);
        virtual void processFeatureVector(Core::Ref<const Feature::Vector> f) {
            verify_(currentHistogramVector_ != 0); currentHistogramVector_->accumulate(*f);
        }
    public:
        HistogramEstimator(const Core::Configuration&);
        virtual void signOn(CorpusVisitor&);

        void clear() { histogramVectorCache_.clear(); currentHistogramVector_ = 0; }
    };
} // namespace Speech

#endif // _SPEECH_HISTOGRAM_ESTIMATOR_HH
