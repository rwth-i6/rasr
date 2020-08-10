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
#include "HistogramEstimator.hh"

using namespace Speech;

const Core::ParameterFloat HistogramEstimator::paramBucketSize(
        "bucket-size", "bucket size in the histogram", 0.0002);

HistogramEstimator::HistogramEstimator(const Core::Configuration& c)
        : Component(c),
          Precursor(c),
          corpusKey_(new Bliss::CorpusKey(select("corpus-key"))),
          currentHistogramVector_(0),
          featureDimension_(0),
          bucketSize_(paramBucketSize(c)),
          histogramVectorCache_(select("histograms-cache")) {}

void HistogramEstimator::signOn(CorpusVisitor& corpusVisitor) {
    corpusVisitor.signOn(corpusKey_);
    Precursor::signOn(corpusVisitor);
}

void HistogramEstimator::setFeatureVectorDescription(const Mm::FeatureDescription::Stream& description) {
    size_t dimension;
    description.getValue(Mm::FeatureDescription::nameDimension, dimension);
    std::string key;
    corpusKey_->resolve(key);

    currentHistogramVector_ = histogramVectorCache_.findForWriteAccess(key);

    if (currentHistogramVector_ == 0) {
        if (bucketSize_ == 0)
            criticalError("Bucket size is 0.");
        if (dimension == 0)
            warning("Input vector size is 0");

        currentHistogramVector_ = new HistogramVector(dimension, bucketSize_);
        if (!histogramVectorCache_.insert(key, currentHistogramVector_))
            defect();
    }
    verify(currentHistogramVector_->size() == dimension);
}
