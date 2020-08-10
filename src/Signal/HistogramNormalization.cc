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
#include "HistogramNormalization.hh"
#include <Core/Directory.hh>
#include <Core/Utility.hh>
#include <Core/XmlStream.hh>
#include <Flow/Vector.hh>
using namespace Signal;

//===============================================================================================

void HistogramNormalization::setTrainingHistograms(const std::vector<Histogram<Value>>& trainingHistograms,
                                                   Probability                          probabilityBucketSize) {
    inverseTrainingCdfs_.resize(trainingHistograms.size());

    Cdf trainingCdf;
    for (size_t i = 0; i < trainingHistograms.size(); ++i) {
        trainingHistograms[i].getCdf(trainingCdf);
        inverseTrainingCdfs_[i] = Cdf(probabilityBucketSize);
        trainingCdf.getInverse(inverseTrainingCdfs_[i]);
    }
}

void HistogramNormalization::setTrainingHistograms(const std::vector<HistogramVector<Value>>& trainingHistograms,
                                                   const std::vector<HistogramWeight>&        scales,
                                                   Probability                                probabilityBucketSize) {
    size_t nScales = trainingHistograms.size();
    require(nScales == scales.size());
    require(nScales > 0);

    size_t dimension = trainingHistograms[0].size();

    Value minimalBucketSize = Core::Type<Value>::max;
    for (size_t i = 0; i < nScales; ++i)
        minimalBucketSize = std::min(minimalBucketSize, trainingHistograms[i].minimalBucketSize());

    HistogramVector<Value> interpolatedHistograms(dimension, minimalBucketSize);
    for (size_t i = 0; i < nScales; ++i) {
        require(trainingHistograms[i].size() == dimension);
        for (size_t d = 0; d < dimension; ++d) {
            Histogram<Value> toAdd(trainingHistograms[i][d]);
            toAdd.normalizeSurface();
            toAdd *= scales[i];
            interpolatedHistograms[d] += toAdd;
        }
    }
    setTrainingHistograms(interpolatedHistograms, probabilityBucketSize);
}

void HistogramNormalization::setTestHistograms(const std::vector<Histogram<Value>>& testHistograms) {
    testCdfs_.resize(testHistograms.size());
    for (size_t i = 0; i < testHistograms.size(); ++i)
        testHistograms[i].getCdf(testCdfs_[i]);
}

void HistogramNormalization::apply(const std::vector<Value>& in, std::vector<Value>& out) {
    verify(inverseTrainingCdfs_.size() == in.size());
    verify(testCdfs_.size() == in.size());

    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = inverseTrainingCdfs_[i][testCdfs_[i][in[i]]];
}

bool HistogramNormalization::areScalesWellDefined(const std::vector<HistogramWeight>& scales) {
    // 0 <= scale <= 1
    if (std::find_if(scales.begin(), scales.end(), std::bind2nd(std::less<HistogramWeight>(), 0)) != scales.end())
        return false;
    if (std::find_if(scales.begin(), scales.end(), std::bind2nd(std::greater<HistogramWeight>(), 1)) != scales.end())
        return false;
    return true;
}

bool HistogramNormalization::areScalesNormalized(const std::vector<HistogramWeight>& scales) {
    HistogramWeight sum = std::accumulate(scales.begin(), scales.end(), 0.0, std::plus<HistogramWeight>());
    return Core::isAlmostEqual(sum, (HistogramWeight)1.0);
}

void HistogramNormalization::normalizeScales(std::vector<HistogramWeight>& scales) {
    scales.insert(scales.begin(), (HistogramWeight)1.0 - std::accumulate(scales.begin(), scales.end(), 0.0));
}

//===============================================================================================
const Core::ParameterFloat HistogramNormalizationNode::paramProbabilityBucketSize(
        "probability-bucket-size", "probability bucket size (if 0, heuristical value will be used.)", 0, 0);

const Core::ParameterStringVector HistogramNormalizationNode::paramTrainingHistogramsFilenames(
        "training-histograms", "file name(s) of training histograms");

const Core::ParameterString HistogramNormalizationNode::paramCorpusKey(
        "corpus-key", "template expression for key of test histograms");

const std::string HistogramNormalizationNode::scalePortname_("histogram-scale-");

//===============================================================================================
HistogramNormalizationNode::HistogramNormalizationNode(const Core::Configuration& c)
        : Component(c),
          Flow::Node(c),
          firstScalePortId_(Flow::IllegalPortId),
          probabilityBucketSize_(0),
          testHistograms_(select("histograms-cache"), Core::reuseObjectCacheMode),
          needInit_(true) {
    addInput(0);
    firstScalePortId_ = 1;
    addOutput(0);
    setProbabilityBucketSize(paramProbabilityBucketSize(c));
    setTrainingHistogramFilenames(paramTrainingHistogramsFilenames(c));
    setTestHistograms(paramCorpusKey(c));
}

void HistogramNormalizationNode::setTestHistograms(const std::string& corpusKey) {
    const std::vector<Histogram<Value>>* histograms = testHistograms_.findForReadAccess(corpusKey);
    if (histograms != 0)
        HistogramNormalization::setTestHistograms(*histograms);
    else if (!corpusKey.empty())
        criticalError("No test-histogram found for the corpus-key \"%s\".", corpusKey.c_str());
}

bool HistogramNormalizationNode::setParameter(const std::string& name, const std::string& value) {
    if (paramProbabilityBucketSize.match(name))
        setProbabilityBucketSize(paramProbabilityBucketSize(value));
    else if (paramTrainingHistogramsFilenames.match(name))
        setTrainingHistogramFilenames(paramTrainingHistogramsFilenames(value));
    else if (paramCorpusKey.match(name))
        setTestHistograms(paramCorpusKey(value));
    else
        return testHistograms_.setParameter(name, value);
    return true;
}

bool HistogramNormalizationNode::configure() {
    reset();

    Core::Ref<Flow::Attributes> featureAttributes(new Flow::Attributes);
    getInputAttributes(0, *featureAttributes);
    if (!configureDatatype(featureAttributes, Flow::Vector<f32>::type()))
        return false;

    Core::Ref<Flow::Attributes> scaleAttributes(new Flow::Attributes);
    for (Flow::PortId i = firstScalePortId_; i < nInputs(); ++i) {
        Core::Ref<const Flow::Attributes> scaleAttributes = getInputAttributes(i);
        if (!configureDatatype(scaleAttributes, Flow::DataAdaptor<HistogramWeight>::type()))
            return false;
        featureAttributes->merge(*scaleAttributes);
    }
    featureAttributes->set("datatype", Flow::Vector<f32>::type()->name());
    return putOutputAttributes(0, featureAttributes);
}

Flow::PortId HistogramNormalizationNode::getInput(const std::string& name) {
    if (name == "")
        return 0;
    u32 id;
    if (sscanf(name.c_str(), std::string(scalePortname_ + "%u").c_str(), &id) != 1)
        criticalError() << "Scale port names must have format '" << scalePortname_ << "<order=1,2,...>'";
    if (id == 0) {
        criticalError() << "The scale '" << scalePortname_ << "0' is free parameter. "
                        << "It will be derived from the rest of the scales.";
    }
    needInit_ = true;
    return addInput(firstScalePortId_ + id - 1);
}

void HistogramNormalizationNode::init(size_t featureDimension) {
    if (nTestHistograms() != featureDimension) {
        error() << "Mismatch between #test-histograms(" << nTestHistograms()
                << ") and feature dimension(" << featureDimension << ").";
    }

    verify(nInputs() >= firstScalePortId_);
    histogramScales_.resize(nInputs() - firstScalePortId_);
    if ((histogramScales_.size() + 1) != trainingHistogramFilenames_.size()) {
        error() << "Mismatch between #training-histograms(" << trainingHistogramFilenames_.size()
                << ") and #scale-ports(" << histogramScales_.size() << ").";
    }

    loadTrainingHistograms(featureDimension);

    respondToDelayedErrors();

    if (trainingHistograms_.size() == 1)
        HistogramNormalization::setTrainingHistograms(trainingHistograms_[0], probabilityBucketSize_);

    reset();
    needInit_ = false;
}

void HistogramNormalizationNode::loadTrainingHistograms(size_t featureDimension) {
    trainingHistograms_.resize(trainingHistogramFilenames_.size());
    for (size_t i = 0; i < trainingHistograms_.size(); ++i) {
        const std::string& filename(trainingHistogramFilenames_[i]);
        log() << "Reading 'training histogram " << i << " from file '" << filename << "' ...";
        Core::BinaryInputStream is(filename);
        bool                    success = is.isOpen();
        if (success) {
            trainingHistograms_[i].read(is);
            success = is.good();
            if (success) {
                if (trainingHistograms_[i].size() != featureDimension) {
                    error() << "Mismatch between #training-histograms(" << trainingHistograms_[i].size()
                            << ") and feature dimension(" << featureDimension << ").";
                }
            }
        }
        if (!success)
            error() << "Failed to read training histogram from file '" << filename << "'.";
    }
}

bool HistogramNormalizationNode::work(Flow::PortId p) {
    Flow::DataPtr<Flow::Vector<f32>> in;
    if (!getData(0, in))
        return putData(0, in.get());

    if (needInit_)
        init(in->size());

    if (updateScales(*in) && !updateTrainingHistograms())
        return putEos(0);

    in.makePrivate();
    apply(*in, *in);
    return putData(0, in.get());
}

bool HistogramNormalizationNode::updateScales(const Flow::Timestamp& timestamp) {
    bool changed = false;
    for (size_t i = 0; i < histogramScales_.size(); ++i) {
        while (!histogramScales_[i].contains(timestamp)) {
            Flow::DataPtr<Flow::DataAdaptor<HistogramWeight>> in;
            Flow::PortId                                      portId = firstScalePortId_ + i;
            if (getData(portId, in)) {
                histogramScales_[i] = *in;
                changed             = true;
            }
            else {
                criticalError() << "The " << scalePortname_ << portId
                                << " stream stopped before start-time ("
                                << timestamp.startTime() << ").";
            }
        }
    }
    return changed;
}

bool HistogramNormalizationNode::updateTrainingHistograms() {
    bool                         result = true;
    std::vector<HistogramWeight> histogramScales(histogramScales_.size());
    for (size_t i = 0; i < histogramScales_.size(); ++i)
        histogramScales[i] = histogramScales_[i].data();

    normalizeScales(histogramScales);
    if (!(result = areScalesWellDefined(histogramScales)))
        error() << "One or more histogram scales are smaller than zero or larger than 1.";

    verify(areScalesNormalized(histogramScales));
    HistogramNormalization::setTrainingHistograms(trainingHistograms_, histogramScales, probabilityBucketSize_);
    return result;
}

void HistogramNormalizationNode::reset() {
    for (size_t i = 0; i < histogramScales_.size(); ++i) {
        histogramScales_[i].setStartTime(Core::Type<Flow::Time>::min);
        histogramScales_[i].setEndTime(histogramScales_[i].startTime());
    }
}
