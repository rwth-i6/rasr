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
#include "MeanEstimator.hh"
#include <Flow/Vector.hh>
#include <Math/Module.hh>

using namespace Signal;

const Core::ParameterString MeanEstimator::paramFilename(
        "file", "Output filename for mean vector");
const Core::ParameterInt MeanEstimator::paramOutputPrecision(
        "output-precision", "Number of decimal digits in text output formats", 20);

MeanEstimator::MeanEstimator(const Core::Configuration& configuration)
        : Precursor(configuration),
          featureDimension_(0),
          needInit_(true) {}

MeanEstimator::~MeanEstimator() {}

void MeanEstimator::initialize() {
    vectorSum_.resize(featureDimension_);
    std::fill(vectorSum_.begin(), vectorSum_.end(), Sum(0));

    count_    = 0;
    needInit_ = false;
}

void MeanEstimator::accumulate(const Math::Vector<Data>& x) {
    require_(x.size() == featureDimension_);

    if (needInit_)
        initialize();

    vectorSum_ += x;
    count_++;
}

bool MeanEstimator::finalize(Math::Vector<Data>& mean) const {
    if (needInit_) {
        error("No observation has been seen.");
        return false;
    }
    mean.resize(featureDimension_);
    std::transform(vectorSum_.begin(), vectorSum_.end(), mean.begin(), std::bind2nd(std::divides<Sum>(), count_));
    return true;
}

void MeanEstimator::setDimension(size_t dimension) {
    if (featureDimension_ != dimension) {
        featureDimension_ = dimension;
        needInit_         = true;
    }
}

bool MeanEstimator::write() const {
    bool               success = true;
    Math::Vector<Data> mean;
    if (!finalize(mean))
        success = false;

    std::string filename = paramFilename(config);
    if (Math::Module::instance().formats().write(
                filename, mean, paramOutputPrecision(config)))
        log("Mean vector written to '%s'.", filename.c_str());
    else {
        error("Failed to write mean to '%s'.", filename.c_str());
        success = false;
    }

    return success;
}

void MeanEstimator::reset() {
    initialize();
}

// =======================================

bool MeanEstimatorNode::configure() {
    if (!Precursor::configure())
        return false;
    Core::Ref<Flow::Attributes> a(new Flow::Attributes());
    getInputAttributes(0, *a);

    if (!configureDatatype(a, Flow::Vector<f32>::type())) {
        Node::error("wrong datatype. expected datatype was vector-f32");
        return false;
    }
    MeanEstimator::reset();
    return putOutputAttributes(0, a);
}

bool MeanEstimatorNode::work(Flow::PortId p) {
    Flow::DataPtr<Flow::Vector<f32>> in;
    u32                              count = 0;
    std::vector<Flow::Timestamp>     timestamps;
    while (Precursor::getData(0, in)) {
        if (count == 0)
            MeanEstimator::setDimension(in->size());
        MeanEstimator::accumulate(*in);
        timestamps.push_back(*in);
        ++count;
    }
    Math::Vector<f32> mean;
    if (!MeanEstimator::finalize(mean))
        return false;
    for (u32 i = 0; i < count; ++i) {
        Flow::DataPtr<Flow::Vector<f32>> out(new Flow::Vector<f32>(mean));
        out->setTimestamp(timestamps[i]);
        putData(0, out.get());
    }
    return putData(0, Flow::Data::eos());
}
