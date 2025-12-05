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
#include <RandomVector.hh>

using namespace Signal;

const Core::ParameterInt RandomVectorNode::paramSize(
        "size", "number of components", 1, 0);
const Core::ParameterFloat RandomVectorNode::paramStartTime(
        "start-time", "start time of the first vector of this segment.", 0);
const Core::ParameterFloat RandomVectorNode::paramSampleRate(
        "sample-rate", "sample rate of the output vectors", 1);
const Core::ParameterFloat RandomVectorNode::paramFrameShift(
        "frame-shift", "difference between the start time of two subsequent vectors", 1);

RandomVectorNode::RandomVectorNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          randomVectorGenerator_(0),
          size_(0),
          sampleRate_(1),
          startTime_(0),
          frameShift_(1),
          nOutputs_(0) {
    setType((Math::RandomVectorGenerator::Type)Math::RandomVectorGenerator::paramType(c));
    size_       = paramSize(c);
    sampleRate_ = paramSampleRate(c);
    startTime_  = paramStartTime(c);
    frameShift_ = paramFrameShift(c);
}

RandomVectorNode::~RandomVectorNode() {
    delete randomVectorGenerator_;
}

bool RandomVectorNode::configure() {
    reset();
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    attributes->set("datatype", Flow::Vector<Data>::type()->name());
    attributes->set("sample-rate", sampleRate_);
    attributes->set("frame-shift", frameShift_);
    return putOutputAttributes(0, attributes);
}

bool RandomVectorNode::setParameter(const std::string& name, const std::string& value) {
    if (Math::RandomVectorGenerator::paramType.match(name))
        setType((Math::RandomVectorGenerator::Type)Math::RandomVectorGenerator::paramType(value));
    else if (paramSize.match(name))
        size_ = paramSize(value);
    else if (paramSampleRate.match(name))
        sampleRate_ = paramSampleRate(value);
    else if (paramStartTime.match(name))
        startTime_ = paramStartTime(value);
    else if (paramFrameShift.match(name))
        frameShift_ = paramFrameShift(value);
    else
        return false;
    return true;
}

void RandomVectorNode::setType(Math::RandomVectorGenerator::Type type) {
    delete randomVectorGenerator_;
    randomVectorGenerator_ = Math::RandomVectorGenerator::create(type);
}

bool RandomVectorNode::work(Flow::PortId p) {
    Flow::Vector<Data>* result = createOutput();
    ensure(result != 0);
    nOutputs_++;
    return putData(0, result);
}

Flow::Vector<RandomVectorNode::Data>* RandomVectorNode::createOutput() const {
    Flow::Vector<Data>* result = new Flow::Vector<Data>(size_);
    verify(randomVectorGenerator_ != 0);
    randomVectorGenerator_->work(*result);
    result->setStartTime(startTime_ + nOutputs_ * frameShift_);
    result->setEndTime(result->startTime() + size_ / sampleRate_);
    return result;
}
