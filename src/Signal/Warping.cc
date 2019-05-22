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
#include "Warping.hh"
#include <Flow/Vector.hh>
#include <numeric>

using namespace Core;
using namespace Signal;

//===============================================================================================
Warping::Warping()
        : inputSize_(0) {}

void Warping::setWarpingFunction(Math::UnaryAnalyticFunctionRef warpingFunction, size_t inputSize,
                                 MergeType mergeType, InterpolationType interpolationType) {
    require(warpingFunction);

    inputSize_ = inputSize;
    deleteInverseItems();
    inverseWarpingFunction_.resize((size_t)Core::floor(warpingFunction->value(inputSize - 1) + 1));

    u32 index, previousIndex               = 0;
    u32 warpingIndex, previousWarpingIndex = (u32)Core::floor(warpingFunction->value(previousIndex));

    for (index = previousIndex + 1; index < inputSize; ++index) {
        warpingIndex = (u32)Core::floor(warpingFunction->value(index));
        // only monoton increasing warping functions are supported
        verify(warpingIndex >= previousWarpingIndex);

        if (warpingIndex != previousWarpingIndex) {
            verify(inverseWarpingFunction_.size() > warpingIndex);

            inverseWarpingFunction_[previousWarpingIndex] = createMerger(mergeType, previousIndex, index);
            ensure(inverseWarpingFunction_[previousWarpingIndex] != 0);

            for (u32 w = previousWarpingIndex + 1; w < warpingIndex; ++w) {
                f64 relativePosition = f32(w - previousWarpingIndex) / f32(warpingIndex - previousWarpingIndex);

                inverseWarpingFunction_[w] = createInterpolator(interpolationType, previousIndex, index, relativePosition);
                ensure(inverseWarpingFunction_[w] != 0);
            }

            previousWarpingIndex = warpingIndex;
            previousIndex        = index;
        }
    }
    inverseWarpingFunction_[previousWarpingIndex] = createMerger(mergeType, previousIndex, index);
    ensure(inverseWarpingFunction_[previousWarpingIndex] != 0);
}

void Warping::setInverseWarpingFunction(Math::UnaryAnalyticFunctionRef inverseWarpingFunction,
                                        size_t inputSize, InterpolationType interpolationType) {
    require(inverseWarpingFunction);

    inputSize_ = inputSize;
    deleteInverseItems();

    for (u32 warpingIndex = 0;; ++warpingIndex) {
        FloatIndex floatIndex = inverseWarpingFunction->value(warpingIndex);
        ensure(floatIndex >= 0);

        u32 leftOfIndex      = (u32)Core::floor(floatIndex);
        f64 relativePosition = floatIndex - leftOfIndex;

        Warping::InverseItem* interpolator = createInterpolator(interpolationType,
                                                                leftOfIndex,
                                                                leftOfIndex + 1,
                                                                relativePosition);
        if (interpolator != 0)
            inverseWarpingFunction_.push_back(interpolator);
        else
            return;
    }
}

void Warping::deleteInverseItems() {
    for (u32 warpingIndex = 0; warpingIndex < inverseWarpingFunction_.size(); ++warpingIndex)
        delete inverseWarpingFunction_[warpingIndex];
    inverseWarpingFunction_.clear();
}

Warping::InverseItem* Warping::createMerger(MergeType type, u32 begin, u32 end) const {
    require(begin < end);

    if (begin >= inputSize_ || end > inputSize_)
        return 0;
    else if (begin + 1 == end) {
        return new CopyInverseItem(begin);
    }
    else {
        switch (type) {
            case AritmeticMean: return new AritmeticMeanInverseItem(begin, end);
            case SelectBegin: return new SelectBeginInverseItem(begin, end);
        }
    }
    defect();
}

Warping::InverseItem* Warping::createInterpolator(InterpolationType type, u32 first, u32 last, f64 relativePosition) const {
    require(first < last);
    require(0.0 <= relativePosition && relativePosition <= 1.0);

    if (first >= inputSize_) {
        return 0;
    }
    else if (last >= inputSize_) {
        if (equalRelativePosition(relativePosition, 0)) {
            return new CopyInverseItem(first);
        }
        else
            return 0;
    }
    else if (relativePosition == 0) {
        // using equalRelativePosition here could result in more CopyInverseItems thus faster warping
        return new CopyInverseItem(first);
    }
    else if (relativePosition == 1) {
        // using equalRelativePosition here could result in more CopyInverseItems thus faster warping
        return new CopyInverseItem(last);
    }
    else {
        switch (type) {
            case InsertZero: return new InsertZeroInverseItem(first, last, relativePosition);
            case KeepEnd: return new KeepEndInverseItem(first, last, relativePosition);
            case LinearInterpolation: return new LinearInterpolationInverseItem(first, last, relativePosition);
        }
    }
    defect();
}

bool Warping::equalRelativePosition(f64 x, f64 y) {
    require(0 <= x && x <= 1.0);
    require(0 <= y && y <= 1.0);
    static f64 tolerance = 1e-10;
    return Core::abs(x - y) < tolerance;
}

void Warping::apply(const std::vector<Data>& in, std::vector<Data>& out) const {
    require(in.size() == inputSize_);
    out.resize(inverseWarpingFunction_.size());

    for (u32 warpingIndex = 0; warpingIndex < inverseWarpingFunction_.size(); ++warpingIndex)
        out[warpingIndex] = inverseWarpingFunction_[warpingIndex]->apply(in);
}

//===============================================================================================
const Choice WarpingNode::choiceMergeType(
        "aritmetic-mean", Warping::AritmeticMean,
        "select-begin", Warping::SelectBegin,
        Choice::endMark());
const ParameterChoice WarpingNode::paramMergeType(
        "merge-type", &choiceMergeType, "merge type of inverse warping function", Warping::AritmeticMean);

const Choice WarpingNode::choiceInterpolationType(
        "step-function", Warping::KeepEnd,
        "insert-zero", Warping::InsertZero,
        "linear", Warping::LinearInterpolation,
        Choice::endMark());
const ParameterChoice WarpingNode::paramInterpolationType(
        "interpolation-type", &choiceInterpolationType,
        "interpolation type of inverse warping function", Warping::KeepEnd);

const ParameterBool WarpingNode::paramInterpolateOverWarpedAxis(
        "interpolate-over-warped-axis",
        "yes: interpolation and merge done over warped axis. No: interpolation done over original axis",
        true);

WarpingNode::WarpingNode(const Core::Configuration& c)
        : Component(c),
          Node(c),
          needInit_(true),
          inputSize_(0),
          sampleRate_(0),
          mergeType_(Warping::AritmeticMean),
          interpolationType_(Warping::KeepEnd),
          interpolateOverWarpedAxis_(true) {
    addInput(0);
    addOutput(0);

    setMergeType((Warping::MergeType)paramMergeType(c));
    setInterpolationType((Warping::InterpolationType)paramInterpolationType(c));
    setInterpolateOverWarpedAxis(paramInterpolateOverWarpedAxis(c));
}

void WarpingNode::setSampleRate(f64 sampleRate) {
    if (sampleRate_ != sampleRate) {
        sampleRate_ = sampleRate;
        setNeedInit();
    }
}

void WarpingNode::setMergeType(const Warping::MergeType type) {
    if (mergeType_ != type) {
        mergeType_ = type;
        setNeedInit();
    }
}

void WarpingNode::setInterpolationType(const Warping::InterpolationType type) {
    if (interpolationType_ != type) {
        interpolationType_ = type;
        setNeedInit();
    }
}

void WarpingNode::setInterpolateOverWarpedAxis(bool interpolateOverWarpedAxis) {
    if (interpolateOverWarpedAxis_ != interpolateOverWarpedAxis) {
        interpolateOverWarpedAxis_ = interpolateOverWarpedAxis;
        setNeedInit();
    }
}

void WarpingNode::init(size_t inputSize) {
    inputSize_ = inputSize;
    initWarping();
    resetNeedInit();
}

bool WarpingNode::setParameter(const std::string& name, const std::string& value) {
    if (paramMergeType.match(name))
        setMergeType((Warping::MergeType)paramMergeType(value));
    else if (paramInterpolationType.match(name))
        setInterpolationType((Warping::InterpolationType)paramInterpolationType(value));
    else if (paramInterpolateOverWarpedAxis.match(name))
        setInterpolateOverWarpedAxis(paramInterpolateOverWarpedAxis(value));
    else
        return false;

    return true;
}

bool WarpingNode::configure(const Flow::Attributes& successorAttributes) {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    getInputAttributes(0, *attributes);
    if (!configureDatatype(attributes, Flow::Vector<f32>::type()))
        return false;
    setSampleRate(atof(attributes->get("sample-rate").c_str()));
    attributes->merge(successorAttributes);
    attributes->set("datatype", Flow::Vector<f32>::type()->name());
    return putOutputAttributes(0, attributes);
}

bool WarpingNode::work(Flow::PortId p) {
    Flow::DataPtr<Flow::Vector<f32>> in;
    if (!getData(0, in))
        return putData(0, in.get());

    if (needInit_)
        init(in->size());

    if (in->size() != inputSize_)
        criticalError("Input size (%zd) does not match the expected input size (%zd)",
                      in->size(), inputSize_);

    Flow::Vector<f32>* out = new Flow::Vector<f32>();
    out->setTimestamp(*in);
    apply(*in, *out);
    return putData(0, out);
}
