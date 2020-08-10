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
#include "PolinomialVectorInterpolation.hh"
#include <Core/StringUtilities.hh>

using namespace Signal;

// PolinomialVectorInterpolation
////////////////////////////////

PolinomialVectorInterpolation::PolinomialVectorInterpolation()
        : equationSystemSolved_(false) {}

bool PolinomialVectorInterpolation::work(const Flow::Timestamp& timestamp, DataPointer& out) {
    Time time = timestamp.startTime();
    seek(time);
    if (!checkInterpolationTime(time))
        return false;
    if (copyControlPoint(time, out))
        return true;
    if (!equationSystemSolved_)
        calculateParameters();
    return calculateOutput(time, out);
}

void PolinomialVectorInterpolation::seek(Time time) {
    DataPointer dataPointer;

    while (slidingWindow_.size() < slidingWindow_.maxSize() ||
           !slidingWindow_.out(dataPointer) ||
           Core::isSignificantlyLess(dataPointer->startTime(), time, Flow::timeTolerance)) {
        if (!nextData(dataPointer))
            break;
        slidingWindow_.add(dataPointer);
        equationSystemSolved_ = false;
    }
}

bool PolinomialVectorInterpolation::checkInterpolationTime(Time time) {
    if (slidingWindow_.maxSize() >= 2) {
        verify(slidingWindow_.size() > 0);
        if (Core::isSignificantlyLess(time, slidingWindow_.back()->startTime(), Flow::timeTolerance) ||
            Core::isSignificantlyGreater(time, slidingWindow_.front()->startTime(), Flow::timeTolerance)) {
            lastError_ = Core::form("Target time %f lies outside of the input stream", time);
            return false;
        }
    }
    return true;
}

bool PolinomialVectorInterpolation::copyControlPoint(Time time, DataPointer& out) const {
    SlidingWindow<DataPointer>::ConstantIterator i = slidingWindow_.begin();
    for (; i != slidingWindow_.end(); ++i) {
        Time startTime = (*i)->startTime();

        if (Core::isSignificantlyLess(startTime, time, Flow::timeTolerance))
            return false;
        else if (Core::isAlmostEqual(startTime, time, Flow::timeTolerance)) {
            out = *i;
            return true;
        }
    }
    return false;
}

void PolinomialVectorInterpolation::resize() {
    verify(slidingWindow_.size() > 0);

    A_.resize(slidingWindow_.size(), slidingWindow_.size());
    B_.resize(slidingWindow_.size(), slidingWindow_.front()->size());
}

void PolinomialVectorInterpolation::calculateParameters() {
    verify(!equationSystemSolved_);
    resize();
    for (u32 row = 0; row < A_.nRows(); ++row) {
        Data& controlPoint = **(slidingWindow_.reverseBegin() + row);
        verify(controlPoint.size() == B_.nColumns());

        for (u32 column = 0; column < A_.nColumns(); ++column)
            A_(row, column) = pow(controlPoint.startTime(), column);

        for (u32 dimension = 0; dimension < B_.nColumns(); ++dimension)
            B_(row, dimension) = controlPoint[dimension];
    }
    if (getrf(A_, pivotIndices_) != 0 || getrs(A_, B_, pivotIndices_) != 0)
        defect();
    equationSystemSolved_ = true;
}

bool PolinomialVectorInterpolation::calculateOutput(Time time, DataPointer& out) {
    verify(equationSystemSolved_);

    out = DataPointer(new Data);
    out->setStartTime(time);
    out->setEndTime(time);
    out->resize(B_.nColumns());
    std::fill(out->begin(), out->end(), 0);

    for (u32 row = 0; row < B_.nRows(); ++row) {
        for (u32 dimension = 0; dimension < B_.nColumns(); ++dimension)
            (*out)[dimension] += B_(row, dimension) * pow(time, row);
    }
    return true;
}

void PolinomialVectorInterpolation::setOrder(u32 order) {
    reset();
    slidingWindow_.init(order + 1, order / 2);
}

void PolinomialVectorInterpolation::reset() {
    slidingWindow_.clear();
    equationSystemSolved_ = false;
    lastError_            = "";
}
