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
#include "Window.hh"
#include <Core/Assertions.hh>
#include <math.h>

using namespace Flow;
using namespace Signal;

// Window
/////////

Window::Window()
        : lengthInS_(0.0),
          shiftInS_(0.0),
          inputLengthInS_(0.0),
          windowOffsetInS_(0.0),
          windowFunction_(nullptr) {}

void Window::setWindowFunction(std::unique_ptr<WindowFunction>&& windowFunction) {
    windowFunction_ = std::move(windowFunction);
}

void Window::setLengthInS(Time length) {
    if (lengthInS_ != length) {
        lengthInS_ = length;
        setNeedInit();
    }
}

void Window::setShiftInS(Time shift) {
    if (shiftInS_ != shift) {
        shiftInS_ = shift;
        setNeedInit();
    }
}

void Window::setInputLengthInS(Time inputLength) {
    if (inputLengthInS_ != inputLength) {
        inputLengthInS_ = inputLength;
        setNeedInit();
    }
}

void Window::setWindowOffsetInS(Time windowOffset) {
    windowOffsetInS_ = windowOffset;
}

void Window::setSampleRate(f64 sampleRate) {
    require(sampleRate > 0.0);
    if (WindowBuffer::sampleRate() != sampleRate) {
        WindowBuffer::setSampleRate(sampleRate);
        setNeedInit();
    }
}

void Window::init() {
    verify(windowFunction_.get());
    verify(sampleRate() > 0);

    if (inputLengthInS_ > 0) {
        setLength(rint(inputLengthInS_ * sampleRate()));
    }
    else {
        setLength(rint(lengthInS_ * sampleRate()));
    }
    setShift(rint(shiftInS_ * sampleRate()));

    Predecessor::init();
}

void Window::transform(Vector<Sample>& out) {
    u32 offset = rint(windowOffsetInS_ * sampleRate());
    offset     = std::min<u32>(out.size(), offset);

    u32 windowLength = rint(lengthInS_ * sampleRate());
    windowFunction_->setLength(windowLength);

    std::fill(out.begin(), out.begin() + offset, 0.0);
    if (!windowFunction_->work(out.begin() + offset, out.end())) {
        hope(false);
    }
}

// WindowNode
/////////////

const Core::ParameterFloat WindowNode::paramShift(
        "shift", "shift of window", 0.0, 0.0);

const Core::ParameterFloat WindowNode::paramLength(
        "length", "length of window function (in seconds) that is applied to the input", 0.0, 0.0);

const Core::ParameterFloat WindowNode::paramInputLength(
        "input-length", "length of the input processed by the window (if not set same as length), samples not covered by the window function are set to 0", 0.0, 0.0);

const Core::ParameterFloat WindowNode::paramWindowOffset(
        "window-offset", "Window is applied starting at an offset to the signal", 0, 0);

const Core::ParameterBool WindowNode::paramFlushAll(
        "flush-all", "if false, segments stops after the last sample was delivered", false);

const Core::ParameterBool WindowNode::paramFlushBeforeGap(
        "flush-before-gap", "if true, flushes before a gap in the input samples", true);

WindowNode::WindowNode(const Core::Configuration& c)
        : Component(c), Predecessor(c) {
    setWindowFunction(std::unique_ptr<WindowFunction>(WindowFunction::create(static_cast<WindowFunction::Type>(WindowFunction::paramType(c)))));
    setShiftInS(paramShift(c));
    setLengthInS(paramLength(c));
    setInputLengthInS(paramInputLength(c));
    setWindowOffsetInS(paramWindowOffset(c));
    setFlushAll(paramFlushAll(c));
    setFlushBeforeGap(paramFlushBeforeGap(c));
}

bool WindowNode::setParameter(const std::string& name, const std::string& value) {
    if (WindowFunction::paramType.match(name)) {
        setWindowFunction(std::unique_ptr<WindowFunction>(WindowFunction::create(static_cast<WindowFunction::Type>WindowFunction::paramType(value)))));
    }
    else if (paramShift.match(name)) {
        setShiftInS(paramShift(value));
    }
    else if (paramLength.match(name)) {
        setLengthInS(paramLength(value));
    }
    else if (paramInputLength.match(name)) {
        setInputLengthInS(paramInputLength(value));
    }
    else if (paramWindowOffset.match(name)) {
        setWindowOffsetInS(paramWindowOffset(value));
    }
    else if (paramFlushAll.match(name)) {
        setFlushAll(paramFlushAll(value));
    }
    else if (paramFlushBeforeGap.match(name)) {
        setFlushBeforeGap(paramFlushBeforeGap(value));
    }
    else {
        return false;
    }

    return true;
}

bool WindowNode::configure() {
    Core::Ref<Flow::Attributes> a(new Flow::Attributes());
    getInputAttributes(0, *a);

    if (!configureDatatype(a, Flow::Vector<f32>::type())) {
        return false;
    }

    a->set("frame-shift", shiftInS());
    f64 sampleRate = atof(a->get("sample-rate").c_str());
    if (sampleRate > 0.0) {
        setSampleRate(sampleRate);
    }
    else {
        criticalError("Sample rate is not positive: %f", sampleRate);
    }
    reset();

    return putOutputAttributes(0, a);
}
