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
#ifndef _SIGNAL_WINDOW_HH
#define _SIGNAL_WINDOW_HH

#include <Core/Parameter.hh>

#include <Flow/Data.hh>
#include <Flow/Vector.hh>

#include "SlidingAlgorithmNode.hh"
#include "WindowBuffer.hh"
#include "WindowFunction.hh"

namespace Signal {

/** Window */

class Window : public WindowBuffer {
public:
    typedef WindowBuffer         Predecessor;
    typedef WindowBuffer::Time   Time;
    typedef WindowBuffer::Sample Sample;

    Window();
    virtual ~Window() = default;

    void setWindowFunction(std::unique_ptr<WindowFunction>&& windowFunction);

    void setSampleRate(f64 sampleRate);

    void setLengthInS(Time length);
    Time lengthInS() const {
        return lengthInS_;
    }

    void setInputLengthInS(Time length);
    Time inputLengthInS() const {
        return lengthInS_;
    }

    void setShiftInS(Time shift);
    Time shiftInS() const {
        return shiftInS_;
    }

    void setWindowOffsetInS(Time window_offset);
    Time windowOffsetInS() const {
        return windowOffsetInS_;
    }

protected:
    virtual void init();
    virtual void transform(Flow::Vector<Sample>& out);

private:
    Time lengthInS_;
    Time shiftInS_;
    Time inputLengthInS_;
    Time windowOffsetInS_;

    std::unique_ptr<WindowFunction> windowFunction_;
};

/** WindowNode */
class WindowNode : public SlidingAlgorithmNode<Window> {
public:
    typedef SlidingAlgorithmNode<Window> Predecessor;

private:
    static const Core::ParameterFloat paramShift;
    static const Core::ParameterFloat paramLength;
    static const Core::ParameterFloat paramInputLength;
    static const Core::ParameterFloat paramWindowOffset;
    static const Core::ParameterBool  paramFlushAll;
    static const Core::ParameterBool  paramFlushBeforeGap;

public:
    static std::string filterName() {
        return "signal-window";
    }

    WindowNode(const Core::Configuration& c);
    virtual ~WindowNode() {}

    virtual bool setParameter(const std::string& name, const std::string& value);
    virtual bool configure();
};

}  // namespace Signal

#endif  // _SIGNAL_WINDOW_HH
