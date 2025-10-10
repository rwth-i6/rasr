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
#ifndef _SIGNAL_WINDOWFUNCTION_HH
#define _SIGNAL_WINDOWFUNCTION_HH

#include <Core/Parameter.hh>
#include <Core/Types.hh>
#include <math.h>

namespace Signal {

/** WindowFunction is base class for different windowing functions
 *
 *  In the particular windowing functions override the function
 *    virtual bool init() and fill in the member variable window_.
 */

class WindowFunction {
public:
    typedef f32 Float;
    enum Type { Rectangular,
                Hamming,
                Hanning,
                PeriodicHanning,
                Bartlett,
                Blackman,
                Kaiser };

    static Core::Choice          typeChoice;
    static Core::ParameterChoice paramType;

    static WindowFunction* create(Type type);

protected:
    std::vector<Float> window_;

    bool needInit_;

protected:
    virtual bool init() {
        return !(needInit_ = false);
    }

public:
    WindowFunction()
            : needInit_(true) {}

    virtual ~WindowFunction() {}

    void setLength(u32 l);

    u32 length() {
        return window_.size();
    }

    const std::vector<Float>& getWindow() {
        if (needInit_)
            init();
        return window_;
    }

    template<class Iterator>
    bool work(const Iterator& begin, const Iterator& end);
};

template<class Iterator>
bool WindowFunction::work(const Iterator& begin, const Iterator& end) {
    if (needInit_ && !init()) {
        return false;
    }

    size_t effectiveWindow = std::min<size_t>(std::distance(begin, end), length());

    std::transform(window_.begin(), window_.begin() + effectiveWindow, begin, begin, std::multiplies<Float>());

    // disregard samples that do not fit in window
    std::fill(begin + effectiveWindow, end, 0.0);

    return true;
}

/** RectangularWindowFunction */

class RectangularWindowFunction : public WindowFunction {
protected:
    virtual bool init();
};

/** BartlettWindowFunction */

class BartlettWindowFunction : public WindowFunction {
protected:
    virtual bool init();
};

/** HammingWindowFunction */

class HammingWindowFunction : public WindowFunction {
protected:
    virtual bool init();
};

/** HanningWindowFunction */

class HanningWindowFunction : public WindowFunction {
public:
    HanningWindowFunction(bool periodic)
            : periodic_(periodic) {}

protected:
    virtual bool init();

private:
    /**
     * Indicates if this is a periodic window or not, i.e. if the last value in the window equals the first.
     * Equivalent to Pytorch's periodic parameter for hann_window:
     * https://docs.pytorch.org/docs/stable/generated/torch.hann_window.html
     */
    bool periodic_;
};

/** BlackmanWindowFunction */

class BlackmanWindowFunction : public WindowFunction {
protected:
    virtual bool init();
};

}  // namespace Signal

#endif  // _SIGNAL_WINDOWFUNCTION_HH
