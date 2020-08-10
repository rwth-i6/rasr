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
#ifndef _SIGNAL_TEMPORAL_PATTERN_HH
#define _SIGNAL_TEMPORAL_PATTERN_HH

#include <Core/Parameter.hh>
#include <Flow/Node.hh>
#include <Flow/Vector.hh>
#include <Math/Vector.hh>

#include "CosineTransform.hh"
#include "WindowFunction.hh"

namespace Signal {

/** Estimation of temporal pattern (TRAP).
 *
 *  Implementation of the TRAP based feature extraction introduced in
 *  "TRAPs - Classifiers of Temporal Patterns" by H. Hermansky, S. Sharm
 *  ICSLP 1998.
 *
 *  The main idea is to use a large window of up to 50 frames for each dimension
 *  and apply a cosine transform for decorrelation and dimension reduction.
 */
class TemporalPattern {
public:
    typedef f32 Value;

    TemporalPattern();
    virtual ~TemporalPattern();

    void init(size_t nFeatures, size_t nFrames, size_t dctSize) {
        setFeatures(nFeatures);
        setFrames(nFrames);
        setDctSize(dctSize);
    }
    size_t nFeatures() const {
        return nFeatures_;
    }
    size_t nFrames() const {
        return nFrames_;
    }
    size_t dctSize() const {
        return dctSize_;
    }

    void                    setWindowFunction(Signal::WindowFunction* windowFunction);
    Signal::WindowFunction* windowFunction() const {
        return windowFunction_;
    }
    Signal::CosineTransform& cosineTransform() {
        return cosineTransform_;
    }

    /** Calculates cosine transform of @param in.
     *  If normalization is set at initialization, the result (@param out) is divided by N.
     *  Remark:
     *   Normalization is not included in the transformation matrix, neither in (inverse) FFT.
     */
    bool apply(std::vector<Value>& in, std::vector<Value>& out);

private:
    Math::Vector<Value> bandVec;
    Math::Vector<Value> dctVec;

    bool   needInit_;
    size_t nFeatures_;
    size_t nFrames_;
    size_t dctSize_;

    Signal::CosineTransform cosineTransform_;
    Signal::WindowFunction* windowFunction_;

    bool init();

    void setFeatures(size_t nFeatures) {
        if (nFeatures_ != nFeatures) {
            nFeatures_ = nFeatures;
            needInit_  = true;
        }
    }
    void setFrames(size_t nFrames) {
        if (nFrames_ != nFrames) {
            nFrames_  = nFrames;
            needInit_ = true;
        }
    }
    void setDctSize(size_t size) {
        if (dctSize_ != size) {
            dctSize_  = size;
            needInit_ = true;
        }
    }

    void getBand(size_t band, std::vector<Value>& in, std::vector<Value>& out);
    void setBand(size_t band, std::vector<Value>& in, std::vector<Value>& out);
    void applyWindow(std::vector<Value>& in);
    void applyDCT(std::vector<Value>& in, std::vector<Value>& out);
};

/** Calculate spectrum coefficients (CRBE) from autoregressive coefficients.
 *  Input: autoregressive-parameter.
 *  Output spectrum coefficients.
 *  Parameter: number of spectrum coefficients.
 */
class TemporalPatternNode : public Flow::SleeveNode, TemporalPattern {
    typedef Flow::SleeveNode Precursor;
    typedef f32              Value;

private:
    bool                            needInit_;
    void                            init(size_t length);
    size_t                          contextLength_;
    static const Core::ParameterInt paramContextLength;
    void                            setContextLength(size_t length) {
        if (contextLength_ != length) {
            contextLength_ = length;
            needInit_      = true;
        }
    };
    size_t                          outputSize_;
    static const Core::ParameterInt paramOutputSize;
    void                            setOutputSize(size_t size) {
        if (outputSize_ != size) {
            outputSize_ = size;
            needInit_   = true;
        }
    };

public:
    static std::string filterName() {
        return std::string("nn-temporal-pattern");
    };
    TemporalPatternNode(const Core::Configuration& c);
    virtual ~TemporalPatternNode();
    virtual bool configure();
    virtual bool setParameter(const std::string& name, const std::string& value);
    virtual bool work(Flow::PortId p);
};
}  // namespace Signal

#endif  // _SIGNAL_TEMPORALPATTERN_HH
