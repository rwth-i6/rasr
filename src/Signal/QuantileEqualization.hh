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
#ifndef _SIGNAL_QUANTILE_EQUALIZATION_HH
#define _SIGNAL_QUANTILE_EQUALIZATION_HH

#include <Core/Parameter.hh>
#include <Core/Types.hh>
#include <Flow/Vector.hh>

#include "Node.hh"
#include "SlidingWindow.hh"

namespace Signal {

class QuantileEqualization {
private:
    typedef Flow::DataPtr<Flow::Vector<f32>> Frame;

    SlidingWindow<Frame> slidingWindow_;
    u32                  length_;
    u32                  right_;

    u32 numberOfQuantiles_;

    std::vector<f32> trainingQuantile_;
    std::vector<f32> currentQuantile_;
    std::vector<f64> quantileSum_;

    std::vector<f32> alpha_;
    std::vector<f32> gamma_;
    std::vector<f32> lambda_;
    std::vector<f32> rho_;

    f64 deltaAlpha_;
    f64 deltaGamma_;
    f64 deltaLambda_;
    f64 deltaRho_;
    f32 beta_;

    f32 overestimationFactor_;

    std::vector<f32> mean_;
    std::vector<f32> dev_;

    u32 frameCounter_;

    bool wroteQuantiles_;
    bool changed_;
    bool newIn_;

    bool        equalizeQuantiles_;
    bool        combineNeighbors_;
    bool        estimateQuantiles_;
    std::string filename_;
    bool        poolQuantiles_;
    bool        piecewiseLinear_;

    bool normalizeMean_;
    bool normalizeVariance_;

    f32 max_;

    void updateTransformationParameters();
    void applyTransformations(Frame& out);

    void writeEstimatedQuantilesToFile();
    void readTrainingQuantilesFromFile();

    void equalizeQuantiles(Frame& out) const {
        if (piecewiseLinear_) {
            u32 i;
            f32 y1, y2, x1, x2, a, b;

            for (u16 d = 0; d < dim(); d++) {
                i = 0;

                while ((currentQuantile_[(i + 1) * dim() + d] < (*out)[d]) && (i < numberOfQuantiles_))
                    i++;

                y1 = trainingQuantile_[i * dim() + d];
                y2 = trainingQuantile_[(i + 1) * dim() + d];
                x1 = currentQuantile_[i * dim() + d];
                x2 = currentQuantile_[(i + 1) * dim() + d];

                a = (y2 - y1) / (x2 - x1);
                b = y1 - a * x1;

                (*out)[d] = a * (*out)[d] + b;
            }
        }
        else {
            f32 scaledValue, maximalQuantile;
            for (u16 d = 0; d < dim(); d++) {
                maximalQuantile = std::max(overestimationFactor_ * trainingQuantile_[numberOfQuantiles_ * dim() + d],
                                           overestimationFactor_ * currentQuantile_[numberOfQuantiles_ * dim() + d]);
                scaledValue     = (*out)[d] / maximalQuantile;

                (*out)[d] = maximalQuantile * (alpha_[d] * pow(scaledValue, gamma_[d]) + (1. - alpha_[d]) * scaledValue);
            }
        }
    }
    void combineNeighbors(Frame& out) const {
        Frame tmp;
        tmp = out;
        tmp.makePrivate();

        for (u16 d = 0; d < dim(); d++)
            (*tmp)[d] = (1. - lambda_[d] - rho_[d]) * (*out)[d] + lambda_[d] * (*out)[std::max((s32)d - 1, 0)] + rho_[d] * (*out)[std::min((s32)d + 1, (s32)dim() - 1)];
        for (u16 d = 0; d < dim(); d++)
            (*out)[d] = (*tmp)[d];
    }
    void normalizeMean(Frame& out) const {
        for (u16 d = 0; d < dim(); d++)
            (*out)[d] -= mean_[d];
    }
    void normalizeVariance(Frame& out) const {
        for (u16 d = 0; d < dim(); d++)
            (*out)[d] /= dev_[d];
    }

protected:
    bool needInit_;

    virtual void init(u16 dim);

    u32 dim() const {
        return mean_.size();
    };

public:
    QuantileEqualization()
            : length_(0),
              right_(0),
              numberOfQuantiles_(0),
              changed_(true),
              equalizeQuantiles_(true),
              combineNeighbors_(false),
              normalizeMean_(true),
              normalizeVariance_(false),
              needInit_(true) {}

    virtual ~QuantileEqualization() {
        reset();
    }

    void setQuantileEqualization(bool norm) {
        if (equalizeQuantiles_ != norm) {
            equalizeQuantiles_ = norm;
            reset();
        }
    }
    void setCombineNeighbors(bool norm) {
        if (combineNeighbors_ != norm) {
            combineNeighbors_ = norm;
            reset();
        }
    }
    void setPoolQuantiles(bool norm) {
        if (poolQuantiles_ != norm) {
            poolQuantiles_ = norm;
            reset();
        }
    }
    void setPiecewiseLinear(bool norm) {
        if (piecewiseLinear_ != norm) {
            piecewiseLinear_ = norm;
            reset();
        }
    }
    void setQuantileEstimation(bool norm) {
        if (estimateQuantiles_ != norm) {
            estimateQuantiles_ = norm;
            reset();
        }
    }
    void setQuantileFile(std::string filename) {
        if (filename_ != filename) {
            filename_ = filename;
            reset();
        }
    }
    void setNormalizeMean(bool norm) {
        if (normalizeMean_ != norm) {
            normalizeMean_ = norm;
            reset();
        }
    }
    void setNormalizeVariance(bool norm) {
        if (normalizeVariance_ != norm) {
            normalizeVariance_ = norm;
            reset();
        }
    }
    void setLength(u32 length) {
        if (length_ != length) {
            length_ = length;
            reset();
        }
    }
    void setRight(u32 right) {
        if (right_ != right) {
            right_ = right;
            reset();
        }
    }
    void setNumberOfQuantiles(u32 numberOfQuantiles) {
        if (numberOfQuantiles_ != numberOfQuantiles) {
            numberOfQuantiles_ = numberOfQuantiles;
            reset();
        }
    }
    void setOverestimationFactor(f32 overestimationFactor) {
        if (overestimationFactor_ != overestimationFactor) {
            overestimationFactor_ = overestimationFactor;
            reset();
        }
    }
    void setDeltaAlpha(f32 deltaAlpha) {
        if (deltaAlpha_ != deltaAlpha) {
            deltaAlpha_ = deltaAlpha;
            reset();
        }
    }
    void setDeltaGamma(f32 deltaGamma) {
        if (deltaGamma_ != deltaGamma) {
            deltaGamma_ = deltaGamma;
            reset();
        }
    }
    void setDeltaLambdaAndRho(f32 deltaLambdaAndRho) {
        if (deltaLambda_ != deltaLambdaAndRho) {
            deltaLambda_ = deltaLambdaAndRho;
            reset();
        }
        if (deltaRho_ != deltaLambdaAndRho) {
            deltaRho_ = deltaLambdaAndRho;
            reset();
        }
    }
    void setBeta(f32 beta) {
        if (beta_ != beta) {
            beta_ = beta;
            reset();
        }
    }
    bool update(const Frame& in, Frame& out);
    bool flush(Frame& out) {
        if (needInit_)
            return false;
        return update(Frame(), out);
    }

    virtual void reset() {
        needInit_ = true;
    }
};

class QuantileEqualizationNode : public SleeveNode, QuantileEqualization {
private:
    static Core::ParameterBool   paramQuantileEqualization;
    static Core::ParameterBool   paramCombineNeighbors;
    static Core::ParameterBool   paramQuantileEstimation;
    static Core::ParameterBool   paramPoolQuantiles;
    static Core::ParameterBool   paramPiecewiseLinear;
    static Core::ParameterString paramQuantileFile;
    static Core::ParameterBool   paramMeanNormalization;
    static Core::ParameterBool   paramVarianceNormalization;
    static Core::ParameterInt    paramLength;
    static Core::ParameterInt    paramRight;
    static Core::ParameterInt    paramNumberOfQuantiles;
    static Core::ParameterFloat  paramOverestimationFactor;
    static Core::ParameterFloat  paramDeltaAlpha;
    static Core::ParameterFloat  paramDeltaGamma;
    static Core::ParameterFloat  paramDeltaLambdaAndRho;
    static Core::ParameterFloat  paramBeta;

public:
    static std::string filterName() {
        return "signal-quantile-equalization";
    }
    QuantileEqualizationNode(const Core::Configuration& c)
            : Core::Component(c), SleeveNode(c) {
        setQuantileEqualization(paramQuantileEqualization(c));
        setCombineNeighbors(paramCombineNeighbors(c));
        setQuantileEstimation(paramQuantileEstimation(c));
        setQuantileFile(paramQuantileFile(c));
        setPoolQuantiles(paramPoolQuantiles(c));
        setNormalizeMean(paramMeanNormalization(c));
        setNormalizeVariance(paramVarianceNormalization(c));
        setLength(paramLength(c));
        setRight(paramRight(c));
        setNumberOfQuantiles(paramNumberOfQuantiles(c));
        setOverestimationFactor(paramOverestimationFactor(c));
        setDeltaAlpha(paramDeltaAlpha(c));
        setDeltaGamma(paramDeltaGamma(c));
        setDeltaLambdaAndRho(paramDeltaLambdaAndRho(c));
        setBeta(paramBeta(c));
        addDatatype(Flow::Vector<f32>::type());
    }

    virtual ~QuantileEqualizationNode() {}

    virtual bool setParameter(const std::string& name, const std::string& value) {
        if (paramQuantileEqualization.match(name))
            setQuantileEqualization(paramQuantileEqualization(value));
        else if (paramCombineNeighbors.match(name))
            setCombineNeighbors(paramCombineNeighbors(value));
        else if (paramQuantileEstimation.match(name))
            setQuantileEstimation(paramQuantileEstimation(value));
        else if (paramPoolQuantiles.match(name))
            setPoolQuantiles(paramPoolQuantiles(value));
        else if (paramQuantileFile.match(name))
            setQuantileFile(paramQuantileFile(value));
        else if (paramMeanNormalization.match(name))
            setNormalizeMean(paramMeanNormalization(value));
        else if (paramVarianceNormalization.match(name))
            setNormalizeVariance(paramVarianceNormalization(value));
        else if (paramLength.match(name))
            setLength(paramLength(value));
        else if (paramRight.match(name))
            setRight(paramRight(value));
        else if (paramNumberOfQuantiles.match(name))
            setNumberOfQuantiles(paramNumberOfQuantiles(value));
        else if (paramOverestimationFactor.match(name))
            setOverestimationFactor(paramOverestimationFactor(value));
        else if (paramDeltaAlpha.match(name))
            setDeltaAlpha(paramDeltaAlpha(value));
        else if (paramDeltaGamma.match(name))
            setDeltaGamma(paramDeltaGamma(value));
        else if (paramDeltaLambdaAndRho.match(name))
            setDeltaLambdaAndRho(paramDeltaLambdaAndRho(value));
        else if (paramBeta.match(name))
            setBeta(paramBeta(value));
        else
            return false;
        return true;
    }

    virtual bool configure() {
        Core::Ref<const Flow::Attributes> a = getInputAttributes(0);
        if (!configureDatatype(a, Flow::Vector<f32>::type()))
            return false;
        reset();
        return putOutputAttributes(0, a);
    }

    virtual void reset() {
        QuantileEqualization::reset();
    }

    virtual bool work(Flow::PortId p) {
        Flow::DataPtr<Flow::Vector<f32>> in, out;

        while (getData(0, in)) {
            if (update(in, out))
                return putData(0, out.get());
        }

        // in is invalid
        while (flush(out))
            putData(0, out.get());

        reset();
        return putData(0, in.get());
    }
};

}  // namespace Signal

#endif  // _SIGNAL_QUANTILE_EQUALIZATION_HH
