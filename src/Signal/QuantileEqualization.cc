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
#include <Core/Assertions.hh>

#include "QuantileEqualization.hh"

using namespace Signal;
using namespace Core;
using namespace Flow;

// Quantile Equalization and
// joint mean/variance normalization
// without additional delay
////////////////////////////////

void QuantileEqualization::init(u16 dim) {
    static bool firstcall = true;
    mean_.resize(dim);
    dev_.resize(dim);

    trainingQuantile_.resize(dim * (numberOfQuantiles_ + 1));
    currentQuantile_.resize(dim * (numberOfQuantiles_ + 1));

    alpha_.resize(dim);
    gamma_.resize(dim);
    lambda_.resize(dim);
    rho_.resize(dim);

    for (u16 d = 0; d < dim; d++) {
        alpha_[d]  = 0.;
        gamma_[d]  = 1.;
        lambda_[d] = 0.;
        rho_[d]    = 0.;
    }

    if (estimateQuantiles_ && firstcall) {
        firstcall     = false;
        frameCounter_ = 0;
        quantileSum_.resize(dim * (numberOfQuantiles_ + 1));
        for (u32 i = 0; i < dim * (numberOfQuantiles_ + 1); i++)
            quantileSum_[i] = 0.;
    }

    if (equalizeQuantiles_ && !estimateQuantiles_ && firstcall) {
        firstcall = false;
        readTrainingQuantilesFromFile();
    }

    wroteQuantiles_ = false;
    changed_        = true;

    if (!slidingWindow_.init(length_, right_))
        hope(false);

    needInit_ = false;
}

void QuantileEqualization::readTrainingQuantilesFromFile() {
    FILE* f;
    if ((f = fopen(filename_.c_str(), "rt")) == NULL) {
        fprintf(stderr, "Can't open training quantile file: \"%s\"", filename_.c_str());
        hope(false);
    }
    else {
        u32 i;
        int res;
        for (u16 d = 0; d < dim(); d++) {
            res = fscanf(f, "%u ", &i);
            for (u16 i = 0; i <= numberOfQuantiles_; i++)
                res = fscanf(f, "%f ", &trainingQuantile_[i * dim() + d]);
            res = fscanf(f, "\n");
        }
    }
    fclose(f);

    if (poolQuantiles_) {
        for (u16 i = 0; i <= numberOfQuantiles_; i++) {
            f32 average = 0.;

            for (u16 d = 0; d < dim(); d++)
                average += trainingQuantile_[i * dim() + d];

            average /= dim();

            for (u16 d = 0; d < dim(); d++) {
                trainingQuantile_[i * dim() + d] = average;
            }
        }
    }
}

void QuantileEqualization::writeEstimatedQuantilesToFile() {
    FILE* f;
    if ((f = fopen(filename_.c_str(), "wt")) == NULL) {
        fprintf(stderr, "Can't open training quantile output file: \"%s\"", filename_.c_str());
        hope(false);
    }
    else {
        for (u16 d = 0; d < dim(); d++) {
            fprintf(f, "%i ", d);
            for (u16 i = 0; i <= numberOfQuantiles_; i++) {
                fprintf(f, "%f ", quantileSum_[i * dim() + d] / frameCounter_);
            }
            fprintf(f, "\n");
        }
    }
    fclose(f);
}

bool QuantileEqualization::update(const Frame& in, Frame& out) {
    if (needInit_)
        init(in->size());

    Frame removed;
    if (in) {
        slidingWindow_.add(in);
        slidingWindow_.removed(removed);
    }
    else
        slidingWindow_.flushOut();

    if (in || removed)
        changed_ = true;

    if (in)
        newIn_ = true;

    out = Frame();
    if (slidingWindow_.out(out)) {
        out.makePrivate();
        applyTransformations(out);
        return true;
    }

    return false;
}

void QuantileEqualization::updateTransformationParameters() {
    if (!changed_)
        return;

    SlidingWindow<Frame> slidingTransformedWindow;
    if (!slidingTransformedWindow.init(length_, right_))
        hope(false);

    if (equalizeQuantiles_) {
        // determine quantiles
        std::vector<f32> sortedList;
        sortedList.clear();
        sortedList.resize(slidingWindow_.size());

        for (u16 d = 0; d < dim(); d++) {
            for (u32 i = 0; i < sortedList.size(); i++)
                sortedList[i] = (*slidingWindow_[i])[d];

            sort(sortedList.begin(), sortedList.end());

            for (u16 i = 0; i <= numberOfQuantiles_; i++) {
                currentQuantile_[i * dim() + d] = sortedList[(u32)(i * (sortedList.size() - 1) / numberOfQuantiles_)];
                if (estimateQuantiles_)
                    quantileSum_[i * dim() + d] += currentQuantile_[i * dim() + d];
            }
        }

        // update parameters
        if (!estimateQuantiles_ && !piecewiseLinear_) {
            f32 lowA, highA, lowG, highG, a, g, i;
            f32 lowL, highL, lowR, highR, r, l;
            f32 minimalDistance, distance;
            f32 scaledQuantile, maximalQuantile, transformedQuantile, tmp;

            for (u16 d = 0; d < dim(); d++) {
                if ((s32)right_ == Type<s32>::max) {  // utterance wise: search whole grid
                    lowA  = 0.;
                    highA = 1.;
                    lowG  = 1.0;
                    highG = 3.0;
                }
                else {  // online: just update
                    lowA  = std::max(alpha_[d] - deltaAlpha_, 0.);
                    highA = std::min(alpha_[d] + deltaAlpha_, 1.);
                    lowG  = std::max(gamma_[d] - deltaGamma_, 1.);
                    highG = std::min(gamma_[d] + deltaGamma_, 3.);
                }

                minimalDistance = Type<f32>::max;

                maximalQuantile = std::max(overestimationFactor_ * trainingQuantile_[numberOfQuantiles_ * dim() + d],
                                           overestimationFactor_ * currentQuantile_[numberOfQuantiles_ * dim() + d]);

                for (a = lowA; a <= highA; a += deltaAlpha_) {
                    for (g = lowG; g <= highG; g += deltaGamma_) {
                        distance = 0.;

                        for (i = 1; i < numberOfQuantiles_; i++) {
                            scaledQuantile      = std::max(trainingQuantile_[i * dim() + d], currentQuantile_[i * dim() + d]) / maximalQuantile;
                            transformedQuantile = maximalQuantile * (a * pow(scaledQuantile, g) + (1. - a) * scaledQuantile);
                            tmp                 = transformedQuantile - trainingQuantile_[i * dim() + d];
                            distance += tmp * tmp;
                        }

                        if (distance < minimalDistance) {
                            minimalDistance = distance;
                            alpha_[d]       = a;
                            gamma_[d]       = g;
                        }
                    }
                }

                for (i = 1; i < numberOfQuantiles_; i++) {
                    scaledQuantile                  = std::max(trainingQuantile_[i * dim() + d], currentQuantile_[i * dim() + d]) / maximalQuantile;
                    currentQuantile_[i * dim() + d] = maximalQuantile * (alpha_[d] * pow(scaledQuantile, gamma_[d]) + (1. - alpha_[d]) * scaledQuantile);
                }
            }

            if (combineNeighbors_) {
                for (u16 d = 0; d < dim(); d++) {
                    if ((s32)right_ == Type<s32>::max) {  // utterance wise: search whole grid
                        lowL  = 0.;
                        highL = 0.5;
                        lowR  = 0.;
                        highR = 0.5;
                    }
                    else {  // online: just update
                        lowL  = std::max(lambda_[d] - deltaLambda_, 0.);
                        highL = std::min(lambda_[d] + deltaLambda_, 0.5);
                        lowR  = std::max(rho_[d] - deltaRho_, 0.);
                        highR = std::min(rho_[d] + deltaRho_, 0.5);
                    }

                    minimalDistance = Type<f32>::max;

                    for (l = lowL; l <= highL; l += deltaLambda_) {
                        for (r = lowR; r <= highR; r += deltaRho_) {
                            distance = 0.;

                            for (i = 1; i < numberOfQuantiles_; i++) {
                                transformedQuantile = (1. - l - r) * currentQuantile_[i * dim() + d] + l * currentQuantile_[i * dim() + std::max((s32)d - 1, 0)] + r * currentQuantile_[i * dim() + std::min((s32)d + 1, (s32)dim() - 1)];
                                tmp                 = transformedQuantile - trainingQuantile_[i * dim() + d];
                                distance += tmp * tmp;
                            }

                            distance += (l * l + r * r) * beta_;

                            if (distance < minimalDistance) {
                                minimalDistance = distance;
                                lambda_[d]      = l;
                                rho_[d]         = r;
                            }
                        }
                    }
                }
            }
        }

        Frame in;
        for (u32 i = 0; i < slidingWindow_.size(); i++) {
            in = slidingWindow_[i];
            in.makePrivate();

            if (!estimateQuantiles_) {
                equalizeQuantiles(in);

                if (combineNeighbors_)
                    combineNeighbors(in);
            }

            slidingTransformedWindow.add(in);
        }
    }
    else {  // no quantile equalization

        for (u32 i = 0; i < slidingWindow_.size(); i++)
            slidingTransformedWindow.add(slidingWindow_[i]);
    }

    // calculate means
    for (u16 d = 0; d < dim(); d++) {
        f64 sum       = 0.0;
        f64 sumSquare = 0.0;

        for (u32 i = 0; i < slidingTransformedWindow.size(); i++) {
            if (normalizeMean_) {
                sum += (f64)(*slidingTransformedWindow[i])[d];

                if (normalizeVariance_)
                    sumSquare += (f64)(*slidingTransformedWindow[i])[d] * (f64)(*slidingTransformedWindow[i])[d];
            }
        }

        mean_[d] = (f32)(sum / slidingTransformedWindow.size());

        if (normalizeVariance_)
            dev_[d] = (f32)sqrt((sumSquare - sum * sum / slidingTransformedWindow.size()) / slidingTransformedWindow.size());
    }
}

void QuantileEqualization::applyTransformations(Frame& out) {
    require(out);

    updateTransformationParameters();

    if (estimateQuantiles_) {
        if (newIn_) {
            frameCounter_++;
        }
        else if (!wroteQuantiles_) {
            writeEstimatedQuantilesToFile();
            wroteQuantiles_ = true;
        }
    }

    if (equalizeQuantiles_)
        equalizeQuantiles(out);

    if (combineNeighbors_)
        combineNeighbors(out);

    if (normalizeMean_) {
        normalizeMean(out);

        if (normalizeVariance_)
            normalizeVariance(out);
    }

    changed_ = false;
    newIn_   = false;
}

ParameterBool   QuantileEqualizationNode::paramQuantileEqualization("quantiles", "use quantile equalization", true);
ParameterBool   QuantileEqualizationNode::paramCombineNeighbors("combination", "combine neighboring filter channels", false);
ParameterBool   QuantileEqualizationNode::paramQuantileEstimation("estimate", "estimate quantiles", false);
ParameterString QuantileEqualizationNode::paramQuantileFile("filename", "file with quantiles", "quantiles.txt");
ParameterBool   QuantileEqualizationNode::paramMeanNormalization("mean", "use mean normalization", true);
ParameterBool   QuantileEqualizationNode::paramVarianceNormalization("variance", "use also variance normalization", false);
ParameterInt    QuantileEqualizationNode::paramLength("length", "length of the sliding window in frames");
ParameterInt    QuantileEqualizationNode::paramRight("right", "output point");
ParameterInt    QuantileEqualizationNode::paramNumberOfQuantiles("numberOfQuantiles", "number of quantiles", 4);
ParameterFloat  QuantileEqualizationNode::paramOverestimationFactor("overestimationFactor", "overestimation factor for largest quantile", 1.0);
ParameterFloat  QuantileEqualizationNode::paramDeltaAlpha("deltaAlpha", "update alpha in range", 0.005);
ParameterFloat  QuantileEqualizationNode::paramDeltaGamma("deltaGamma", "update gamma in range", 0.01);
ParameterFloat  QuantileEqualizationNode::paramDeltaLambdaAndRho("deltaLambdaAndRho", "update lambda and rho in range", 0.005);
ParameterFloat  QuantileEqualizationNode::paramBeta("beta", "penalty factor for filter combination", 0.05);
ParameterBool   QuantileEqualizationNode::paramPoolQuantiles("poolQuantiles", "pool training quantiles for all components", true);
ParameterBool   QuantileEqualizationNode::paramPiecewiseLinear("piecewiseLinear", "apply piecewise linear transformation", false);
