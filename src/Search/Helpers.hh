/** Copyright 2025 RWTH Aachen University. All rights reserved.
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
#ifndef SEARCH_HELPERS_HH
#define SEARCH_HELPERS_HH

#include <algorithm>
#include <cmath>

#include <Core/Types.hh>
#include <Search/Types.hh>

namespace Search {

inline bool isBackwardRecognition(const Core::Configuration& config) {
    return config.getSelection().find(".backward") != std::string::npos;
}

inline f32 scaledLogAdd(f32 a, f32 b, f32 scale, f32 invertedScale) {
    if (b == Core::Type<f32>::max)
        return a;
    if (a == Core::Type<f32>::max)
        return b;
    a *= invertedScale;
    b *= invertedScale;
    return scale * (std::min(a, b) - ::log1p(::exp(std::min(a, b) - std::max(a, b))));
}

inline bool approximatelyEqual(double a, double b, const double threshold = 0.001) {
    double diff = a - b;
    return diff > -threshold && diff < threshold;
}

/*
 * Penalty charged on every BLANK_EXIT so that among the otherwise identically scored segmentations of
 * a sequence of blank frames, the one with the fewest blank exits wins the recombination.
 *
 * It scales with the score because `Score` is a 32-bit float: a constant epsilon would round away once
 * the accumulated score grows. At ~8-16 ULP it always changes the score and never outweighs a real one.
 */
inline Score blankExitPenalty(Score score) {
    static constexpr Score relativeEpsilon = 8 * Core::Type<Score>::epsilon;
    // The lower bound on the magnitude keeps the penalty well-defined for a score of (close to) zero.
    return relativeEpsilon * std::max(std::abs(score), static_cast<Score>(1.0));
}

}  // namespace Search

#endif  // SEARCH_HELPERS_HH
