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

#include <Core/Types.hh>

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

}  // namespace Search

#endif  // SEARCH_HELPERS_HH
