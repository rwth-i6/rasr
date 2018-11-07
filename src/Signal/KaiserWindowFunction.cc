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
#include "KaiserWindowFunction.hh"
#include <Math/Nr/nr.h>

using namespace Signal;
using namespace Math::Nr;

bool KaiserWindowFunction::init() {

    if (window_.size() <= 1)
        return false;

    u32 M = window_.size() - 1;

    for (u32 n = 0; n <= M / 2; n ++) {

            window_[n] = window_[M - n] =
                bessi0(beta_ * sqrt(1.0 - ((f64)n / (M / 2.0) - 1.0) * ((f64)n / (M / 2.0) - 1.0))) /
                bessi0(beta_);
    }

    return WindowFunction::init();
}
