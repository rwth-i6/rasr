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

#include "StopWatch.hh"

namespace Core {

StopWatch::StopWatch()
        : running_(false), startTime_(), elapsedTime_(0.0) {}

void StopWatch::start() {
    if (running_) {
        return;
    }

    startTime_ = std::chrono::steady_clock::now();
    running_   = true;
}

void StopWatch::stop() {
    if (not running_) {
        return;
    }
    auto endTime = std::chrono::steady_clock::now();
    elapsedTime_ += std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime_).count();
    running_ = false;
}

void StopWatch::reset() {
    elapsedTime_ = 0;
    running_     = false;
}

double StopWatch::elapsedSeconds() const {
    return elapsedNanoseconds() / 1e9;
}

double StopWatch::elapsedCentiseconds() const {
    return elapsedNanoseconds() / 1e7;
}

double StopWatch::elapsedMilliseconds() const {
    return elapsedNanoseconds() / 1e6;
}

double StopWatch::elapsedMicroseconds() const {
    return elapsedNanoseconds() / 1e3;
}

double StopWatch::elapsedNanoseconds() const {
    if (running_) {
        auto currentTime = std::chrono::steady_clock::now();
        return elapsedTime_ + std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - startTime_).count();
    }
    return elapsedTime_;
}

}  // namespace Core
