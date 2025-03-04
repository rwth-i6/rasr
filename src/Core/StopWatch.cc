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
#include "Utility.hh"

namespace Core {

StopWatch::StopWatch()
        : running_(false), startTime_(), elapsedSeconds_(0.0) {}

void StopWatch::start() {
    if (running_) {
        return;
    }

    TIMER_START(startTime_);
    running_ = true;
}

void StopWatch::stop() {
    if (not running_) {
        return;
    }

    timeval endTime;
    double  diff = 0;  // in seconds
    TIMER_STOP(startTime_, endTime, diff);

    elapsedSeconds_ += diff;
    running_ = false;
}

void StopWatch::reset() {
    elapsedSeconds_ = 0;
    running_        = false;
}

double StopWatch::elapsedSeconds() {
    if (running_) {
        timeval endTime;
        double  diff = 0;  // in seconds
        TIMER_STOP(startTime_, endTime, diff);
    }
    return elapsedSeconds_;
}

double StopWatch::elapsedCentiseconds() {
    return elapsedSeconds() * 1e2;
}

double StopWatch::elapsedMilliseconds() {
    return elapsedSeconds() * 1e3;
}

double StopWatch::elapsedMicroseconds() {
    return elapsedSeconds() * 1e6;
}

double StopWatch::elapsedNanoseconds() {
    return elapsedSeconds() * 1e9;
}

}  // namespace Core
