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
#ifndef STOPWATCH_HH
#define STOPWATCH_HH

#include <chrono>

namespace Core {

/*
 * Simple timer class with start/stop functions that accumulates all the timed intervals
 * to a total.
 */
struct StopWatch {
public:
    StopWatch();

    /*
     * Stops timer if it is running and resets accumulated time to zero.
     */
    void reset();

    /*
     * Start timer. Does nothing if timer is already running.
     */
    void start();

    /*
     * End running timer and add duration to total. Does nothing if timer is not running.
     */
    void stop();

    /*
     * Getter functions to get the total elapsed time in different units. Includes the current interval
     * if the timer is running.
     */
    double elapsedSeconds() const;
    double elapsedCentiseconds() const;
    double elapsedMilliseconds() const;
    double elapsedMicroseconds() const;
    double elapsedNanoseconds() const;

private:
    bool                                  running_;
    std::chrono::steady_clock::time_point startTime_;
    double                                elapsedTime_;
};

}  // namespace Core

#endif  // TIMER_HH
