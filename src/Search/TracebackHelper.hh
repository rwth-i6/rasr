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

#include "Traceback.hh"

namespace Search {

inline Bliss::LemmaPronunciation* epsilonLemmaPronunciation() {
    return reinterpret_cast<Bliss::LemmaPronunciation*>(1);
}

inline Core::Ref<const Traceback> traceback(Core::Ref<LatticeTrace> end, Core::Ref<LatticeTrace> boundary) {
    Traceback* result = new Traceback();
    for (; end && end != boundary; end = end->predecessor) {
        result->push_back(*end);
    }
    std::reverse(result->begin(), result->end());
    return Core::ref(result);
}

/*
 * Used to track the longest common prefix of hypotheses during search (i.e. the stable prefix).
 *
 * It internally saves the most recent stable prefix and can advance it forward using a list of traces assuming the current
 * stable prefix is a common prefix of all the traces.
 */
class StableTraceTracker {
public:
    /*
     * Initializes the stable trace tracker with an empty stable prefix.
     */
    StableTraceTracker();

    /*
     * Initializes the stable trace tracker with the given initial trace.
     */
    StableTraceTracker(Core::Ref<LatticeTrace const> const& initialTrace);

    /*
     * Forcefully sets the stable trace tracker to the given trace.
     */
    void setTrace(Core::Ref<LatticeTrace const> const& trace);

    /*
     * Get currently stored stable trace.
     */
    Core::Ref<LatticeTrace const> getStablePrefixTrace() const;

    /*
     * Advances the stable trace as much as possible using the given list of traces.
     * Assumes that all the given traces contain the current stable trace somewhere as a predecessor.
     */
    void advanceStablePrefix(std::vector<Core::Ref<LatticeTrace const>> const& traces);

private:
    Core::Ref<LatticeTrace const> stablePrefixTrace_;
};

}  // namespace Search
