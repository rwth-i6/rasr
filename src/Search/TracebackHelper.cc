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

#include "TracebackHelper.hh"

namespace Search {

StableTraceTracker::StableTraceTracker() : stablePrefixTrace_() {}

StableTraceTracker::StableTraceTracker(Core::Ref<LatticeTrace const> const& initialTrace) : stablePrefixTrace_(initialTrace) {}

void StableTraceTracker::setTrace(Core::Ref<LatticeTrace const> const& trace) {
    stablePrefixTrace_ = trace;
}

Core::Ref<LatticeTrace const> StableTraceTracker::getStablePrefixTrace() const {
    return stablePrefixTrace_;
}

void StableTraceTracker::advanceStablePrefix(std::vector<Core::Ref<LatticeTrace const>> const& extendedTraces) {
    // Extend stable prefix one-by-one until it's no longer possible
    while (true) {
        // Successor of `stablePrefixTrace_` in `extendedTraces`.
        // If this is unique, `stablePrefixTrace_` can be updated to this, otherwise some elements of `extendedTraces`
        // disagree after `stablePrefixTrace_` and we can advance no more.
        Core::Ref<LatticeTrace const> candidateNext;

        for (auto const& trace : extendedTraces) {
            if (trace == stablePrefixTrace_) {
                // `trace` itself is contained in `extendedTraces` and thus doesn't have a successor candidate
                return;
            }

            // Go backwards from `trace` until until `curr` is the successor of `stablePrefixTrace_`
            auto curr = trace;
            while (curr->predecessor and curr->predecessor != stablePrefixTrace_) {
                curr = curr->predecessor;
            }

            // If this fails then `stablePrefixTrace_` is not along the predecessors of `trace` which violates the precondition.
            require(curr->predecessor == stablePrefixTrace_);

            if (not candidateNext) {
                candidateNext = curr;
            }
            else if (candidateNext != curr) {
                // stable prefix can not be extended because there are multiple possible successors
                return;
            }
        }

        if (not candidateNext) {
            return;
        }

        // All hyps agree on `candidateNext` trace so the stable prefix can be extended there and we repeat the process
        stablePrefixTrace_ = candidateNext;
    }
}

}  // namespace Search
