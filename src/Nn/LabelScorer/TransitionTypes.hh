/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef TRANSITION_TYPES_HH
#define TRANSITION_TYPES_HH

#include <Core/Component.hh>
#include "Core/Configuration.hh"

namespace Nn {

// When updating these, remember to also update the corresponding Python binding in src/Tools/LibRASR/LabelScorer.cc
enum TransitionType {
    LABEL_TO_LABEL,
    LABEL_LOOP,
    LABEL_TO_BLANK,
    BLANK_TO_LABEL,
    BLANK_LOOP,
    INITIAL_LABEL,
    INITIAL_BLANK,
    WORD_EXIT,
    NONWORD_EXIT,
    SILENCE_EXIT,
    SENTENCE_END,
    numTypes,  // must remain at the end
};

inline constexpr auto TransitionTypeArray = std::to_array<std::pair<std::string_view, TransitionType>>({
        {"label-to-label", LABEL_TO_LABEL},
        {"label-loop", LABEL_LOOP},
        {"label-to-blank", LABEL_TO_BLANK},
        {"blank-to-label", BLANK_TO_LABEL},
        {"blank-loop", BLANK_LOOP},
        {"initial-label", INITIAL_LABEL},
        {"initial-blank", INITIAL_BLANK},
        {"word-exit", WORD_EXIT},
        {"nonword-exit", NONWORD_EXIT},
        {"silence-exit", SILENCE_EXIT},
        {"sentence-end", SENTENCE_END},
});
static_assert(TransitionTypeArray.size() == TransitionType::numTypes, "TransitionTypeArray size must match number of TransitionType values");

enum TransitionPresetType {
    NONE,
    ALL,
    CTC,
    TRANSDUCER,
    AED,
    LM,
};

/*
 * Class representing a set of transition types with fast membership checks
 * via bit-masks.
 * Initialized by taking a preset either from config or from default and optionally
 * adding extra transition types from config.
 */
class TransitionSet {
public:
    static const Core::Choice          choiceTransitionPreset;
    static const Core::ParameterChoice paramTransitionPreset;

    static const Core::ParameterStringVector paramExtraTransitionTypes;

    TransitionSet(Core::Configuration const& config, TransitionPresetType defaultPreset);
    TransitionSet(TransitionSet const& other);

    void enable(TransitionType transitionType);
    void enablePreset(TransitionPresetType preset);
    void enableUnion(TransitionSet const& other);
    void enableIntersection(TransitionSet const& other);

    bool contains(TransitionType transitionType) const;

private:
    using Mask = uint32_t;  // Enough to contain <= 32 transition types

    TransitionSet(Mask mask);

    Mask mask_;
};

}  // namespace Nn

#endif  // TRANSITION_TYPES_HH
