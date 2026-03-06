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

#include "TransitionTypes.hh"

namespace Nn {

const Core::Choice TransitionSet::choiceTransitionPreset(
        "none", TransitionPresetType::NONE,
        "all", TransitionPresetType::ALL,
        "ctc", TransitionPresetType::CTC,
        "transducer", TransitionPresetType::TRANSDUCER,
        "aed", TransitionPresetType::AED,
        "lm", TransitionPresetType::LM,
        Core::Choice::endMark());

const Core::ParameterChoice TransitionSet::paramTransitionPreset(
        "transition-preset",
        &TransitionSet::choiceTransitionPreset,
        "Preset for which transition types should be enabled.",
        TransitionPresetType::NONE);

const Core::ParameterStringVector TransitionSet::paramExtraTransitionTypes(
        "extra-transition-types",
        "Transition types that should be enabled in addition to the ones given by the preset.",
        ",");

TransitionSet::TransitionSet(Core::Configuration const& config, TransitionPresetType defaultPreset)
        : mask_(0) {
    enablePreset(static_cast<TransitionPresetType>(paramTransitionPreset(config, defaultPreset)));

    auto extraTransitionTypeStrings = paramExtraTransitionTypes(config);
    for (auto const& transitionTypeString : extraTransitionTypeStrings) {
        auto it = std::find_if(TransitionTypeArray.begin(),
                               TransitionTypeArray.end(),
                               [&](auto const& entry) { return entry.first == transitionTypeString; });
        require(it != TransitionTypeArray.end());  // Transition type name not found
        enable(it->second);
    }
}

TransitionSet::TransitionSet(TransitionSet const& other)
        : mask_(other.mask_) {}

TransitionSet::TransitionSet(Mask mask)
        : mask_(mask) {}

void TransitionSet::enable(TransitionType transitionType) {
    auto idx = static_cast<unsigned>(transitionType);
    verify(idx < TransitionType::numTypes);
    mask_ |= (Mask{1} << idx);
}

void TransitionSet::enablePreset(TransitionPresetType preset) {
    switch (preset) {
        case TransitionPresetType::NONE:
            break;
        case TransitionPresetType::ALL:
            for (auto const& [_, transitionType] : TransitionTypeArray) {
                enable(transitionType);
            }
            break;
        case TransitionPresetType::CTC:
            enable(LABEL_TO_LABEL);
            enable(LABEL_LOOP);
            enable(LABEL_TO_BLANK);
            enable(BLANK_TO_LABEL);
            enable(BLANK_LOOP);
            enable(INITIAL_LABEL);
            enable(INITIAL_BLANK);
            break;
        case TransitionPresetType::TRANSDUCER:
            enable(LABEL_TO_LABEL);
            enable(LABEL_TO_BLANK);
            enable(BLANK_TO_LABEL);
            enable(BLANK_LOOP);
            enable(INITIAL_LABEL);
            enable(INITIAL_BLANK);
            break;
        case TransitionPresetType::AED:
        case TransitionPresetType::LM:
            enable(LABEL_TO_LABEL);
            enable(BLANK_TO_LABEL);
            enable(INITIAL_LABEL);
            enable(SENTENCE_END);
            break;
    }
}

void TransitionSet::enableUnion(TransitionSet const& other) {
    mask_ |= other.mask_;
}

void TransitionSet::enableIntersection(TransitionSet const& other) {
    mask_ &= other.mask_;
}

bool TransitionSet::contains(TransitionType transitionType) const {
    return (mask_ & (Mask{1} << static_cast<unsigned>(transitionType)));
}

}  // namespace Nn
