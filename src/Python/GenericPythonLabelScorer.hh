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

#ifndef GENERIC_PYTHON_LABEL_SCORER_HH
#define GENERIC_PYTHON_LABEL_SCORER_HH

#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Core/FIFOCache.hh>
#include <Core/ReferenceCounting.hh>
#include <Nn/LabelScorer/BufferedLabelScorer.hh>
#include <Nn/LabelScorer/ScoringContext.hh>
#include <Speech/Feature.hh>
#include <optional>
#undef ensure  // macro duplication in pybind11/numpy.h
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace Python {

/*
 * Label Scorer that performs scoring by forwarding the input feature at the current timestep together
 * with a fixed-size sequence of history tokens through an ONNX model
 */
class GenericPythonLabelScorer : public Nn::BufferedLabelScorer {
    using Precursor = BufferedLabelScorer;

    static const Core::ParameterString paramInitScoringContextCallbackName;
    static const Core::ParameterString paramExtendScoringContextCallbackName;
    static const Core::ParameterString paramScoreCallbackName;
    static const Core::ParameterString paramFinishCheckCallbackName;
    static const Core::ParameterBool   paramBlankUpdatesHistory;
    static const Core::ParameterBool   paramLoopUpdatesHistory;
    static const Core::ParameterInt    paramMaxCachedScores;

public:
    GenericPythonLabelScorer(Core::Configuration const& config);
    virtual ~GenericPythonLabelScorer() = default;

    // Clear feature buffer and cached scores
    void reset() override;

    // Initial scoring context contains step 0 and a history vector filled with the start label index
    Nn::ScoringContextRef getInitialScoringContext() override;

    // May increment the step by 1 (except for vertical transitions) and may append the next token to the
    // history label sequence depending on the transition type and whether loops/blanks update the history
    // or not
    Nn::ScoringContextRef extendedScoringContext(LabelScorer::Request const& request) override;

    // If scores for the given scoring contexts are not yet cached, prepare and run an ONNX session to
    // compute the scores and cache them
    // Then, retreive scores from cache
    std::optional<LabelScorer::ScoresWithTimes> computeScoresWithTimes(std::vector<LabelScorer::Request> const& requests) override;

    // Uses `getScoresWithTimes` internally with some wrapping for vector packing/expansion
    std::optional<LabelScorer::ScoreWithTime> computeScoreWithTime(LabelScorer::Request const& request) override;

    virtual void registerPythonCallback(std::string const& name, py::function const& callback) override;

protected:
    Speech::TimeframeIndex minActiveTimeIndex(Core::CollapsedVector<Nn::ScoringContextRef> const& activeContexts) const override;

private:
    // Forward a batch of histories through the ONNX model and put the resulting scores into the score cache
    // Assumes that all histories in the batch are based on the same timestep
    void forwardBatch(std::vector<Nn::PythonScoringContextRef> const& contextBatch);

    py::object computeInitialState();

    void setupEncoderStatesValue();

    std::string initScoringContextCallbackName_;
    std::string extendScoringContextCallbackName_;
    std::string scoreCallbackName_;
    std::string finishCheckCallbackName_;

    py::function initScoringContextCallback_;
    py::function extendScoringContextCallback_;
    py::function scoreCallback_;
    py::function finishCheckCallback_;

    bool   blankUpdatesHistory_;
    bool   loopUpdatesHistory_;
    size_t maxBatchSize_;

    py::array_t<f32> encoderStates_;
    py::object       initialState_;

    Core::FIFOCache<Nn::PythonScoringContextRef, std::vector<Score>, Nn::ScoringContextHash, Nn::ScoringContextEq> scoreCache_;
};

}  // namespace Python

#endif  // GENERIC_ONNX_LABEL_SCORER_HH
