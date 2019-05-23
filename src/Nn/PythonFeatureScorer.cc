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
#include "PythonFeatureScorer.hh"
#include <Python/Numpy.hh>
#include <Python/Utilities.hh>

/*
 * PythonFeatureScorer uses a Python interface to get the scores.
 * The scores will we forwarded as-is. Negative log scores are expected, i.e. something like -log p(x|s).
 * So if you have a NN with posteriors, you need to divide by the prior on the Python side.
 *
 * For an RNN, esp a bidirectional one, we need to first get all the features before we can
 * forward it and get the scores. That is important for the design of this class.
 *
 * Also, this class supports buffering (isBuffered() == true), so both the
 * addFeature()/flush() but also the (traditionally non-buffered) getScorer()
 * has to be supported.
 * getScorer() is used e.g. in the aligner.
 * addFeature()/flush() is used by the recognizer.
 *
 * Python interface:
 *
 * PythonControl is used. See its documentation. In short:
 * init(sprint_unit='PythonFeatureScorer', ...) from the Python module will get called and is expected to return some object.
 * Member functions of this object will get called for the further communication.
 *
 * Member functions which will get called:
 *
 *   init(input_dim: int, output_dim: int)  # output-dim is the number of emission classes for us
 *   get_feature_buffer_size()  # expected to return -1 for now
 *   add_feature(feature: numpy.ndarray, time: int)  # feature is of shape (input_dim,)
 *   reset(num_frames: int)  # signals that we can flush any buffers
 *   compute(num_frames: int)  # all the features which we received so far should be evaluated
 *   get_scores(time: int)  # expected to return a numpy.ndarray of shape (output_dim,)
 *
 */

namespace Nn {

static const Core::ParameterInt paramFeatureDimension(
        "feature-dimension", "feature = input dimension");

static const Core::ParameterInt paramOutputDimension(
        "python-feature-scorer-output-dimension", "if set, will ignore the number of mixtures", -1);

PythonFeatureScorer::PythonFeatureScorer(const Core::Configuration& config, Core::Ref<const Mm::MixtureSet> mixtureSet)
        : Core::Component(config),
          Mm::FeatureScorer(config),
          featureBufferSize_(0),
          numFeaturesReceived_(0),
          currentFeature_(0),
          scoresComputed_(false),
          scoresCachePosition_(u32(-1)),
          nClasses_(mixtureSet->nMixtures()),
          inputDimension_(paramFeatureDimension(config)),
          batchIteration_(0),
          pythonControl_(config, /* sprintUnit */ "PythonFeatureScorer", /* isOptional */ false) {
    int outputDim = paramOutputDimension(config);
    if (outputDim >= 0) {
        log("PythonFeatureScorer: will ignore mixture-set number of classes %i but use %i instead", nClasses_, outputDim);
        nClasses_ = outputDim;
    }
    log("PythonFeatureScorer: initialize with feature dimension %i, number of classes %i", inputDimension_, nClasses_);
    require_gt(inputDimension_, 0);

    Python::ScopedGIL gil;
    pythonControl_.run_custom("init", "{s:i,s:i}", "input_dim", (int)inputDimension_, "output_dim", (int)nClasses_);
    {
        PyObject* res = pythonControl_.run_custom_with_result("get_feature_buffer_size", "{}");
        if (res) {
            long resLong = PyLong_AsLong(res);
            if ((resLong == -1) && PyErr_Occurred())
                pythonControl_.pythonCriticalError("PythonFeatureScorer: get_feature_buffer_size did not return an integer");
            featureBufferSize_ = (u32)resLong;
            if (featureBufferSize_ == 0)
                criticalError("PythonFeatureScorer: get_feature_buffer_size returned 0");
            Py_CLEAR(res);
        }
    }
}

PythonFeatureScorer::~PythonFeatureScorer() {
    pythonControl_.exit();
}

/**
 * Stores the current feature and the number of buffered features.
 * All computations are done in BatchFeatureScorer.
 * This class is used only because it is required by the
 * FeatureScorer interface.
 */
class PythonFeatureScorer::ContextScorer : public FeatureScorer::ContextScorer {
public:
    ContextScorer(const PythonFeatureScorer* parent, u32 currentFeature, u32 batchIteration)
            : parent_(parent),
              currentFeature_(currentFeature),
              batchIteration_(batchIteration) {}
    virtual ~ContextScorer() {}
    virtual Mm::EmissionIndex nEmissions() const {
        return parent_->nMixtures();
    }
    virtual Mm::Score score(Mm::EmissionIndex e) const {
        require_eq(batchIteration_, parent_->batchIteration_);
        return parent_->getScore(e, currentFeature_);
    }

private:
    const PythonFeatureScorer* parent_;
    u32                        currentFeature_;
    u32                        batchIteration_;
};

void PythonFeatureScorer::_addFeature(const Mm::FeatureVector& f) const {
    require(!bufferFilled());
    require(!scoresComputed_);
    if (inputDimension_ != f.size()) {
        criticalError("PythonFeatureScorer: was configured with input dimension %i but we got features with dimension %zu",
                      inputDimension_, f.size());
    }

    Python::ScopedGIL gil;
    PyObject*         numpyArray = NULL;
    if (!Python::stdVec2numpy(pythonControl_.getPythonCriticalErrorFunc(), numpyArray, f))
        return;
    pythonControl_.run_custom("add_feature", "{s:O,s:i}", "feature", numpyArray, "time", (int)numFeaturesReceived_);
    numFeaturesReceived_++;
}

void PythonFeatureScorer::addFeature(const Mm::FeatureVector& f) const {
    // Lazily call reset() when flush() went through all the buffer before.
    if (currentFeature_ > 0 && currentFeature_ >= numFeaturesReceived_)
        reset();
    _addFeature(f);
}

void PythonFeatureScorer::reset() const {
    pythonControl_.run_custom("reset", "{s:i}", "num_frames", (int)numFeaturesReceived_);
    numFeaturesReceived_ = 0;
    scoresComputed_      = false;
    scoresCache_.clear();
    scoresCachePosition_ = u32(-1);
    currentFeature_      = 0;
    batchIteration_++;
}

Mm::EmissionIndex PythonFeatureScorer::nMixtures() const {
    require_gt(nClasses_, 0);
    return nClasses_;
}

void PythonFeatureScorer::getFeatureDescription(Mm::FeatureDescription& description) const {
    require_gt(inputDimension_, 0);
    description.mainStream().setValue(Mm::FeatureDescription::nameDimension, inputDimension_);
}

// See comment in header. The scorer is not for `f`.
Mm::FeatureScorer::Scorer PythonFeatureScorer::getScorer(const Mm::FeatureVector& f) const {
    _addFeature(f);  // Don't reset() yet.
    return flush();
}

Mm::FeatureScorer::Scorer PythonFeatureScorer::flush() const {
    require_lt(currentFeature_, numFeaturesReceived_);
    Scorer scorer(new ContextScorer(this, currentFeature_, batchIteration_));
    currentFeature_++;
    // We must not call reset() here because the calls to getScore() will be delayed.
    return scorer;
}

Mm::Score PythonFeatureScorer::getScore(Mm::EmissionIndex e, u32 position) const {
    require_lt(position, numFeaturesReceived_);
    require_lt(e, nClasses_);
    // process buffer if needed
    if (!scoresComputed_) {
        pythonControl_.run_custom("compute", "{s:i}", "num_frames", (int)numFeaturesReceived_);
        // mark computed
        scoresComputed_ = true;
    }
    if (scoresCachePosition_ != position) {
        Python::ScopedGIL gil;
        PyObject*         res = pythonControl_.run_custom_with_result("get_scores", "{s:i}", "time", (int)position);
        if (!res)
            return 0.0;
        if (!Python::numpy2stdVec(pythonControl_.getPythonCriticalErrorFunc(), res, scoresCache_))
            return 0.0;
        Py_CLEAR(res);
        if (scoresCache_.size() != nClasses_) {
            criticalError("PythonFeatureScorer: get_scores returned vector of len %i but we expected len (num classes) %i",
                          (int)scoresCache_.size(), (int)nClasses_);
            return 0.0;
        }
        scoresCachePosition_ = position;
    }
    // return score in -log space
    return scoresCache_[e];
}

Mm::FeatureScorer::Scorer PythonFeatureScorer::getTimeIndexedScorer(u32 time) const {
    require_lt(time, numFeaturesReceived_);
    Scorer scorer(new ContextScorer(this, time, batchIteration_));
    return scorer;
}

// Must never be full. We want to support segments of any len, and we want to
// get all features in advance before we calculate the scores (to support bi-RNNs).
bool PythonFeatureScorer::bufferFilled() const {
    return numFeaturesReceived_ >= bufferSize();
}  // == cannot call addFeature() anymore
bool PythonFeatureScorer::bufferEmpty() const {
    return currentFeature_ >= numFeaturesReceived_;
}  // == cannot call flush() anymore
u32 PythonFeatureScorer::bufferSize() const {
    return featureBufferSize_;
}

}  // namespace Nn
