#include "FactoredHybridFeatureScorer.hh"

namespace Tensorflow {

    const Core::Choice TFFactoredHybridFeatureScorer::choiceContextType(
            "monophone", TFFactoredHybridFeatureScorer::ContextTypeMonophone,
            "monophone-delta", TFFactoredHybridFeatureScorer::ContextTypeMonophoneDelta,
            "diphone", TFFactoredHybridFeatureScorer::ContextTypeDiphone,
            "diphone-delta", TFFactoredHybridFeatureScorer::ContextTypeDiphoneDelta,
            "triphone-forward", TFFactoredHybridFeatureScorer::ContextTypeTriphoneForward,
            "triphone-forward-delta", TFFactoredHybridFeatureScorer::ContextTypeTriphoneForwardDelta,
            "triphone-symmetric", TFFactoredHybridFeatureScorer::ContextTypeTriphoneSymmetric,
            "triphone-backward", TFFactoredHybridFeatureScorer::ContextTypeTriphoneBackward,
            Core::Choice::endMark());
    const Core::Choice TFFactoredHybridFeatureScorer::choiceTransitionType(
            "default", TFFactoredHybridFeatureScorer::TransitionTypeDefault, // dim 2
            "constant", TFFactoredHybridFeatureScorer::TransitionTypeConstant, // dim n_center_states
            "feature", TFFactoredHybridFeatureScorer::TransitionTypeFeature, // dim 2
            "center", TFFactoredHybridFeatureScorer::TransitionTypeCenter, // dim n_center_states
            "feature-center", TFFactoredHybridFeatureScorer::TransitionTypeFeatureCenter, // dim n_center_states
            "feature-center-left", TFFactoredHybridFeatureScorer::TransitionTypeFeatureCenterLeft, // dim n_center_states*n_contexts
            "feature-center-right", TFFactoredHybridFeatureScorer::TransitionTypeFeatureCenterRight, // dim n_center_states*n_contexts
            Core::Choice::endMark());

    const Core::ParameterString TFFactoredHybridFeatureScorer::paramContextType(
            "context-type",
            "type of context model for the label posterior, check the choices above",
            "monophone");

    const Core::ParameterString TFFactoredHybridFeatureScorer::paramTransitionType(
            "transition-type",
            "type of the transition model, check the choices above",
            "default");

    const Core::ParameterInt TFFactoredHybridFeatureScorer::paramNumStatesPerPhone(
            "num-states-per-phone",
            "number of states per each phoneme",
            3);

    const Core::ParameterInt TFFactoredHybridFeatureScorer::paramNumContexts(
            "num-label-contexts",
            "number of contexts including boundary");

    const Core::ParameterInt TFFactoredHybridFeatureScorer::paramSilenceId(
            "silence-id",
            "the silence id in the phoneme inventory derived from the lexicon",
            3);

    const Core::ParameterInt TFFactoredHybridFeatureScorer::paramLenEncoderOutput(
            "num-encoder-output",
            "length of encoder output feature on time axis");

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramRightContextScale(
            "right-context-scale",
            "scaling of the right context score",
            1.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramCenterStateScale(
            "center-state-scale",
            "scaling of the center state score",
            1.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramLeftContextScale(
            "left-context-scale",
            "scaling of the left context score",
            1.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramRightContextPriorScale(
            "right-context-prior-scale",
            "scaling of the right context prior",
            1.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramCenterStatePriorScale(
            "center-state-prior-scale",
            "scaling of the center state prior",
            1.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramLeftContextPriorScale(
            "left-context-prior-scale",
            "scaling of the left context prior",
            1.0);

    const Core::ParameterString TFFactoredHybridFeatureScorer::paramLeftContextPriorFileName(
            "left-context-prior-file",
            "prior file path for the left context.");

    const Core::ParameterString TFFactoredHybridFeatureScorer::paramCenterStatePriorFileName(
            "center-state-prior-file",
            "prior file path for the center state.");

    const Core::ParameterString TFFactoredHybridFeatureScorer::paramRightContextPriorFileName(
            "right-context-prior-file",
            "prior file path for the right context.");

    const Core::ParameterString TFFactoredHybridFeatureScorer::paramConstantForwardProbFileName(
            "constant-forward-prob-file",
            "pre-estimated probabilities for forward");

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramLoopScale(
            "loop-scale",
            "scaling of the logarithmized loop probability",
            1.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramForwardScale(
            "forward-scale",
            "scaling of the logarithmized forward probability",
            1.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramSilLoopPenalty(
            "silence-loop-penalty",
            "silence loop penalty for only input dependent delta model",
            0.0);

    const Core::ParameterFloat TFFactoredHybridFeatureScorer::paramSilForwardPenalty(
            "silence-forward-penalty",
            "silence forward penalty for only input dependent delta model",
            0.0);

    const Core::ParameterBool TFFactoredHybridFeatureScorer::paramMinDuration(
            "is-min-duration",
            "set true when the center phoneme three states have the same label",
            false);

    const Core::ParameterBool TFFactoredHybridFeatureScorer::paramUseWordEndClasses(
            "use-word-end-classes",
            "set true when the center state is distinguished in additional two classes {None, @i} and, {@f, @i@f}",
            false);

    const Core::ParameterBool TFFactoredHybridFeatureScorer::paramUseBoundaryClasses(
            "use-boundary-classes",
            "set true when for monophone model the three outputs are combined",
            false);

    const Core::ParameterBool TFFactoredHybridFeatureScorer::paramMultiEncoderOutput(
            "is-multi-encoder-output",
            "set true when you have more than one encoder output used",
            false);


    class TFFactoredHybridFeatureScorer;

    //////////////////////////////////////////////////////////
    // General purpose and initialization
    //////////////////////////////////////////////////////////

    u32 TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::getCacheLength() const {
        if (parentScorer_->contextType_ == ContextTypeMonophone or parentScorer_->contextType_ == ContextTypeMonophoneDelta) {
            return parentScorer_->nCenterStates();
        }
        else if (parentScorer_->contextType_ == ContextTypeDiphone or parentScorer_->contextType_ == ContextTypeDiphoneDelta) {
            return parentScorer_->nCenterStates() * parentScorer_->nContextLabels();
        }

        return parentScorer_->nCenterStates() * parentScorer_->nContextLabels() * parentScorer_->nContextLabels();

    }

    void TFFactoredHybridFeatureScorer::getFeatureDescription(Mm::FeatureDescription& description) const {
        auto featureSize = lenEncoderOutput_;
        if (isMultiEncoderOutput_){
            //for now let us assume we have the same size for the delta-specific encoder output
            featureSize *= 2;
        }
        description.mainStream().setValue(Mm::FeatureDescription::nameDimension, featureSize);
    }

    void TFFactoredHybridFeatureScorer::setContextType() {
        Core::Choice::Value contextChoiceId = choiceContextType[TFFactoredHybridFeatureScorer::paramContextType_];
        if (contextChoiceId == Core::Choice::IllegalValue)
            criticalError("Unknown context type \"%s\"", TFFactoredHybridFeatureScorer::paramContextType_.c_str());
        contextType_ = ContextType(contextChoiceId);
    }

    void TFFactoredHybridFeatureScorer::setTransitionType() {
        Core::Choice::Value transitionChoiceId = choiceTransitionType[TFFactoredHybridFeatureScorer::paramTransitionType_];
        if (transitionChoiceId == Core::Choice::IllegalValue)
            criticalError("Unknown transition type \"%s\"", TFFactoredHybridFeatureScorer::paramTransitionType_.c_str());
        transitionType_ = TransitionType(transitionChoiceId);
    }

    void TFFactoredHybridFeatureScorer::setTransitionCaches() {

        u32 cacheLength;
        switch (transitionType_){
            case TransitionTypeConstant: {
                Math::Vector <f32> forwardProbs = readPriorVector(constantForwardProbFileName_);
                cacheLength = forwardProbs.size();
                for (auto i = 0u; i < cacheLength; i++) {
                    forwardScores_.push_back(forwardScale_ * Core::log(forwardProbs[i]));
                    loopScores_.push_back(loopScale_ * Core::log(1 - forwardProbs[i]));
                }
                return;
            }
            case TransitionTypeFeature: {
                cacheLength = 2;
            }
                break;
            case TransitionTypeCenter:
            case TransitionTypeFeatureCenter: {
                cacheLength = nCenterStates();
                break;
            }
            case TransitionTypeFeatureCenterLeft:
            case TransitionTypeFeatureCenterRight: {
                cacheLength = nCenterStates() * nContextLabels();
                break;
            }
            default:
                cacheLength = 2;
                break;
                //other cases decide for the length and then before leaving just initialize the two caches.
        }
        //This is in case we did not initialize them with constant given values
        for (auto i = 0u; i < cacheLength; i++){
            forwardScores_.push_back(0);
            loopScores_.push_back(0);
        }

        return;

    }

    Math::Vector<f32> TFFactoredHybridFeatureScorer::readPriorVector(std::string fileName) {
        // Vector Prior is only for the context-independent outputs. This is for all models the
        // left context, except for triphone backward which is the center state
        Math::Vector<f32> prior;
        u32               vectorDim;

        if (!Math::Module::instance().formats().read(fileName, prior)) {
            std::cerr << "no file for the context label priors is provided, they are set uniformly" << std::endl;
        }
        else {
            log() << "Vector priors set succesfully.";
        }
        return prior;
    }

    Math::Matrix<f32> TFFactoredHybridFeatureScorer::readPriorMatrix(std::string fileName) {
        // All context dependent priors are saved as an matrix. For context one we have for p(a|b)
        // a matrix of B rows and A columns. In case of p(a| b,c) this is C*B rows and again A columns
        Math::Matrix<f32> prior;
        if (!Math::Module::instance().formats().read(fileName, prior)) {
            std::cerr << "no file for the diphone label priors is provided, they are set uniformly" << std::endl;
        }
        else {
            log() << "Matrix priors set succesfully.";
        }

        return prior;
    }

    void TFFactoredHybridFeatureScorer::setPriors() {
        switch (contextType_) {
            case ContextTypeMonophone:
            case ContextTypeMonophoneDelta: {
                centerStatePriors_  = readPriorVector(centerStatePriorFileName_);
                break;
            }
            case ContextTypeDiphone:
            case ContextTypeDiphoneDelta: {
                contextDependentCenterStatePriors_ = readPriorMatrix(centerStatePriorFileName_);
                leftContextPriors_                 = readPriorVector(leftContextPriorFileName_);
                break;
            }
            case ContextTypeTriphoneForward:
            case ContextTypeTriphoneForwardDelta: {
                contextDependentRightContextPriors_ = readPriorMatrix(rightContextPriorFileName_);
                contextDependentCenterStatePriors_  = readPriorMatrix(centerStatePriorFileName_);
                leftContextPriors_                  = readPriorVector(leftContextPriorFileName_);
                break;
            }
            case ContextTypeTriphoneSymmetric: {
                rightContextPriors_                = readPriorVector(rightContextPriorFileName_);
                contextDependentCenterStatePriors_ = readPriorMatrix(centerStatePriorFileName_);
                leftContextPriors_                 = readPriorVector(leftContextPriorFileName_);
                break;
            }
            case ContextTypeTriphoneBackward: {
                contextDependentRightContextPriors_ = readPriorMatrix(rightContextPriorFileName_);
                centerStatePriors_                  = readPriorVector(centerStatePriorFileName_);
                contextDependentLeftContextPriors_  = readPriorMatrix(leftContextPriorFileName_);
                break;
            }
            default:
                defect();
        }
    }

    void TFFactoredHybridFeatureScorer::setTensorNames() {
        //all outputs are ordered by the order of context-dependency, from triphone to monophone
        //the target defines the rasr tensor tag
        switch (contextType_) {
            case ContextTypeMonophone: {
                auto const& featureInfo   = inputMap_.get_info("encoder-output");
                inputsTensorNames_.push_back(featureInfo.tensor_name());

                if (isMultiEncoderOutput_){
                    auto const& deltaFeatureInfo = inputMap_.get_info("deltaEncoder-output");
                    inputsTensorNames_.push_back(deltaFeatureInfo.tensor_name());
                }

                auto const& centerOutputInfo = outputMap_.get_info("center-state-posteriors");
                outputTensorNames_.push_back(centerOutputInfo.tensor_name());
                break;
            }
            case ContextTypeMonophoneDelta: {
                auto const& featureInfo   = inputMap_.get_info("encoder-output");
                inputsTensorNames_.push_back(featureInfo.tensor_name());
                if (isMultiEncoderOutput_){
                    auto const& deltaFeatureInfo = inputMap_.get_info("deltaEncoder-output");
                    inputsTensorNames_.push_back(deltaFeatureInfo.tensor_name());
                }
                auto const& centerOutputInfo = outputMap_.get_info("center-state-posteriors");
                outputTensorNames_.push_back(centerOutputInfo.tensor_name());
                auto const& deltaOutputInfo   = outputMap_.get_info("delta-posteriors");
                outputTensorNames_.push_back(deltaOutputInfo.tensor_name());
                break;
            }

            case ContextTypeDiphone: {
                auto const& featureInfo   = inputMap_.get_info("encoder-output");
                auto const& denseLabelInfo = inputMap_.get_info("dense-classes");
                inputsTensorNames_.push_back(featureInfo.tensor_name());
                inputsTensorNames_.push_back(denseLabelInfo.tensor_name());

                auto const&  centerOutputInfo = outputMap_.get_info("center-state-posteriors");
                auto const& contextOutputInfo = outputMap_.get_info("left-context-posteriors");
                outputTensorNames_.push_back(centerOutputInfo.tensor_name());
                outputTensorNames_.push_back(contextOutputInfo.tensor_name());
                break;
            }
            case ContextTypeDiphoneDelta: {
                auto const& featureInfo   = inputMap_.get_info("encoder-output");
                auto const& denseLabelInfo = inputMap_.get_info("dense-classes");
                inputsTensorNames_.push_back(featureInfo.tensor_name());
                inputsTensorNames_.push_back(denseLabelInfo.tensor_name());
                if (isMultiEncoderOutput_){
                    auto const& deltaFeatureInfo = inputMap_.get_info("deltaEncoder-output");
                    inputsTensorNames_.push_back(deltaFeatureInfo.tensor_name());
                }
                auto const&  centerOutputInfo = outputMap_.get_info("center-state-posteriors");
                auto const& contextOutputInfo = outputMap_.get_info("left-context-posteriors");
                auto const& deltaOutputInfo   = outputMap_.get_info("delta-posteriors");
                outputTensorNames_.push_back(centerOutputInfo.tensor_name());
                outputTensorNames_.push_back(contextOutputInfo.tensor_name());
                outputTensorNames_.push_back(deltaOutputInfo.tensor_name());
                break;
            }

            case ContextTypeTriphoneForward: {
                auto const& featureInfo      = inputMap_.get_info("encoder-output");
                auto const& denseLabelInfo = inputMap_.get_info("dense-classes");
                inputsTensorNames_.push_back(featureInfo.tensor_name());
                inputsTensorNames_.push_back(denseLabelInfo.tensor_name());
                if (isMultiEncoderOutput_){
                    auto const& deltaFeatureInfo = inputMap_.get_info("deltaEncoder-output");
                    inputsTensorNames_.push_back(deltaFeatureInfo.tensor_name());
                }

                auto const& rightContextOutputInfo = outputMap_.get_info("right-context-posteriors");
                auto const& centerOutputInfo       = outputMap_.get_info("center-state-posteriors");
                auto const& leftContextOutputInfo  = outputMap_.get_info("left-context-posteriors");
                outputTensorNames_.push_back(rightContextOutputInfo.tensor_name());
                outputTensorNames_.push_back(centerOutputInfo.tensor_name());
                outputTensorNames_.push_back(leftContextOutputInfo.tensor_name());
                break;
            }
            case ContextTypeTriphoneForwardDelta: {
                auto const& featureInfo      = inputMap_.get_info("encoder-output");
                auto const& denseLabelInfo   = inputMap_.get_info("dense-classes");
                inputsTensorNames_.push_back(featureInfo.tensor_name());
                inputsTensorNames_.push_back(denseLabelInfo.tensor_name());

                if (isMultiEncoderOutput_){
                    auto const& deltaFeatureInfo = inputMap_.get_info("deltaEncoder-output");
                    inputsTensorNames_.push_back(deltaFeatureInfo.tensor_name());
                }


                auto const& rightContextOutputInfo = outputMap_.get_info("right-context-posteriors");
                auto const& centerOutputInfo       = outputMap_.get_info("center-state-posteriors");
                auto const& leftContextOutputInfo  = outputMap_.get_info("left-context-posteriors");
                auto const& deltaOutputInfo    = outputMap_.get_info("delta-posteriors");
                outputTensorNames_.push_back(rightContextOutputInfo.tensor_name());
                outputTensorNames_.push_back(centerOutputInfo.tensor_name());
                outputTensorNames_.push_back(leftContextOutputInfo.tensor_name());
                outputTensorNames_.push_back(deltaOutputInfo.tensor_name());
                break;
            }
            case ContextTypeTriphoneSymmetric: {
                auto const& featureInfo   = inputMap_.get_info("encoder-output");
                auto const& denseLabelInfo = inputMap_.get_info("dense-classes");
                inputsTensorNames_.push_back(featureInfo.tensor_name());
                inputsTensorNames_.push_back(denseLabelInfo.tensor_name());
                //Todo: update the names
                auto const& triphoneOutputInfo = outputMap_.get_info("center-state-posteriors");
                auto const& pastOutputInfo     = outputMap_.get_info("left-context-posteriors");
                auto const& futureOutputInfo   = outputMap_.get_info("right-context-posteriors");
                outputTensorNames_.push_back(triphoneOutputInfo.tensor_name());
                outputTensorNames_.push_back(pastOutputInfo.tensor_name());
                outputTensorNames_.push_back(futureOutputInfo.tensor_name());
                break;
            }
            case ContextTypeTriphoneBackward: {
                auto const& featureInfo   = inputMap_.get_info("encoder-output");
                auto const& denseLabelInfo = inputMap_.get_info("dense-classes");
                inputsTensorNames_.push_back(featureInfo.tensor_name());
                inputsTensorNames_.push_back(denseLabelInfo.tensor_name());
                //Todo: update the names
                auto const& triphoneOutputInfo     = outputMap_.get_info("left-context-posteriors");
                auto const& diphoneOutputInfo      = outputMap_.get_info("right-context-posteriors");
                auto const& currentStateOutputInfo = outputMap_.get_info("center-state-posteriors");
                outputTensorNames_.push_back(triphoneOutputInfo.tensor_name());
                outputTensorNames_.push_back(diphoneOutputInfo.tensor_name());
                outputTensorNames_.push_back(currentStateOutputInfo.tensor_name());
                break;
            }

            default:
                defect();
        }
    }
    bool TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::isTriphone() const {
        return !(parentScorer_->contextType_ == ContextTypeDiphone or
                 parentScorer_->contextType_ == ContextTypeDiphoneDelta or
                 parentScorer_->contextType_ == ContextTypeMonophone or
                 parentScorer_->contextType_ == ContextTypeMonophoneDelta);
    }

    bool TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::isDelta() const {
        return ( parentScorer_->contextType_ == ContextTypeMonophoneDelta or
                parentScorer_->contextType_ == ContextTypeDiphoneDelta or
                parentScorer_->contextType_ == ContextTypeTriphoneForwardDelta);
    }

    TFFactoredHybridFeatureScorer::ModelIndex TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::getDeltaIndex(Mm::EmissionIndex stateId) const {

        if (parentScorer_->contextType_ == ContextTypeTriphoneForward or parentScorer_->contextType_ == ContextTypeTriphoneForwardDelta){
            return stateId;
        }

         if (parentScorer_->contextType_ == ContextTypeMonophoneDelta){
             std::vector<u32> labels = getLabelIndices(stateId);
             ModelIndex centerPhonemeState = labels[1];

             switch (parentScorer_->transitionType_){
                 case TransitionTypeConstant:
                 case TransitionTypeCenter:
                 case TransitionTypeFeatureCenter:{
                     return centerPhonemeState;
                 }
                    break;
                 case TransitionTypeFeature: {
                     if (centerPhonemeState == parentScorer_->getSilenceLabelId()){
                         return 1;
                     }
                     else return 0;
                 }
                     break;

                 case TransitionTypeFeatureCenterLeft:
                 case TransitionTypeFeatureCenterRight:
                     //ToDo: decide about index calculation
                     defect();
             }

         }

        if (parentScorer_->contextType_ == ContextTypeDiphoneDelta){
            std::vector<u32> labels       = getLabelIndices(stateId);
            Mm::EmissionIndex centerState = labels[1];
            return centerState;
        }

    }

    std::vector<u32> TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::getLabelIndices(Mm::EmissionIndex e) const {

        u32 nLabels = parentScorer_->nContextLabels();
        std::vector<u32> labels;


        ModelIndex result = e;

        ModelIndex rightPhoneme = result % nLabels;
        result = (result - rightPhoneme)/ nLabels;
        ModelIndex leftPhoneme = result % nLabels;
        ModelIndex centerPhonemeState = (result - leftPhoneme) / nLabels;

        //ToDo: decide about minimm duration


        labels.emplace_back(leftPhoneme);
        labels.emplace_back(centerPhonemeState);
        labels.emplace_back(rightPhoneme);


        return labels;
    }

    Mm::EmissionIndex TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::mapLabelSetToDense(ModelIndex left,
                                                                                                       ModelIndex center,
                                                                                                       ModelIndex right) const {
        ModelIndex nLabels = parentScorer_->nContextLabels();
        return (((center * nLabels) + left) * nLabels) + right;

    }

    TFFactoredHybridFeatureScorer::ModelIndex TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::mapPhonemeIdToContextId(Bliss::Phoneme::Id phonemeId) const {
         //you might have a state-tying that has different indices with respect to your modeling approach. If you need to remap the label set
         //for example merging noise phonemes or not using silence.{1,2}
        return phonemeId;

    }




    TFFactoredHybridFeatureScorer::ModelIndex TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::calculateCacheIndex(ModelIndex center,
                                                                                                                                ModelIndex left,
                                                                                                                                ModelIndex right) const {
        switch (parentScorer_->contextType_) {
            case ContextTypeMonophone:
            case ContextTypeMonophoneDelta: {
                return center;
            }
            case ContextTypeDiphone:
            case ContextTypeDiphoneDelta: {
                return ((center * parentScorer_->nContextLabels()) + left);
            }
            case ContextTypeTriphoneForward:
            case ContextTypeTriphoneForwardDelta:
            case ContextTypeTriphoneSymmetric:
            case ContextTypeTriphoneBackward:{
                return ((parentScorer_->nCenterStates() * parentScorer_->nContextLabels() * left) + (parentScorer_->nContextLabels() * center) + right);
            }
        }
    }

    ///////////////////////////////////////////////////////
    // Scoring with fixed or absent tdps
    //////////////////////////////////////////////////////

    void TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::setMonophoneScores() const {

        std::vector<std::pair<std::string, Tensor>> inputs;
        std::vector<Math::FastMatrix<f32>> encoderOutput;
        Math::FastMatrix<f32> f(1, parentScorer_->lenEncoderOutput_);
        for (ModelIndex i = 0u; i < parentScorer_->lenEncoderOutput_; i++) {
            f.at(0, i) = currentFeature_[i];
        }
        encoderOutput.emplace_back(f);
        inputs.push_back(std::make_pair(parentScorer_->inputsTensorNames_[0], Tensor::create(encoderOutput)));

        //declare output for the session run
        std::vector<Tensor> output;
        //run the session
        parentScorer_->session_.run(inputs, parentScorer_->outputTensorNames_, {}, output);
        //get the output of softmax and store it
        std::vector<Math::FastMatrix<Mm::Score>> monophoneScores;
        output[0].get(monophoneScores, false);

        // cache the scores for all calls at current ime frame from the search space
        for (ModelIndex centerstateIdx = 0u; centerstateIdx < parentScorer_->nCenterStates(); centerstateIdx++) {
            //ModelIndex outIdx = calculateCacheIndex(centerstateIdx, 0, 0);
            // we do not use log softmax

            Mm::Score  score  = -(parentScorer_->centerStateScale_ * Core::log(monophoneScores[0](0, centerstateIdx)));
            // prior is prepared in log space
            score += parentScorer_->centerStatePriorScale_* parentScorer_->centerStatePriors_[centerstateIdx];
            cache_.set(centerstateIdx, score);
        }

    }

    void TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::setMonophoneScoresWithTransition() const {

        std::vector<std::pair<std::string, Tensor>> inputs;
        std::vector<Math::FastMatrix<f32>> encoderOutput;
        Math::FastMatrix<f32> f(1, parentScorer_->lenEncoderOutput_);
        for (ModelIndex i = 0u; i < parentScorer_->lenEncoderOutput_; i++) {
            f.at(0, i) = currentFeature_[i];
        }
        encoderOutput.emplace_back(f);
        inputs.push_back(std::make_pair(parentScorer_->inputsTensorNames_[0], Tensor::create(encoderOutput)));

        if (parentScorer_->isMultiEncoderOutput_){

            u32 len_deltaEncoder = parentScorer_->lenEncoderOutput_;
            Math::FastMatrix<f32> f_delta(1, len_deltaEncoder);
            for (ModelIndex i = 0u; i < len_deltaEncoder; i++) {
                f_delta.at(0, i) = currentFeature_[parentScorer_->lenEncoderOutput_+i];
            }
            std::vector<Math::FastMatrix<f32>> deltaEncoderOutput;
            deltaEncoderOutput.emplace_back(f_delta);
            inputs.push_back(std::make_pair(parentScorer_->inputsTensorNames_[1], Tensor::create(deltaEncoderOutput)));
        }

        //declare output for the session run
        std::vector<Tensor> output;
        //run the session
        parentScorer_->session_.run(inputs, parentScorer_->outputTensorNames_, {}, output);
        //get the output of softmax and store it
        std::vector<Math::FastMatrix<Mm::Score>> monophoneScores;
        output[0].get(monophoneScores, false);

        switch (parentScorer_->transitionType_){
            case TransitionTypeFeature: {
                Math::FastMatrix<Mm::Score> transitionScores;
                output[1].get(transitionScores, false);

                parentScorer_->forwardScores_[0] = - transitionScores(0, 0);
                parentScorer_->forwardScores_[1] = - transitionScores(1, 0);
                parentScorer_->loopScores_[0]    = - transitionScores(0, 1);
                parentScorer_->loopScores_[1]    = - transitionScores(1, 1);
            }
                break;
            case TransitionTypeCenter: {
                Math::FastMatrix<Mm::Score> transitionScores;
                output[1].get(transitionScores, false);
                //scores are already in log
                for (ModelIndex centerstateIdx = 0u; centerstateIdx < parentScorer_->nCenterStates(); centerstateIdx++) {
                    parentScorer_->forwardScores_[centerstateIdx] = parentScorer_->forwardScale_ * (- transitionScores(centerstateIdx, 0));
                    parentScorer_->loopScores_[centerstateIdx]    = parentScorer_->loopScale_ * (- transitionScores(centerstateIdx, 1));
                }
                break;
            }
            case TransitionTypeFeatureCenter: {
                std::vector<Math::FastMatrix<Mm::Score>> transitionScores;
                output[1].get(transitionScores, false);
                for (ModelIndex centerstateIdx = 0u; centerstateIdx < parentScorer_->nCenterStates(); centerstateIdx++) {
                    Mm::Score fwdScore = transitionScores[0](0, centerstateIdx);


                    parentScorer_->forwardScores_[centerstateIdx] = parentScorer_->forwardScale_ * (- Core::log(fwdScore));
                    parentScorer_->loopScores_[centerstateIdx]    = parentScorer_->loopScale_ * (- Core::log(1 - fwdScore));
                }
                break;
            }
            case TransitionTypeFeatureCenterLeft:
            case TransitionTypeFeatureCenterRight: {
                defect();
            }
        }






        // cache the scores for all calls at current ime frame from the search space
        for (ModelIndex centerstateIdx = 0u; centerstateIdx < parentScorer_->nCenterStates(); centerstateIdx++) {
            //ModelIndex outIdx = calculateCacheIndex(centerstateIdx, 0, 0);
            // we do not use log softmax

            Mm::Score  score  = -(parentScorer_->centerStateScale_ * Core::log(monophoneScores[0](0, centerstateIdx)));
            // prior is prepared in log space
            score += parentScorer_->centerStatePriorScale_* parentScorer_->centerStatePriors_[centerstateIdx];
            cache_.set(centerstateIdx, score);
        }

    }

    void TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::setDiphoneScoresForAllContextsWithSilAdjust() const {

            u32 batchSize   = static_cast<u32>(parentScorer_->nContextLabels());

            std::vector<Math::FastMatrix<f32>> encoderOutput;
            Math::FastMatrix<s32>              currentStateIdentity(1, batchSize);

            Math::FastMatrix<f32> f(1, parentScorer_->lenEncoderOutput_);
            for (ModelIndex i = 0u; i < parentScorer_->lenEncoderOutput_; i++) {
                f.at(0, i) = currentFeature_[i];
            }

            for (ModelIndex pIdx = 0; pIdx < parentScorer_->nContextLabels(); pIdx++) {
                encoderOutput.emplace_back(f);
                currentStateIdentity.at(0, pIdx) = static_cast<s32>(mapLabelSetToDense(pIdx, 0, 0));
            }

            std::vector<std::pair<std::string, Tensor>> inputs;
            inputs.push_back(std::make_pair(parentScorer_->inputsTensorNames_[0], Tensor::create(encoderOutput)));
            inputs.push_back(std::make_pair(parentScorer_->inputsTensorNames_[1], Tensor::create(currentStateIdentity)));

            //declare output for the session run
            std::vector<Tensor> output;
            //run the session
            parentScorer_->session_.run(inputs, parentScorer_->outputTensorNames_, {}, output);

            //get the output of softmax and store it
            std::vector<Math::FastMatrix<Mm::Score>> contextDependentCenterStateScores;
            std::vector<Math::FastMatrix<Mm::Score>> contextScores;
            output[0].get(contextDependentCenterStateScores, false);
            output[1].get(contextScores, false);



            Bliss::Phoneme::Id silenceId = parentScorer_->getSilenceId();
            Mm::Score mergedScore = contextScores[0](0, 0) + contextScores[0](0, silenceId);

            std::vector<Mm::Score> mergedDpSore(parentScorer_->nCenterStates());
            for (ModelIndex midx = 0u; midx < parentScorer_->nCenterStates(); midx++) {
                mergedDpSore[midx] += contextDependentCenterStateScores[0](0, midx) + contextDependentCenterStateScores[0](silenceId, midx);

            }

            for (ModelIndex lidx = 0u; lidx < parentScorer_->nContextLabels(); lidx++) {
                for (ModelIndex cidx = 0u; cidx < parentScorer_->nCenterStates(); cidx++) {
                    ModelIndex outIdx = calculateCacheIndex(cidx, lidx, 0);

                    Mm::Score  score;
                    if (lidx == 0 or lidx == silenceId){
                        score  = -(parentScorer_->centerStateScale_ * Core::log(mergedDpSore[cidx]) +
                                   parentScorer_->leftContextScale_ * Core::log(mergedScore));
                    }
                    else{
                        //left context will have repetitive vectores you can also get just index (0, contextScores)
                        score  = -(parentScorer_->centerStateScale_ * Core::log(contextDependentCenterStateScores[0](lidx, cidx)) +
                                   parentScorer_->leftContextScale_ * Core::log(contextScores[0](lidx, lidx)));
                    }
                    // subtract negative prior score
                    score += (parentScorer_->centerStatePriorScale_ * parentScorer_->contextDependentCenterStatePriors_[lidx][cidx]) +
                             (parentScorer_->leftContextPriorScale_ * parentScorer_->leftContextPriors_[lidx]);

                    cache_.set(outIdx, score);

                }
            }

        }

    void TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::scoreActiveStatesTriphoneForward(const std::vector<Mm::MixtureIndex>& stateIdentities) const {

        auto batchSize   = stateIdentities.size();
        auto featureSize = static_cast<ModelIndex>(currentFeature_.size());
        std::vector<bool> visited(parentScorer_->nContextLabels() * parentScorer_->nCenterStates());

        std::vector<Math::FastMatrix<f32>> encoderOutput;
        Math::FastMatrix<f32> f(1, featureSize);
        for (ModelIndex i = 0u; i < currentFeature_.size(); i++) {
            f.at(0, i) = currentFeature_[i];
        }

        std::vector<ModelIndex>            pastContextIds;
        std::vector<ModelIndex>            centerStateIds;
        std::vector<s32>                   denseLabels;

        for (ModelIndex b = 0u; b < batchSize; b++){

            Mm::MixtureIndex     stateId            = stateIdentities[b];

            std::vector<u32> labels = getLabelIndices(stateId);
            Bliss::Phoneme::Id   pastContextLabel   = labels[0];
            Mm::EmissionIndex    centerStateLabel   = labels[1];

            ModelIndex vIdx = (pastContextLabel * parentScorer_->nCenterStates()) + centerStateLabel;
            if (!visited[vIdx]){

                visited[vIdx] = true;
                pastContextIds.emplace_back(pastContextLabel);
                centerStateIds.emplace_back(centerStateLabel);

                encoderOutput.emplace_back(f);
                denseLabels.emplace_back(stateId);

            }

        }

        batchSize = pastContextIds.size();
        Math::FastMatrix<s32>     currentStateIdentity(1, batchSize);
        for (ModelIndex i = 0u; i < batchSize; i++) {
            currentStateIdentity.at(0, i) = denseLabels[i];
        }



        std::vector<std::pair<std::string, Tensor>> inputs;
        inputs.push_back(std::make_pair(parentScorer_->inputsTensorNames_[0], Tensor::create(encoderOutput)));
        inputs.push_back(std::make_pair(parentScorer_->inputsTensorNames_[1], Tensor::create(currentStateIdentity)));

        //declare output for the session run
        std::vector<Tensor> output;
        //run the session
        //auto start = std::chrono::steady_clock::now();
        parentScorer_->session_.run(inputs, parentScorer_->outputTensorNames_, {}, output);

        //auto end = std::chrono::steady_clock::now();
        //double tf_time = std::chrono::duration<double, std::milli>(end - start).count();
        //addSetTime(tf_time);

        std::vector<Math::FastMatrix<Mm::Score>> triphoneScores;
        std::vector<Math::FastMatrix<Mm::Score>> diphoneScores;
        std::vector<Math::FastMatrix<Mm::Score>> pastContextScores;
        output[0].get(triphoneScores, false);
        output[1].get(diphoneScores, false);
        output[2].get(pastContextScores, false);




        for (ModelIndex b = 0u; b < batchSize; b++) {
            ModelIndex        e              = centerStateIds[b];
            ModelIndex        pastContextIdx = pastContextIds[b];

            for (ModelIndex futureIdx = 0u; futureIdx < parentScorer_->nContextLabels(); futureIdx++) {
                ModelIndex outIdx = calculateCacheIndex(e, pastContextIdx, futureIdx);

                Mm::Score score;


                score = -(parentScorer_->rightContextScale_ * Core::log(triphoneScores[0](b, futureIdx)) +
                          parentScorer_->centerStateScale_ * Core::log(diphoneScores[0](b, e)) +
                          parentScorer_->leftContextScale_ * Core::log(pastContextScores[0](b, pastContextIdx)));

                score += (parentScorer_->rightContextPriorScale_ * parentScorer_->contextDependentRightContextPriors_[pastContextIdx * parentScorer_->nCenterStates() + e][futureIdx]) +
                         (parentScorer_->centerStatePriorScale_  * parentScorer_->contextDependentCenterStatePriors_[pastContextIdx][e]) +
                         (parentScorer_->leftContextPriorScale_  * parentScorer_->leftContextPriors_[pastContextIdx]);


                cache_.set(outIdx, score);
            }
        }


    }

    void TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::scoreActiveStates(const std::vector<Mm::MixtureIndex>& stateIdentities) const {

        switch (parentScorer_->contextType_) {
            case ContextTypeMonophone:
                setMonophoneScores();
                break;
            case ContextTypeMonophoneDelta:
                setMonophoneScoresWithTransition();
                break;
            case ContextTypeDiphone:
                setDiphoneScoresForAllContextsWithSilAdjust();
                break;
            case ContextTypeDiphoneDelta:
                //ToDo:: implement
            case ContextTypeTriphoneForward:
                scoreActiveStatesTriphoneForward(stateIdentities);
                break;
            case ContextTypeTriphoneForwardDelta:
                //ToDo:: implement
            case ContextTypeTriphoneSymmetric:
                //ToDo:: implement
            case ContextTypeTriphoneBackward:
                //ToDo:: implement
            default:
                defect();
        }

    }

    Mm::Score TFFactoredHybridFeatureScorer::TFFactoredHybridContextScorer::scoreWithContext(Mm::MixtureIndex stateIdentiy) const {


        std::vector<u32> labels = getLabelIndices(stateIdentiy);

        ModelIndex      left   = labels[0];
        ModelIndex      center = labels[1];
        ModelIndex      right  = labels[2];

        //std::ofstream logfile;
        //logfile.open("/u/raissi/Desktop/labels_scorer.txt", std::ofstream::app);
        //logfile << "dense: " << stateIdentiy << ", left: " << left << ", center: " << center << ", right: " << right << std::endl;
        //logfile << "---------------------------------------------------" << std::endl;

        ModelIndex outputIndex = calculateCacheIndex(center,
                                                     mapPhonemeIdToContextId(left),
                                                     mapPhonemeIdToContextId(right));

        if (cache_.isCalculated(outputIndex)) {
            return cache_[outputIndex];
        }

        switch (parentScorer_->contextType_) {
            case ContextTypeMonophone:
                setMonophoneScores();
                break;
            case ContextTypeMonophoneDelta:
                setMonophoneScoresWithTransition();
                break;
            case ContextTypeDiphone:
                setDiphoneScoresForAllContextsWithSilAdjust();
                break;
            case ContextTypeDiphoneDelta:
                //ToDo:: implement
            case ContextTypeTriphoneForward:
                //ToDo:: implement
            case ContextTypeTriphoneForwardDelta:
                //ToDo:: implement
            case ContextTypeTriphoneSymmetric:
                //ToDo:: implement
            case ContextTypeTriphoneBackward:
                //ToDo:: implement
            default:
                defect();
        }

        return cache_[outputIndex];
    }




}  // namespace Tensorflow



