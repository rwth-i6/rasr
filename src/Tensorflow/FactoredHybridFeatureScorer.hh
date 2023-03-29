#ifndef RASR_FACTOREDHYBRIDFEATURESCORER_H
#define RASR_FACTOREDHYBRIDFEATURESCORER_H

#include <Bliss/Phoneme.hh>
#include <Math/Module.hh>
#include <Mm/FeatureScorer.hh>
#include <Mm/MixtureSet.hh>
#include <Nn/ClassLabelWrapper.hh>
#include <Nn/Types.hh>

#include "GraphLoader.hh"
#include "Module.hh"
#include "Session.hh"
#include "Tensor.hh"
#include "TensorMap.hh"

namespace Tensorflow {

    class TFFactoredHybridFeatureScorer : public Mm::FeatureScorer {
    public:
        typedef FeatureScorer Precursor;
        typedef u32           ModelIndex;
        typedef enum {
            ContextTypeMonophone,
            ContextTypeMonophoneDelta,
            ContextTypeDiphone,
            ContextTypeDiphoneDelta,
            ContextTypeTriphoneForward,
            ContextTypeTriphoneForwardDelta,
            ContextTypeTriphoneSymmetric,
            ContextTypeTriphoneBackward,
        } ContextType;
        typedef enum {
            TransitionTypeDefault,
            TransitionTypeConstant,
            TransitionTypeFeature,
            TransitionTypeCenter,
            TransitionTypeFeatureCenter,
            TransitionTypeFeatureCenterLeft,
            TransitionTypeFeatureCenterRight,
        } TransitionType;

        static const Core::Choice          choiceContextType;
        static const Core::Choice          choiceTransitionType;
        static const Core::ParameterString paramContextType;
        static const Core::ParameterString paramTransitionType;
        static const Core::ParameterInt    paramNumStatesPerPhone;
        static const Core::ParameterInt    paramNumContexts;
        static const Core::ParameterInt    paramSilenceId;
        static const Core::ParameterInt    paramLenEncoderOutput;
        static const Core::ParameterFloat  paramRightContextScale;
        static const Core::ParameterFloat  paramCenterStateScale;
        static const Core::ParameterFloat  paramLeftContextScale;
        static const Core::ParameterFloat  paramRightContextPriorScale;
        static const Core::ParameterFloat  paramCenterStatePriorScale;
        static const Core::ParameterFloat  paramLeftContextPriorScale;
        static const Core::ParameterString paramRightContextPriorFileName;
        static const Core::ParameterString paramCenterStatePriorFileName;
        static const Core::ParameterString paramLeftContextPriorFileName;
        static const Core::ParameterString paramConstantForwardProbFileName;
        static const Core::ParameterFloat  paramLoopScale;
        static const Core::ParameterFloat  paramForwardScale;
        static const Core::ParameterFloat  paramSilLoopPenalty;
        static const Core::ParameterFloat  paramSilForwardPenalty;
        static const Core::ParameterBool   paramBatchMajor;
        static const Core::ParameterBool   paramMinDuration;
        static const Core::ParameterBool   paramUseWordEndClasses;
        static const Core::ParameterBool   paramUseBoundaryClasses;
        static const Core::ParameterBool   paramMultiEncoderOutput;

        TFFactoredHybridFeatureScorer(const Core::Configuration&      config,
                                      Core::Ref<const Mm::MixtureSet> mixtures)
                : Core::Component(config),
                  Precursor(config),
                  numEmissions_(mixtures->nMixtures()),
                  paramContextType_(paramContextType(config)),
                  paramTransitionType_(paramTransitionType(config)),
                  numStatesPerPhone_(paramNumStatesPerPhone(config)),
                  //Generally contexts are all phonemes plus the boundary sign
                  numContexts_(paramNumContexts(config)),
                  silenceId_(paramSilenceId(config)),
                  lenEncoderOutput_(paramLenEncoderOutput(config)),
                  rightContextScale_(paramRightContextScale(config)),
                  centerStateScale_(paramCenterStateScale(config)),
                  leftContextScale_(paramLeftContextScale(config)),
                  rightContextPriorScale_(paramRightContextPriorScale(config)),
                  centerStatePriorScale_(paramCenterStatePriorScale(config)),
                  leftContextPriorScale_(paramLeftContextPriorScale(config)),
                  rightContextPriorFileName_(paramRightContextPriorFileName(config)),
                  centerStatePriorFileName_(paramCenterStatePriorFileName(config)),
                  leftContextPriorFileName_(paramLeftContextPriorFileName(config)),
                  constantForwardProbFileName_(paramConstantForwardProbFileName(config)),
                  loopScale_(paramLoopScale(config)),
                  forwardScale_(paramForwardScale(config)),
                  silLoopPenalty_(paramSilLoopPenalty(config)),
                  silForwardPenalty_(paramSilForwardPenalty(config)),
                  isBatchMajor_(paramBatchMajor(config)),
                  isMinimumDuration_(paramMinDuration(config)),
                  useWordEndClasses_(paramUseWordEndClasses(config)),
                  useBoundaryClasses_(paramUseBoundaryClasses(config)),
                  isMultiEncoderOutput_(paramMultiEncoderOutput(config)),
                  //no config needed for these
                  //ToDo: do an informed initialization based on the context
                  //loopScores_(mixtures->nMixtures()),
                  //forwardScores_(mixtures->nMixtures()),
                  //
                  session_(select("session")),
                  loader_(Module::instance().createGraphLoader(select("loader"))),
                  graph_(loader_->load_graph()),
                  inputMap_(select("input-map")),
                  outputMap_(select("output-map")) {

            setContextType();
            setTransitionType();
            setTransitionCaches();
            setPriors();
            setTensorNames();
            session_.addGraph(*graph_);
            loader_->initialize(session_);
            //consistency check
            require(!(useBoundaryClasses_ &&  useWordEndClasses_));

        }

        virtual ~TFFactoredHybridFeatureScorer() = default;

        virtual Mm::EmissionIndex nMixtures() const {
            return numEmissions_;
        };
        virtual u32 nContextLabels() const {
            return numContexts_;
        }
        virtual u32 nCenterStates() const {
            // Theoretically this should be number of center phonemes times number of states, by excluding
            // the boundary sign. But we use n-phone dense tying where boundary is counted as one of the phonemes.
            // In order to avoid hacking the indices we have indices 0, 1 and 2 together with silenceId + {1,2}
            // in softmax that are not used.
            u32 nStClasses = numContexts_;
            if(!isMinimumDuration_)
              nStClasses *= numStatesPerPhone_;
            if(useWordEndClasses_)
                nStClasses *= 2;
            return nStClasses;
        }
        virtual Bliss::Phoneme::Id getSilenceId() const {
            return silenceId_;
        }

        virtual ModelIndex getSilenceLabelId() const {
            // this is the silence id for a monophone dense tying.
            //In order to get the dense one should multiply by nContext*nContext
            ModelIndex result = silenceId_ * nCenterStates();
            if (useWordEndClasses_) {
                result *= 2;
            }
            return result;
        }

        virtual void              getFeatureDescription(Mm::FeatureDescription& description) const;
        virtual void              setContextType();
        virtual void              setTransitionType();
        virtual void              setTransitionCaches();
        virtual void              setPriors();
        virtual void              setTensorNames();
        virtual Math::Matrix<f32> readPriorMatrix(std::string fileName);
        virtual Math::Vector<f32> readPriorVector(std::string fileName);

        virtual Scorer getScorer(Core::Ref<const Mm::Feature> f) const {
            return getScorer(*f->mainStream());
        }

        virtual Scorer getScorer(const Mm::FeatureVector& f) const {
            return Scorer(new TFFactoredHybridContextScorer(this, f));
        }

    protected:
        Mm::EmissionIndex      numEmissions_;
        ContextType            contextType_;
        TransitionType         transitionType_;
        u32                    numStatesPerPhone_;
        u32                    numContexts_;
        Bliss::Phoneme::Id     silenceId_;
        u32                    lenEncoderOutput_;
        f32                    rightContextScale_;
        f32                    centerStateScale_;
        f32                    leftContextScale_;
        f32                    rightContextPriorScale_;
        f32                    centerStatePriorScale_;
        f32                    leftContextPriorScale_;
        std::string            rightContextPriorFileName_;
        std::string            centerStatePriorFileName_;
        std::string            leftContextPriorFileName_;
        std::string            constantForwardProbFileName_;
        f32                    loopScale_;
        f32                    forwardScale_;
        f32                    silLoopPenalty_;
        f32                    silForwardPenalty_;
        bool                   isMultiEncoderOutput_;
        bool                   isBatchMajor_;
        bool                   isMinimumDuration_;
        bool                   useWordEndClasses_;
        bool                   useBoundaryClasses_;
        std::string            paramContextType_;
        std::string            paramTransitionType_;
        Math::Vector<f32>      leftContextPriors_;
        Math::Vector<f32>      centerStatePriors_;
        Math::Vector<f32>      rightContextPriors_;
        Math::Matrix<f32>      contextDependentCenterStatePriors_; //for diphone and all triphones
        Math::Matrix<f32>      contextDependentRightContextPriors_; //for triphone forward and backward
        Math::Matrix<f32>      contextDependentLeftContextPriors_; //for triphone backward
        mutable std::vector<Mm::Score> forwardScores_;
        mutable std::vector<Mm::Score> loopScores_;


        //tensorflow related members for the second session
        mutable Session              session_;
        std::unique_ptr<GraphLoader> loader_;
        std::unique_ptr<Graph>       graph_;
        TensorInputMap               inputMap_;
        TensorOutputMap              outputMap_;
        std::vector<std::string>     inputsTensorNames_;
        std::vector<std::string>     outputTensorNames_;

    public:
        class TFFactoredHybridContextScorer : public ContextScorer {
            friend class TFFactoredHybridFeatureScorer;

        public:
            TFFactoredHybridContextScorer(const TFFactoredHybridFeatureScorer* featureScorer,
                                            const Mm::FeatureVector&    currentFeature)
                    : ContextScorer(),
                      parentScorer_(featureScorer),
                      currentFeature_(currentFeature),
                      cache_(getCacheLength())

            {}

            //Sprint scoring mechanism
            virtual Mm::EmissionIndex nEmissions() const {
                return parentScorer_->nMixtures();
            };
            virtual Mm::Score score(Mm::EmissionIndex e) const {
                return scoreWithContext(e);
            };
            virtual Mm::Score score(Mm::EmissionIndex e, ModelIndex modelIndex) {
                return 0.0;
            };
            virtual Mm::Score         scoreWithContext(Mm::MixtureIndex stateIdentiy) const;
            virtual bool              isTriphone() const;
            virtual bool              isDelta() const;
            virtual ModelIndex getDeltaIndex(Mm::EmissionIndex stateId) const;

            //General
            virtual std::vector<Mm::Score> getTransitionScores(const bool isLoop) const {
                ModelIndex silIdx = parentScorer_->getSilenceLabelId();
                parentScorer_->loopScores_[silIdx] += parentScorer_->silLoopPenalty_;
                if(isLoop)
                    return parentScorer_->loopScores_;
                else return parentScorer_->forwardScores_;
            }
            virtual u32 getCacheLength() const;

            std::vector<u32> getLabelIndices(Mm::EmissionIndex stateIdentity) const;
            Mm::EmissionIndex mapLabelSetToDense(Mm::EmissionIndex left, Mm::EmissionIndex e, Mm::EmissionIndex right) const;
            ModelIndex calculateCacheIndex(Mm::EmissionIndex e, ModelIndex pastContextId, ModelIndex futureContextId) const;

            // Currently there is no need to map, but I kept this for possible future mappings in case other
            // state tying types are used
            ModelIndex mapPhonemeIdToContextId(Bliss::Phoneme::Id phonemeId) const;


            //feature scorer specific scoring mechanism
            //batching
            virtual void setMonophoneScores() const;
            virtual void setMonophoneScoresWithTransition() const;
            virtual void setDiphoneScoresForAllContextsWithSilAdjust() const;
            virtual void setDiphoneScoresForAllContextsSilAdjustBatchMajor() const;

            // only active hypotheses
            virtual void scoreActiveStates(const std::vector<Mm::MixtureIndex>& stateIdentities) const;
            virtual void scoreActiveStatesTriphoneForward(const std::vector<Mm::MixtureIndex>& stateIdentities) const;
            virtual void scoreActiveStatesTriphoneForwardBatchMajor(const std::vector<Mm::MixtureIndex>& stateIdentities) const;

        private:
            TFFactoredHybridFeatureScorer const*  parentScorer_;
            const Mm::FeatureVector      currentFeature_;
            mutable Mm::Cache<Mm::Score> cache_;
        };
    };

}  // namespace Tensorflow







#endif //RASR_FACTOREDHYBRIDFEATURESCORER_H




