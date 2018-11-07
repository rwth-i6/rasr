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
#ifndef EMISSIONLATTICERESCORER_HH_
#define EMISSIONLATTICERESCORER_HH_

#include <Speech/LatticeExtractor.hh>
#include <Speech/LatticeRescorerAutomaton.hh>
#include <Speech/SegmentwiseFeatureExtractor.hh>
#include <Mm/FeatureScorer.hh>

#include "SharedNeuralNetwork.hh"
#include "Types.hh"

namespace Nn {

/*
 * emission rescorer automaton for NN acoustic models
 *
 * simple design, not multithreading safe
 * network is forwarded in constructor, activations are stored in network
 *
 * ASSUMPTIONS:
 *  -> output layer is softmax
 *  -> log-prior is already removed from bias parameters
 *  -> for sequence training: network is not modified between rescoring and disciminative accumulation
 *
 */

class EmissionLatticeRescorer : public virtual Speech::AcousticLatticeRescorer {
    typedef Speech::AcousticLatticeRescorer Precursor;
private:
    static const Core::ParameterString paramPortName;
    static const Core::ParameterBool paramCheckValues;
    static const Core::ParameterBool paramMeasureTime;

private:
    // measure runtime
    const bool measureTime_;
    const bool checkValues_;
    f64 timeMemoryAllocation_;
    f64 timeForwarding_;
    f64 timeIO_;
protected:
    Core::Ref<Speech::SegmentwiseFeatureExtractor> segmentwiseFeatureExtractor_;
    Flow::PortId portId_;
protected:
    virtual Lattice::ConstWordLatticeRef work(Lattice::ConstWordLatticeRef, Bliss::SpeechSegment *);
    void logProperties() const;
    NeuralNetwork<f32>& network() const { return SharedNeuralNetwork::network(); }
    const ClassLabelWrapper& labelWrapper() const { return SharedNeuralNetwork::labelWrapper(); }
public:
    EmissionLatticeRescorer(const Core::Configuration &,bool initialize = true);
    EmissionLatticeRescorer(const Core::Configuration &, Core::Ref<Am::AcousticModel>);
    virtual ~EmissionLatticeRescorer() {}

    void setSegmentwiseFeatureExtractor(Core::Ref<Speech::SegmentwiseFeatureExtractor>);
    virtual void setAlignmentGenerator(AlignmentGeneratorRef alignmentGenerator);
    virtual void finalize();
};


/*
 *
 * Automaton that actually does the rescoring
 *
 */
class EmissionLatticeRescorerAutomaton : public Speech::CachedLatticeRescorerAutomaton {
    typedef Speech::CachedLatticeRescorerAutomaton Precursor;
    typedef Speech::ConstSegmentwiseFeaturesRef ConstSegmentwiseFeaturesRef;
    typedef Speech::TimeframeIndex TimeframeIndex;
    typedef Core::Ref<Speech::PhonemeSequenceAlignmentGenerator> AlignmentGeneratorRef;
protected:
    typedef Types<f32>::NnVector NnVector;
    typedef Types<f32>::NnMatrix NnMatrix;
public:
    EmissionLatticeRescorerAutomaton();
    virtual ~EmissionLatticeRescorerAutomaton();
private:
    AlignmentGeneratorRef alignmentGenerator_;
    Speech::Alignment::LabelType labelType_;
    Core::Ref<Am::AcousticModel> acousticModel_;
    ConstSegmentwiseFeaturesRef features_;
    std::vector<NnMatrix> inputFeatures_;
    f64 *timeRetrieveAlignment_;
protected:
    // returns emission score of arc a, outgoing from state s
    virtual Fsa::Weight score(Fsa::StateId s, const Fsa::Arc &a) const;
    virtual bool forwardNetwork(bool checkValues);
    NeuralNetwork<f32>& network() const { return SharedNeuralNetwork::network(); }
    const ClassLabelWrapper& labelWrapper() const { return SharedNeuralNetwork::labelWrapper(); }
public:
    EmissionLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice,
            AlignmentGeneratorRef alignmentGenerator, Core::Ref<Am::AcousticModel> acousticModel, ConstSegmentwiseFeaturesRef features, bool checkValues);

    virtual std::string describe() const {
        return Core::form("nn-emission-rescore(%s)", fsa_->describe().c_str());
    }

    // returns emission score of coarticulatedPronunciation
    // alignment is read from cache or generated on demand
    Fsa::Weight _score(const Bliss::Coarticulated<Bliss::LemmaPronunciation> &coarticulatedPronunciation,
            TimeframeIndex begtime, TimeframeIndex endtime) const;
};

/** CachedNeuralNetworkFeatureScorer
 *
 * Provides the scores which are stored in the activations of a neural network.
 * Here, the static neural network which is created for sequence training is accessed.
 * The feature scorer only provides a method getTimeIndexedScorer(timeIndex), but not the common methods getScorer(feature)
 *
 */
class CachedNeuralNetworkFeatureScorer : public Mm::FeatureScorer {
    typedef Mm::FeatureScorer Precursor;
protected:
    class ActivationLookupScorer : public ContextScorer {
    protected:
        const CachedNeuralNetworkFeatureScorer *featureScorer_;
        const u32 time_;
    public:
        ActivationLookupScorer(const CachedNeuralNetworkFeatureScorer *featureScorer, u32 time) :
            featureScorer_(featureScorer),
            time_(time)
        {}
        virtual ~ActivationLookupScorer() {}

        virtual Mm::EmissionIndex nEmissions() const { return featureScorer_->nMixtures(); }
        virtual Mm::Score score(Mm::EmissionIndex e) const  { return featureScorer_->score(time_, e); }
    };
    friend class ContextScorer;
protected:
    u32 nMixtures_;
protected:
    NeuralNetwork<f32>& network() const { return SharedNeuralNetwork::network(); }
    const ClassLabelWrapper& labelWrapper() const { return SharedNeuralNetwork::labelWrapper(); }
public:
    typedef Core::Ref<const Mm::FeatureScorer::ContextScorer> Scorer;

    CachedNeuralNetworkFeatureScorer(const Core::Configuration &c, Core::Ref<const Mm::MixtureSet> mixtureSet) :
        Core::Component(c),
        Precursor(c),
        nMixtures_(mixtureSet->nMixtures())
    {
        log("creating nn-cached feature scorer");
    }

    virtual ~CachedNeuralNetworkFeatureScorer() {}

    virtual Scorer getTimeIndexedScorer(u32 time) const;

    Mm::Score score(u32 time, Mm::EmissionIndex e) const;

    virtual bool hasTimeIndexedCache() const { return true; }

    // methods required by feature scorer interface
    virtual Mm::EmissionIndex nMixtures() const { return nMixtures_; }
    virtual Mm::ComponentIndex dimension() const;

    virtual void getFeatureDescription(Mm::FeatureDescription &description) const {}
    virtual void getDependencies(Core::DependencySet &dependencies) const {}


    // not available for this feature scorer
    virtual Scorer getScorer(Core::Ref<const Mm::Feature> f) const;
    virtual Scorer getScorer(const Mm::FeatureVector &f) const;


};

}; // namespace Nn


#endif /* EMISSIONLATTICERESCORER_HH_ */
