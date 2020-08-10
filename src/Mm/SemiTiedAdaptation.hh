/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
#ifndef _MM_SEMITIEDADAPTATION_HH
#define _MM_SEMITIEDADAPTATION_HH

#include <Math/Nr/nr.h>
#include "MllrAdaptation.hh"

namespace Mm {

class SemiTiedEstimator : public FullAdaptorViterbiEstimator {
public:
    typedef FullAdaptorViterbiEstimator Precursor;

private:
    typedef std::set<IdSetEntry>    IdSet;
    typedef std::map<NodeId, IdSet> NodeIdToIdSetMap;

    static const Core::ParameterFloat paramMinSemiTiedAdaptationObservations_;
    static const Core::ParameterFloat paramIterationStop_;

    Sum    minSemiTiedAdaptationObservations_;
    double stopCriterion_;

    void solveEstimationEquations(const IdSet& idSet, NodeId qId,
                                  const Math::Vector<Matrix>& z,
                                  const Math::Vector<Matrix>& g);

protected:
    SemiTiedEstimator(const Core::Configuration&          c,
                      ComponentIndex                      dimension,
                      const Core::Ref<Am::AdaptationTree> adaptationTree);

    virtual void estimateWMatrices();

public:
    SemiTiedEstimator(const Core::Configuration&          c,
                      const Core::Ref<Am::AdaptationTree> adaptationTree);
    SemiTiedEstimator(const Core::Configuration&          c,
                      Core::Ref<const Mm::MixtureSet>     mixtureSet,
                      const Core::Ref<Am::AdaptationTree> adaptationTree);
    virtual ~SemiTiedEstimator(){};

    virtual Sum minAdaptationObservations() {
        return minSemiTiedAdaptationObservations_;
    }

    // virtual AdaptorEstimator* clone() const;
    virtual std::string typeName() const {
        return "semi-tied-estimator";
    }
    virtual bool write(Core::BinaryOutputStream& o) const;
    virtual bool read(Core::BinaryInputStream& i);
};

class SemiTiedOptimizationFunction : public Math::Nr::FunctorBase<Math::Vector<double>, double> {
private:
    ComponentIndex             dimension_;
    const Math::Vector<Matrix>&G_, &Z_;

    mutable std::map<NodeId, Math::Vector<double>> lambda_;
    mutable Math::Matrix<double>                   U_, V_;
    mutable Math::Matrix<double>                   UT_;
    mutable Math::Matrix<double>                   f1_, f2_;
    mutable std::map<NodeId, Math::Vector<double>> f0_;
    mutable Math::Matrix<double>                   lc_;
    mutable Math::Matrix<double>                   AT_;

public:
    SemiTiedOptimizationFunction(ComponentIndex d, const Math::Vector<Matrix>& G,
                                 const Math::Vector<Matrix>&                   Z,
                                 const std::map<NodeId, Math::Vector<double>>& l);
    virtual ~SemiTiedOptimizationFunction(){};

    virtual ResultType operator()(const ArgumentType& x) const;
};

class SemiTiedOptimizationGradient : public Math::Nr::GradientBase<Math::Vector<double>, Math::Vector<double>> {
private:
    ComponentIndex             dimension_;
    const Math::Vector<Matrix>&G_, &Z_;

    mutable std::map<NodeId, Math::Vector<double>> lambda_;
    mutable Math::Matrix<double>                   U_, V_;
    mutable Math::Matrix<double>                   UT_, VT_;
    mutable Math::Matrix<double>                   f1_, f2_;
    mutable std::map<NodeId, Math::Vector<double>> f0_;
    mutable Math::Matrix<double>                   UT_U_;
    mutable Math::Matrix<double>                   lc_;
    mutable Math::Matrix<double>                   VT_Gc_V_;

public:
    SemiTiedOptimizationGradient(ComponentIndex d, const Math::Vector<Matrix>& G,
                                 const Math::Vector<Matrix>&                   Z,
                                 const std::map<NodeId, Math::Vector<double>>& l);
    virtual ~SemiTiedOptimizationGradient(){};

    virtual void operator()(const InputType& x, ResultType& f) const;
};

Math::Vector<double> convert2Vector(const std::map<NodeId, Math::Vector<double>>& l,
                                    const Math::Matrix<double>& u, const Math::Matrix<double>& v);
void                 convert2Matrices(const Math::Vector<double>& x, std::map<NodeId, Math::Vector<double>>& l,
                                      Math::Matrix<double>& u, Math::Matrix<double>& v, ComponentIndex dim);

}  //namespace Mm

#endif  //_MM_SEMITIEDADAPTATION_HH
