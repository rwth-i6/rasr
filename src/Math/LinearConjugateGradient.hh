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
#ifndef LINEARCONJUGATEGRADIENT_HH_
#define LINEARCONJUGATEGRADIENT_HH_

#include <Mm/Types.hh>
#include <Math/Blas.hh>
#include <Math/LbfgsSolver.hh>
#include <Core/Application.hh>
#include <cmath>
#include "Utilities.hh"

/*
 * require that S is a vector-like class with the following methods
 *
 *  add
 *  dot
 *  copy
 *  scale
 *
 *  elementwiseDivision
 *
 *  empty
 *  copyStructure
 *  setToZero
 *  clear
 *  copy constructor
 *  swap
 *
 */


namespace Math {

template<class S, typename T>
class CgPreconditioner {
public:
    CgPreconditioner(){};
    virtual void applyInversePreconditioner(const S &in, S &out) = 0;
};

template<class S, typename T>
class DiagonalCgPreconditioner : public CgPreconditioner<S,T> , public S {
public:
    DiagonalCgPreconditioner(){};
    virtual void applyInversePreconditioner(const S &in, S &out);
};

template<class S, typename T>
void DiagonalCgPreconditioner<S,T>::applyInversePreconditioner(const S &in, S &out){
    out.copy(in);
    out.elementwiseDivision(*this);
}

template<class S, typename T>
class LbfgsPreconditioner : public CgPreconditioner<S,T>, public LbfgsSolver<S,T> {
public:
    LbfgsPreconditioner(){};
    virtual void applyInversePreconditioner(const S &in, S &out);
};

template<class S, typename T>
void LbfgsPreconditioner<S,T>::applyInversePreconditioner(const S &in, S &out){
    this->solve(in, out);
}


template<class S, typename T>
class LinearConjugateGradient {
public:
    // CG configuration
    struct CgConfiguration {
        bool usePreconditioning_;
        u32 maxIterations_;
        u32 minIterations_;
        int verbosity_;
        T maxIterateNorm_;
        T residualTolerance_;
        T objectiveFunctionDecreaseTolerance_;
        bool terminateBasedOnAverageObjectiveFunction_;
        bool evaluateObjectiveFunction_;
        bool evaluateIterateNorm_;
        u32 averagingHistoryLength_;
        bool dynamicAveragingHistoryLength_;
        bool terminateBasedOnResidualNorm_;
        bool storeIntermediateResults_;
        f64 backtrackingBase_;


        CgConfiguration() :
            usePreconditioning_(false),
            maxIterations_(250),
            minIterations_(10),
            verbosity_(2),
            maxIterateNorm_(0.0),
            residualTolerance_(0.0),
            objectiveFunctionDecreaseTolerance_(0.0005),
            terminateBasedOnAverageObjectiveFunction_(true),
            evaluateObjectiveFunction_(true),
            evaluateIterateNorm_(true),
            averagingHistoryLength_(10),
            dynamicAveragingHistoryLength_(true),
            terminateBasedOnResidualNorm_(false),
            storeIntermediateResults_(false),
            backtrackingBase_(1.3)
        {
            if (terminateBasedOnAverageObjectiveFunction_){
                minIterations_ = std::max(minIterations_, averagingHistoryLength_);
                evaluateObjectiveFunction_ = true;
            }
            if (maxIterateNorm_ > 0)
                evaluateIterateNorm_ = true;

        }
    };
public:
    enum TerminationReason {
        noTermination = 0,
        zeroResidualTermination = 1,
        residualToleranceTermination = 2,
        maxIterateNormTermination = 3,
        objectiveFunctionTermination = 4,
        maxIterationsTermination = 5
    };

public:
    CgConfiguration configuration;
protected:
    CgPreconditioner<S, T> *preconditioner_;
    // need to be allocated in init method
    S searchDirection_;
    S residual_;
    S pcResidual_;
    S matrixVectorproduct_;
    // vector of pairs (iteration index, searchDirection at iteration)
    std::vector<std::pair<u32, S> > intermediateResults_;
    // passed in solve method, memory not managed by CG
    const S *rhs_;
    S *iterate_;
    // statistics
    // squared norm of residual (in PC norm for PCG)
    T residualNormSquared_;
    T iterateNorm_;
    std::vector<T> objectiveFunction_;
    u32 nextBacktrackingIndex_;
protected:
    void initializeCg(const S &initialization);
    void initializeCgFromZero();
    T getResidualnormSquared() const;
    T getCurvatureproduct() const;
    void updateIterate(T stepsize);
    void updateResidual(const T stepsize);
    void updateSearchdirection(T residualnormratio);
    void storeIntermediateResult(u32 iter);
    T getIntermediateStepSize();

    // override these methods in a particular implementation
    virtual void applyMatrix(const S &in, S &out) = 0;
    virtual int terminationCriterionApplies(u32 iter);

    template<typename M>
    void vlog(int verbosity, const  char *t, const M message) const;
    void logTerminationReason(TerminationReason reason);
public:
    LinearConjugateGradient();
    virtual ~LinearConjugateGradient();

    void setPreconditioner(CgPreconditioner<S, T> *preconditioner){ preconditioner_ = preconditioner; }

    // allocate searchdirection, residual, matrixvectorproduct, and intermediateResults with the same dimensions as rhs
    virtual void allocate(const S &rhs);
    virtual void clear();

    s32 solve(const S &rhs, const S &initialization, S &solution, u32 &nIterations);

    T getCgObjectivefunction();

    u32 numberOfStoredIntermediateResults() const;
    u32 getIntermediateResult(u32 index, const S * &intermediateResult, T &objectiveFunction) const;
    CgPreconditioner<S,T>* getPreconditioner(){return preconditioner_; }
};

template<class S, typename T>
LinearConjugateGradient<S, T>::LinearConjugateGradient() :
preconditioner_(0),
rhs_(0),
iterate_(0),
residualNormSquared_((T) 0),
iterateNorm_((T) 0),
nextBacktrackingIndex_(0)
{
    nextBacktrackingIndex_ = (u32) std::ceil(configuration.backtrackingBase_);
}

template<class S, typename T>
LinearConjugateGradient<S, T>::~LinearConjugateGradient() {
    if (preconditioner_)
        delete preconditioner_;
}


template<class S, typename T>
void LinearConjugateGradient<S, T>::allocate(const S &rhs){
    searchDirection_.copyStructure(rhs);
    residual_.copyStructure(rhs);
    matrixVectorproduct_.copyStructure(rhs);
    searchDirection_.setToZero();
    residual_.setToZero();
    matrixVectorproduct_.setToZero();
    if (configuration.usePreconditioning_){
        pcResidual_.copyStructure(rhs);
        pcResidual_.setToZero();
    }
}

template<class S, typename T>
void LinearConjugateGradient<S, T>::clear(){
    searchDirection_.clear();
    residual_.clear();
    matrixVectorproduct_.clear();
    intermediateResults_.clear();
}


template<class S, typename T>
void LinearConjugateGradient<S, T>::initializeCg(const S &initialization){
    vlog(1, "initializing CG", " explicitly");
    // residual = A initialization - rhs
    applyMatrix(initialization, residual_);
    residual_.add(*rhs_, (T) -1.0);
    if (configuration.usePreconditioning_){
        require(preconditioner_);
        preconditioner_->applyInversePreconditioner(residual_, pcResidual_);
    }

    residualNormSquared_ = configuration.usePreconditioning_ ? pcResidual_.dot(residual_) : residual_.dot(residual_);

    // searchdirection = -residual (resp. -pcResidual for PCG)
    if (configuration.usePreconditioning_)
        searchDirection_.add(pcResidual_, (T) -1.0);
    else
        searchDirection_.add(residual_, (T) -1.0);

    // iterate = initialization
    iterate_->copy(initialization);
    // TODO objective function for PCG ??
    if (configuration.evaluateObjectiveFunction_){
        objectiveFunction_.resize(configuration.maxIterations_ + 1);
        objectiveFunction_.at(0) = getCgObjectivefunction();
        vlog(1, "cg objective function: ", objectiveFunction_.at(0));
    }
    if (configuration.evaluateIterateNorm_){
        iterateNorm_ = std::sqrt(iterate_->dot(*iterate_));
        vlog(1, "iterate norm: ", iterateNorm_);
    }
}

template<class S, typename T>
void LinearConjugateGradient<S, T>::initializeCgFromZero(){
    vlog(1, "initializing CG", " from zero");
    // residual = -rhs
    residual_.add(*rhs_, (T) -1.0);

    if (configuration.usePreconditioning_){
        require(preconditioner_);
        preconditioner_->applyInversePreconditioner(residual_, pcResidual_);
    }


    residualNormSquared_ = configuration.usePreconditioning_ ? pcResidual_.dot(residual_) : residual_.dot(residual_);

    // CG:  searchdirection = -residual = rhs
    // PCG: searchdirection = -pcResidual
    if (configuration.usePreconditioning_)
        searchDirection_.add(pcResidual_, (T) -1.0);
    else
        searchDirection_.copy(*rhs_);

    // iterate = 0
    iterate_->setToZero();
    iterateNorm_ = 0.0;
    if (configuration.evaluateObjectiveFunction_){
        objectiveFunction_.resize(configuration.maxIterations_ + 1);
        objectiveFunction_.at(0) = 0.0;
        vlog(1, "cg objective function: ", objectiveFunction_.at(0));
    }
    if (configuration.evaluateIterateNorm_){
        iterateNorm_ = std::sqrt(iterate_->dot(*iterate_));
        vlog(1, "iterate norm: ", iterateNorm_);
    }

}


template<class S, typename T>
s32 LinearConjugateGradient<S, T>::solve(const S &rhs, const S &initialization, S &iterate, u32 &nIterations){
    rhs_ = &rhs;
    iterate_ = &iterate;

    vlog(1, "running conjugate gradient", "");
    if (!initialization.empty())
        initializeCg(initialization);
    else
        initializeCgFromZero();

    for (u32 iter = 1; iter <= configuration.maxIterations_; iter++){
        vlog(1, "CG iteration ", iter);

        T residualNormSquared = residualNormSquared_;
        // compute matrix vector product Ap
        applyMatrix(searchDirection_, matrixVectorproduct_);

        T curvatureProduct = getCurvatureproduct();
        vlog(1, "curvature product: ", curvatureProduct);
        verify(curvatureProduct >= 0);

        T stepsize = residualNormSquared / curvatureProduct;

        updateIterate(stepsize);

        if (configuration.evaluateIterateNorm_){
            iterateNorm_ = std::sqrt(iterate.dot(iterate));
            vlog(1, "iterate norm: ", iterateNorm_);
        }

        updateResidual(stepsize);
        T newResidualNormSquared = residualNormSquared_;
        if (configuration.usePreconditioning_)
            vlog(1, "residual PC-norm: ", std::sqrt(newResidualNormSquared));
        else
            vlog(1, "residual norm: ", std::sqrt(newResidualNormSquared));


        // evaluation of objective function requires residual norm
        if (configuration.evaluateObjectiveFunction_){
            objectiveFunction_.at(iter) = getCgObjectivefunction();
            vlog(1, "cg objective function: ", objectiveFunction_.at(iter));
        }

        // check termination
        int terminationValue = terminationCriterionApplies(iter);
        if (terminationValue != noTermination){
            nIterations = iter;
            logTerminationReason((TerminationReason) terminationValue);
            // exceeding max iterate norm: undo last step
            if (terminationValue == maxIterateNormTermination ){
                vlog(1, "only going until boundary", "");
                updateIterate(-stepsize);
                T intermediateStepSize = getIntermediateStepSize();
                updateIterate(intermediateStepSize);
                iterateNorm_ = std::sqrt(iterate_->dot(*iterate_));
                // TODO remove
                T checkVal = std::abs(iterateNorm_ - configuration.maxIterateNorm_);
                if (checkVal > 10.0*Core::Type<T>::epsilon)
                    Core::Application::us()->warning("iterate norm is ") << iterateNorm_ << ", mismatch with max iterate norm: " << checkVal;
            }
            return terminationValue;
        }

        // update searchdirection
        T residualNormRatio = newResidualNormSquared / residualNormSquared;
        updateSearchdirection(residualNormRatio);

        // store intermediate result
        if (configuration.storeIntermediateResults_ && nextBacktrackingIndex_ == iter){
            storeIntermediateResult(iter);
            nextBacktrackingIndex_ = std::ceil(nextBacktrackingIndex_ * configuration.backtrackingBase_);
        }
    }
    nIterations = configuration.maxIterations_;
    logTerminationReason(maxIterationsTermination);
    return maxIterationsTermination;
}

template<class S, typename T>
T LinearConjugateGradient<S, T>::getCgObjectivefunction(){
    T result = iterate_ ? ((T) 0.5) * (iterate_->dot(residual_) - iterate_->dot(*rhs_)) : 0;
    return result;
}

// return r' * r
template<class S, typename T>
T LinearConjugateGradient<S, T>::getResidualnormSquared() const{
    return residualNormSquared_;
}

// return p^T A p, requires that Ap is computed
template<class S, typename T>
T LinearConjugateGradient<S, T>::getCurvatureproduct() const {
    return searchDirection_.dot(matrixVectorproduct_);
}

// update iterate: x += stepsize * p
template<class S, typename T>
void LinearConjugateGradient<S, T>::updateIterate(const T stepsize){
    iterate_->add(searchDirection_, stepsize);
}

// update residual: r += alpha * Ap
template<class S, typename T>
void LinearConjugateGradient<S, T>::updateResidual(const T stepsize){
    residual_.add(matrixVectorproduct_, stepsize);
    if (configuration.usePreconditioning_){
        require(preconditioner_);
        preconditioner_->applyInversePreconditioner(residual_, pcResidual_);
    }

    residualNormSquared_ = configuration.usePreconditioning_ ? pcResidual_.dot(residual_) : residual_.dot(residual_);
}

// update search direction:
// (CG)  p = -r + residualnormratio * p
// (PCG) p = -z + residualnormratio * p
template<class S, typename T>
void LinearConjugateGradient<S, T>::updateSearchdirection(T residualnormratio){
    if (configuration.usePreconditioning_)
        searchDirection_.copy(pcResidual_);
    else
        searchDirection_.copy(residual_);

    searchDirection_.scale((T) -1.0);
    searchDirection_.add(searchDirection_, residualnormratio);
}

template<class S, typename T>
void LinearConjugateGradient<S, T>::storeIntermediateResult(u32 iter){
    intermediateResults_.push_back(std::pair<u32, S>(iter, *iterate_));
}

template<class S, typename T>
T LinearConjugateGradient<S, T>::getIntermediateStepSize(){
    T squaredNormSearchDirection = searchDirection_.dot(searchDirection_);
    T p = 2.0 / squaredNormSearchDirection * iterate_->dot(searchDirection_);
    T q = 1.0 / squaredNormSearchDirection * (iterate_->dot(*iterate_) - configuration.maxIterateNorm_ * configuration.maxIterateNorm_);
    T dummySolution = 0.0, solution = 0.0;
    bool hasSolution = solveQuadraticEquation(p, q, solution, dummySolution);
    require(hasSolution);
    return solution;
}

template<class S, typename T>
u32 LinearConjugateGradient<S, T>::numberOfStoredIntermediateResults() const {
    return intermediateResults_.size();
}

template<class S, typename T>
u32 LinearConjugateGradient<S, T>::getIntermediateResult(u32 index, const S * &intermediateResult, T &objectiveFunction) const {
    require_lt(index , intermediateResults_.size());
    intermediateResult = &(intermediateResults_.at(index).second);
    u32 iteration = intermediateResults_.at(index).first;
    verify_lt(iteration, objectiveFunction_.size());
    objectiveFunction = objectiveFunction_.at(iteration);
    return iteration;
}

template<class S, typename T>
int LinearConjugateGradient<S, T>::terminationCriterionApplies(u32 iter){
    if (residualNormSquared_ == (T) 0.0)
        return zeroResidualTermination;
    if (configuration.terminateBasedOnResidualNorm_ && residualNormSquared_ <= configuration.residualTolerance_)
        return residualToleranceTermination;
    if (configuration.maxIterateNorm_ > 0 && iterateNorm_ > configuration.maxIterateNorm_)
        return maxIterateNormTermination;
    if (configuration.terminateBasedOnAverageObjectiveFunction_ && iter >= configuration.averagingHistoryLength_){
        u32 historyLength = configuration.dynamicAveragingHistoryLength_ ?
                std::max(configuration.averagingHistoryLength_, (u32) std::ceil(1.0 / configuration.averagingHistoryLength_ * iter)) :
                configuration.averagingHistoryLength_;
        verify(iter - historyLength >= 0);
        T decrease = (objectiveFunction_.at(iter) - objectiveFunction_.at(iter - historyLength)) / objectiveFunction_.at(iter);
        if (objectiveFunction_.at(iter) < 0 && decrease < historyLength * configuration.objectiveFunctionDecreaseTolerance_)
            return objectiveFunctionTermination;
    }
    return noTermination;
}


template<class S, typename T> template<typename M>
void LinearConjugateGradient<S, T>::vlog(int verbosity, const  char *t, const M message) const {
    if (configuration.verbosity_ >= verbosity)
        Core::Application::us()->log() << t << message;
}

template<class S, typename T>
void LinearConjugateGradient<S, T>::logTerminationReason(TerminationReason reason){
    if (reason == zeroResidualTermination)
        vlog(1, "termination reason: max iterate norm exceeded", "");
    else if (reason == residualToleranceTermination)
        vlog(1, "termination reason: residual tolerance", "");
    else if (reason == maxIterateNormTermination)
        vlog(1, "termination reason: maximal iterate norm reached", "");
    else if (reason == objectiveFunctionTermination)
        vlog(1, "termination reason: change in objective function below threshold", "");
    else if (reason == maxIterationsTermination)
        vlog(1, "termination reason: maximal number of iterations performed", "");
}

}

#endif /* LINEARCONJUGATEGRADIENT_HH_ */
