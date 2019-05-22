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
#ifndef LBFGSSOLVER_HH_
#define LBFGSSOLVER_HH_

namespace Math {

template<class S, typename T>
class LbfgsSolver {
protected:
    std::vector<S*> iterates_;
    std::vector<S*> gradients_;
    std::vector<T>  rhos_;
    std::vector<T>  alphas_;
    int             historySize_;

protected:
    void         addsi(u32 i, T stepsize, S& vector) const;
    void         addyi(u32 i, T stepsize, S& vector) const;
    T            dotsi(const S& vector, u32 i) const;
    T            dotyi(const S& vector, u32 i) const;
    void         setRhos();
    void         setRhoi(u32 i);
    virtual void multiplyWithInitialApproximation(S& vector) const;
    void         firstLoop(S& vector);
    void         secondLoop(S& vector);

public:
    LbfgsSolver()
            : historySize_(-1){};
    virtual void solve(const S& in, S& out);
    void         setIterate(u32 i, S* iterate);
    void         setGradient(u32 i, S* gradient);
    void         setHistorySize(int historyLength) {
        historySize_ = historyLength;
    }
    int getHistorySize() {
        return historySize_ == -1 ? iterates_.size() - 1 : historySize_;
    }
};

template<class S, typename T>
void LbfgsSolver<S, T>::addsi(u32 i, T stepsize, S& vector) const {
    require_lt(i, iterates_.size() - 1);
    vector.add(*iterates_.at(i + 1), stepsize);
    vector.add(*iterates_.at(i), -stepsize);
}

template<class S, typename T>
void LbfgsSolver<S, T>::addyi(u32 i, T stepsize, S& vector) const {
    require_lt(i, gradients_.size() - 1);
    vector.add(*gradients_.at(i + 1), stepsize);
    vector.add(*gradients_.at(i), -stepsize);
}

template<class S, typename T>
T LbfgsSolver<S, T>::dotsi(const S& vector, u32 i) const {
    require_lt(i, iterates_.size() - 1);
    T result = vector.dot(*iterates_.at(i + 1));
    result -= vector.dot(*iterates_.at(i));
    return result;
}

template<class S, typename T>
T LbfgsSolver<S, T>::dotyi(const S& vector, u32 i) const {
    require_lt(i, gradients_.size() - 1);
    T result = vector.dot(*gradients_.at(i + 1));
    result -= vector.dot(*gradients_.at(i));
    return result;
}

template<class S, typename T>
void LbfgsSolver<S, T>::setRhos() {
    require_eq(gradients_.size(), iterates_.size());
    require_gt(iterates_.size(), 1);
    int nIterates = iterates_.size();
    rhos_.resize(nIterates - 1);
    int m = getHistorySize();
    for (int i = nIterates - 2; i >= nIterates - 1 - m; i--)
        setRhoi(i);
}

template<class S, typename T>
void LbfgsSolver<S, T>::setRhoi(u32 i) {
    require_lt(i, iterates_.size() - 1);
    require_lt(i, rhos_.size());
    T rhoInverse = dotyi(*iterates_.at(i + 1), i);
    rhoInverse -= dotyi(*iterates_.at(i), i);
    rhos_.at(i) = 1.0 / rhoInverse;
}

template<class S, typename T>
void LbfgsSolver<S, T>::firstLoop(S& out) {
    require_ge(iterates_.size(), 2);
    int nIterates = iterates_.size();
    alphas_.resize(nIterates - 1);
    int m = getHistorySize();
    for (int i = nIterates - 2; i >= nIterates - 1 - m; i--) {
        alphas_.at(i) = rhos_.at(i) * dotsi(out, i);
        addyi(i, -alphas_.at(i), out);
    }
}

template<class S, typename T>
void LbfgsSolver<S, T>::secondLoop(S& out) {
    require_ge(gradients_.size(), 2);
    require_eq(rhos_.size(), gradients_.size() - 1);
    int nIterates = gradients_.size();
    int m         = getHistorySize();
    for (int i = nIterates - 1 - m; i < nIterates - 1; i++) {
        T beta = rhos_.at(i) * dotyi(out, i);
        addsi(i, alphas_.at(i) - beta, out);
    }
}

template<class S, typename T>
void LbfgsSolver<S, T>::solve(const S& in, S& out) {
    require_eq(iterates_.size(), gradients_.size());
    out.copy(in);
    if (iterates_.size() <= 1)  // no preconditioning possible
        return;
    // compute rhos
    setRhos();
    // run first loop
    firstLoop(out);
    // apply initial Hessian approximation
    multiplyWithInitialApproximation(out);
    // run second loop
    secondLoop(out);
}

template<class S, typename T>
void LbfgsSolver<S, T>::multiplyWithInitialApproximation(S& vector) const {
    require_ge(iterates_.size(), 2);
    require_ge(gradients_.size(), 2);
    int nIterates = iterates_.size();
    T   factor    = dotyi(*iterates_.at(nIterates - 1), nIterates - 2);
    factor -= dotyi(*iterates_.at(nIterates - 2), nIterates - 2);
    T den = dotyi(*gradients_.at(nIterates - 1), nIterates - 2);
    den -= dotyi(*gradients_.at(nIterates - 2), nIterates - 2);
    factor /= den;
    vector.scale(factor);
}

template<class S, typename T>
void LbfgsSolver<S, T>::setIterate(u32 i, S* iterate) {
    iterates_.resize(i + 1);
    iterates_.at(i) = iterate;
}

template<class S, typename T>
void LbfgsSolver<S, T>::setGradient(u32 i, S* gradient) {
    gradients_.resize(i + 1);
    gradients_.at(i) = gradient;
}

}  // namespace Math

#endif /* LBFGSSOLVER_HH_ */
