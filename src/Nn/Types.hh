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
#ifndef TYPES_HH_
#define TYPES_HH_

#include <Core/Types.hh>
#include <Flow/Vector.hh>
#include <Search/Types.hh>

namespace Math {
template<typename T>
class CudaMatrix;
template<typename T>
class CudaVector;
}  // namespace Math

namespace Nn {

template<typename T>
class Types {
public:
    typedef Math::CudaMatrix<T> NnMatrix;
    typedef Math::CudaVector<T> NnVector;
};

typedef Flow::Vector<f32>            FeatureVector;
typedef Flow::DataPtr<FeatureVector> FeatureVectorRef;

typedef Search::Score Score;

/*
 * Auxiliary class to store a value in negative log space and handle automatic conversion into and out of log/neg-log/prob space
 */
class NegLogScore {
private:
    Score value;  // Value in negative log space

public:
    NegLogScore()
            : value(0.0) {}

    // Construct from a value that's already in negative log space
    explicit NegLogScore(Score val)
            : value(val) {}

    static NegLogScore fromProb(Score prob) {
        return NegLogScore(-std::log(prob));
    }

    static NegLogScore fromLogProb(Score logProb) {
        return NegLogScore(-logProb);
    }

    static NegLogScore fromNegLogProb(Score negLogProb) {
        return NegLogScore(negLogProb);
    }

    static NegLogScore probSpaceSum(NegLogScore score1, NegLogScore score2) {
        auto logProb1 = score1.getLogProb();
        auto logProb2 = score2.getLogProb();
        auto maxVal   = std::max(logProb1, logProb2);
        // Perform log-sum-exp trick for numerical stability
        return fromLogProb(maxVal + std::log(std::exp(logProb1 - maxVal) + std::exp(logProb2 - maxVal)));
    }

    static NegLogScore max() {
        return NegLogScore(Core::Type<Score>::max);
    }

    NegLogScore probSpaceSum(NegLogScore other) const {
        return probSpaceSum(*this, other);
    }

    inline Score getProb() const {
        return std::exp(-value);
    }

    inline Score getLogProb() const {
        return -value;
    }

    inline Score getNegLogProb() const {
        return value;
    }

    inline Score scaleScore(Score scale) {
        value *= scale;
        return value;
    }

    inline NegLogScore getScaledScore(Score scale) const {
        return NegLogScore(value * scale);
    }

    Score operator()() const {
        return getNegLogProb();
    }

    NegLogScore operator+(const NegLogScore& other) const {
        return NegLogScore(value + other());
    }

    NegLogScore operator-(const NegLogScore& other) const {
        return NegLogScore(value - other());
    }

    Score operator+=(const NegLogScore& other) {
        value += other();
        return value;
    }

    Score operator-=(const NegLogScore& other) {
        value -= other();
        return value;
    }

    bool operator<(const NegLogScore& other) const {
        return value < other();
    }

    bool operator>(const NegLogScore& other) const {
        return value > other();
    }
};

}  // namespace Nn

#endif /* TYPES_HH_ */
