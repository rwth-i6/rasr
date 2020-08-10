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
#include "BandMllrAdaptation.hh"
#include <Math/Lapack/MatrixTools.hh>
#include "MixtureSet.hh"

using namespace Mm;
//using namespace Math::Lapack;

const Core::ParameterInt BandMllrEstimator::paramNBands_(
        "mllr-bands",
        "number of bands used in diagonal MLLR adaptation",
        1, 0);

BandMllrEstimator::BandMllrEstimator(const Core::Configuration&          c,
                                     Core::Ref<const Mm::MixtureSet>     m,
                                     const Core::Ref<Am::AdaptationTree> adaptationTree)
        : FullAdaptorViterbiEstimator(c, m, adaptationTree),
          nBands_(paramNBands_(config)) {
    log("number of bands for band-diagonal MLLR adaptation ") << nBands_;
}

BandMllrEstimator::BandMllrEstimator(const Core::Configuration&          c,
                                     ComponentIndex                      dim,
                                     const Core::Ref<Am::AdaptationTree> adaptationTree)
        : FullAdaptorViterbiEstimator(c, dim, adaptationTree),
          nBands_(paramNBands_(config)) {
    //no logging because only used for clone()
}

BandMllrEstimator::BandMllrEstimator(const Core::Configuration&          c,
                                     const Core::Ref<Am::AdaptationTree> adaptationTree)
        : FullAdaptorViterbiEstimator(c, adaptationTree),
          nBands_(paramNBands_(config)) {
    //no logging because only used for clone()
}

void BandMllrEstimator::estimateWMatrices() {
    Math::Vector<Precursor::ZAccumulator> z;
    Math::Vector<Precursor::GAccumulator> g;

    propagate(leafZAccumulators_, z, tree_->root());
    propagate(leafGAccumulators_, g, tree_->root());

    Math::Vector<Matrix::Type> resultRow;

    ensure(z.size() == g.size());
    count_.resize(z.size());
    for (u32 id = 0; id < z.size(); ++id) {  //loop over all nodes of tree_
        count_[id] = g[id].count();
        w_[id].resize(dimension_, dimension_ + 1);
        if (count_[id] > minAdaptationObservations_) {
            //solve MLLR equation row-wise
            for (u32 row = 0; row < dimension_; ++row) {
                resultRow   = solveRowEquation(g[id].matrix(), z[id].matrix(), row);
                w_[id][row] = resultRow;
            }
        }
        else
            w_.erase(id);
    }
    return;
}

Math::Vector<Matrix::Type> BandMllrEstimator::solveRowEquation(
        const Matrix& g,
        const Matrix& z,
        u32           wRow) {
    u16                        nRows  = 2 * nBands_ + 1;
    const s16                  offset = wRow - nBands_;  //originally wRow-nBands_-1, but nRow is starts from 0
    Math::Vector<Matrix::Type> result(dimension_ + 1, 0.0);
    Math::Vector<Matrix::Type> x(nRows + 1);
    Math::Vector<Matrix::Type> c(nRows + 1);
    Matrix                     B(nRows + 1, nRows + 1);

    for (u16 row = 1; row < nRows + 1; ++row) {
        if (row + offset > 0 && u16(row + offset) < dimension_ + 1) {
            B[row][0] = g[row + offset][0];
            B[0][row] = g[0][row + offset];  //put here for efficiency, row=col
            for (u16 col = 1; col < nRows + 1; ++col) {
                if (col + offset > 0 && u16(col + offset) < dimension_ + 1) {
                    B[row][col] = g[row + offset][col + offset];
                }
            }
            c[row] = z[wRow][row + offset];
        }
    }

    c[0]    = z[wRow][0];
    B[0][0] = g[0][0];

    Matrix BInverse(B);
    Math::Lapack::pseudoInvert(BInverse);
    x = BInverse * c;
    for (u16 row = 1; row < nRows + 1; ++row) {
        if (row + offset > 0 && u16(row + offset) < dimension_ + 1) {
            result[row + offset] = x[row];
        }
    }
    result[0] = x[0];
    return result;
}

bool BandMllrEstimator::write(Core::BinaryOutputStream& o) const {
    if (Precursor::write(o)) {
        o << nBands_;
    }
    return o.good();
}

bool BandMllrEstimator::read(Core::BinaryInputStream& i) {
    if (Precursor::read(i)) {
        i >> nBands_;
    }
    nBands_ = paramNBands_(config, nBands_);  // Override from configuration
    return i.good();
}
