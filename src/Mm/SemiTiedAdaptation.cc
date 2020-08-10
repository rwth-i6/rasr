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
#include "SemiTiedAdaptation.hh"
#include <Math/Lapack/Svd.hh>
#include <Math/Nr/ConjugateGradient.hh>
#include "MixtureSet.hh"

using namespace Mm;
using namespace Math::Nr;
using namespace Math::Lapack;

const Core::ParameterFloat SemiTiedEstimator::paramMinSemiTiedAdaptationObservations_(
        "semi-min-observation",
        "minimum number of observations for semi-tied-MLLR estimation",
        100, 1);

const Core::ParameterFloat SemiTiedEstimator::paramIterationStop_(
        "iteration-stop",
        "tolerance for conjugate gradients in semi-tied-MLLR estimation",
        1.0e-5);

SemiTiedEstimator::SemiTiedEstimator(const Core::Configuration&          c,
                                     Core::Ref<const MixtureSet>         m,
                                     const Core::Ref<Am::AdaptationTree> adaptationTree)
        : FullAdaptorViterbiEstimator(c, m, adaptationTree),
          stopCriterion_(paramIterationStop_(config)) {
    log("semi-tied MLLR estimation iteration theshold ") << stopCriterion_;
    minSemiTiedAdaptationObservations_ = paramMinSemiTiedAdaptationObservations_(config);
    log("minimum number of observations for semi-tied MLLR estimation: ") << minSemiTiedAdaptationObservations_;
};

SemiTiedEstimator::SemiTiedEstimator(const Core::Configuration&          c,
                                     ComponentIndex                      dim,
                                     const Core::Ref<Am::AdaptationTree> adaptationTree)
        : FullAdaptorViterbiEstimator(c, dim, adaptationTree),
          stopCriterion_(paramIterationStop_(config)) {
    minSemiTiedAdaptationObservations_ = paramMinSemiTiedAdaptationObservations_(config);
    //no logging because only used for clone()
};

SemiTiedEstimator::SemiTiedEstimator(const Core::Configuration&          c,
                                     const Core::Ref<Am::AdaptationTree> adaptationTree)
        : FullAdaptorViterbiEstimator(c, adaptationTree){};

void SemiTiedEstimator::estimateWMatrices() {
#ifdef __semi_tied_one_base__
    log("semi tied one base");
#endif

    FullAdaptorViterbiEstimator::estimateWMatrices();

    NodeId           id, qId;
    NodeIdToIdSetMap tyingScheme;

    for (Core::BinaryTree::LeafList::const_iterator p = tree_->leafList().begin();
         p != tree_->leafList().end(); ++p) {
        id = tree_->id(p);

        while (id != tree_->root() && count_[id] <= minSemiTiedAdaptationObservations_) {
            id = tree_->previous(id);
        }
        //get corresponding matrix

        qId = id;

        MatrixConstIterator pit = w_.find(qId);
        while (pit == w_.end() && qId != tree_->root()) {
            qId = tree_->previous(qId);
            pit = w_.find(qId);
        }
#ifdef __semi_tied_one_base__
        qId = tree_->root();
#endif
        if (qId == tree_->root() && (pit == w_.end())) {
            //not enough observations in complete tree
            log("too few observations for base adaptation\n") << minAdaptationObservations_ << " observations needed, " << count_[qId] << " seen.\n"
                                                              << "resetting matrix to unity";
            w_[qId] = adaptationUnitMatrix(dimension_);
            return;
        }
        if (qId != id)
            tyingScheme[qId].insert(id);  //else: qId is already leaf
    }

    for (NodeIdToIdSetMap::iterator p = tyingScheme.begin();
         p != tyingScheme.end(); ++p) {
        if (p->second.size() == 1) {
            qId = p->first;
            id  = p->second.begin()->id;
            if (tree_->left(qId) == id) {
                p->second.insert(IdSetEntry(tree_->right(qId), false));
            }
            else {
                p->second.insert(IdSetEntry(tree_->left(qId), false));
            }
        }
    }

    if (adaptationDumpChannel_.isOpen()) {
        adaptationDumpChannel_ << Core::XmlOpen("semi-tied-tying-scheme");
        for (NodeIdToIdSetMap::const_iterator p = tyingScheme.begin();
             p != tyingScheme.end(); ++p) {
            const IdSet& s = p->second;
            adaptationDumpChannel_ << "\n"
                                   << p->first << ": ";
            for (IdSetConstIterator q = s.begin(); q != s.end(); ++q) {
                adaptationDumpChannel_ << q->id << "(" << q->isActive << ") ";
            }
        }
        adaptationDumpChannel_ << Core::XmlClose("semi-tied-tying-scheme");
    }

    Math::Vector<Matrix> zMatrices;
    Math::Vector<Matrix> gMatrices;
    double               maxValue = Core::Type<double>::min;

    {
        Math::Vector<Precursor::ZAccumulator> z;
        Math::Vector<Precursor::GAccumulator> g;
        propagate(leafZAccumulators_, z, tree_->root());
        propagate(leafGAccumulators_, g, tree_->root());
        zMatrices.resize(z.size());
        gMatrices.resize(g.size());

        for (u32 i = 0; i < z.size(); ++i) {
            zMatrices[i] = z[i].squareMatrix();
            gMatrices[i] = g[i].squareMatrix();
            double temp  = zMatrices[i].maxElement();
            if (temp > maxValue)
                maxValue = temp;
            temp = gMatrices[i].maxElement();
            if (temp > maxValue)
                maxValue = temp;
        }
    }

    //normalize all matrices by overall largest element
    double weight = 1.0 / maxValue;
    weight        = 1.0;

    ensure(zMatrices.size() == gMatrices.size());
    for (u32 i = 0; i < zMatrices.size(); ++i) {
        zMatrices[i] *= weight;
        gMatrices[i] *= weight;
    }

    Core::ProgressIndicator pi("estimating MLLR matrices");
    pi.start(tyingScheme.size());
    for (NodeIdToIdSetMap::const_iterator p = tyingScheme.begin();
         p != tyingScheme.end(); ++p) {
        qId = p->first;
        solveEstimationEquations(p->second, qId, zMatrices, gMatrices);
        pi.notify();
    }
    pi.finish();
}

void SemiTiedEstimator::solveEstimationEquations(const IdSet&                idSet,
                                                 NodeId                      qId,
                                                 const Math::Vector<Matrix>& z,
                                                 const Math::Vector<Matrix>& g) {
    std::map<NodeId, Math::Vector<double>> lambda;
    std::map<NodeId, Matrix>               previousMatrices;
    Math::Vector<double>                   bias = w_[qId].column(0);
    w_[qId].removeColumn(0);

#ifdef __dumpMatrix__
    string fileName = "w_init.m";

    std::ofstream file(fileName.c_str(), std::ios::binary);
    w_[qId].print_raw(file, 25);
    file.close();

    for (IdSetConstIterator c = idSet.begin(); c != idSet.end(); ++c) {
        std::ostringstream os;
        os << "g_" << c->id << ".m" << ends;
        file.open(os.str().c_str());
        g[c->id].print_raw(file, 25);
        file.close();
        os.str("");
        os << "z_" << c->id << ".m" << ends;
        file.open(os.str().c_str());
        z[c->id].print_raw(file, 25);
        file.close();
    }
#endif

    Matrix               U = w_[qId], V(dimension_);
    Math::Vector<double> wInit(dimension_, 0.0);

    //get initial values;
    svd(U, wInit, V, w_[qId]);

#ifndef __lambda_only__
    for (IdSetConstIterator c = idSet.begin(); c != idSet.end(); ++c) {
        lambda[c->id] = wInit;
    }

    SemiTiedOptimizationFunction function(dimension_, g, z, lambda);
    SemiTiedOptimizationGradient gradient(dimension_, g, z, lambda);
    int                          iterations;
    double                       returnValue;
    Math::Vector<double>         x = convert2Vector(lambda, U, V);

    frprmn(x, stopCriterion_, iterations, returnValue, function, gradient);  //numerical recipes conjugate gradient
    log("number of conjugate gradient iterations for semi-tied MLLR estimation: ") << iterations;

    convert2Matrices(x, lambda, U, V, dimension_);
#else
    log("lambda only estimation");
    Matrix UT   = U.transpose();
    Matrix UT_U = UT * U;
    Matrix VT   = V.transpose();

    for (IdSetConstIterator c = idSet.begin(); c != idSet.end(); ++c) {
        Matrix               VT_Gc_V      = VT * g[c->id] * V;
        Math::Vector<double> diag_UT_Zc_V = Math::diagonal(UT * z[c->id] * V);
        Matrix               Ac           = Math::multiplyElementwise(UT_U, VT_Gc_V);
        Math::Vector<double> x(diag_UT_Zc_V.size());
        s32                  status = solveLinearLeastSquares(x, Ac, diag_UT_Zc_V);
        if (status != 0)
            error("Lapack routine solveLinearLeastSquares failed! status = ") << status;
        lambda[c->id] = x;
    }

#endif

    for (IdSetConstIterator c = idSet.begin(); c != idSet.end(); ++c) {
        if (c->isActive) {
            w_[c->id] = U * Math::makeDiagonalMatrix(lambda[c->id]) * V.transpose();
            w_[c->id].insertColumn(0, bias);
        }
    }
    w_[qId].insertColumn(0, bias);
}

bool SemiTiedEstimator::write(Core::BinaryOutputStream& o) const {
    if (Precursor::write(o)) {
        o << minSemiTiedAdaptationObservations_ << stopCriterion_;
    }
    return o.good();
}

bool SemiTiedEstimator::read(Core::BinaryInputStream& i) {
    if (Precursor::read(i)) {
        i >> minSemiTiedAdaptationObservations_ >> stopCriterion_;
    }

    // Override from configuration.
    stopCriterion_                     = paramIterationStop_(config, stopCriterion_);
    minSemiTiedAdaptationObservations_ = paramMinSemiTiedAdaptationObservations_(config, minSemiTiedAdaptationObservations_);

    return i.good();
}

////////////////////////////////////////////////////////////////////
// SemiTiedOptimizationFunction
////////////////////////////////////////////////////////////////////
SemiTiedOptimizationFunction::SemiTiedOptimizationFunction(ComponentIndex d, const Math::Vector<Matrix>& G,
                                                           const Math::Vector<Matrix>&                   Z,
                                                           const std::map<NodeId, Math::Vector<double>>& l)
        : dimension_(d), G_(G), Z_(Z), lambda_(l), f1_(dimension_), f2_(dimension_) {}

SemiTiedOptimizationFunction::ResultType SemiTiedOptimizationFunction::operator()(const ArgumentType& x) const {
    convert2Matrices(x, lambda_, U_, V_, dimension_);
    UT_ = U_.transpose();
    f1_.fill(0.0);
    f2_.fill(0.0);

    for (std::map<NodeId, Math::Vector<double>>::const_iterator p = lambda_.begin(); p != lambda_.end(); ++p) {
        NodeId c = p->first;
        AT_      = V_ * Math::makeDiagonalMatrix(lambda_.find(c)->second) * UT_;
        f1_      = f1_ + AT_ * AT_.transpose() * G_[c];
        f2_      = f2_ + AT_ * Z_[c];
    }
    return (f1_.trace() - 2 * f2_.trace());
}

////////////////////////////////////////////////////////////////////
// SemiTiedOptimizationGradient
////////////////////////////////////////////////////////////////////
SemiTiedOptimizationGradient::SemiTiedOptimizationGradient(ComponentIndex                                d,
                                                           const Math::Vector<Matrix>&                   G,
                                                           const Math::Vector<Matrix>&                   Z,
                                                           const std::map<NodeId, Math::Vector<double>>& l)
        : dimension_(d),
          G_(G),
          Z_(Z),
          lambda_(l),
          f1_(dimension_),
          f2_(dimension_) {}

void SemiTiedOptimizationGradient::operator()(const InputType& x, ResultType& f) const {
    f.clear();
    convert2Matrices(x, lambda_, U_, V_, dimension_);

    UT_ = U_.transpose();
    VT_ = V_.transpose();

    f1_.fill(0.0);
    f2_.fill(0.0);
    UT_U_ = UT_ * U_;

    for (std::map<NodeId, Math::Vector<double>>::const_iterator p = lambda_.begin(); p != lambda_.end(); ++p) {
        NodeId c                       = p->first;
        lc_                            = Math::makeDiagonalMatrix(lambda_.find(c)->second);
        const Math::Matrix<double>& Gc = G_[c];
        const Math::Matrix<double>& Zc = Z_[c];
        VT_Gc_V_                       = VT_ * Gc * V_;

        f0_[c] = 2.0 * Math::diagonal(UT_U_ * lc_ * VT_Gc_V_ - UT_ * Zc * V_);
        f1_    = f1_ + U_ * lc_ * VT_Gc_V_ * lc_ - Zc * V_ * lc_;
        f2_    = f2_ + Gc * V_ * lc_ * UT_U_ * lc_ - Zc.transpose() * U_ * lc_;
    }
    f = convert2Vector(f0_, 2.0 * f1_, 2.0 * f2_);
}

void Mm::convert2Matrices(const Math::Vector<double>& x, std::map<NodeId, Math::Vector<double>>& l,
                          Math::Matrix<double>& u, Math::Matrix<double>& v, ComponentIndex dim) {
    div_t s1 = div((int)x.size(), (int)dim);
    require(s1.rem == 0);

    size_t nClasses = s1.quot - 2 * dim;
    require(nClasses == l.size());

    u32 c = 0;
    for (std::map<NodeId, Math::Vector<double>>::iterator p = l.begin(); p != l.end(); ++p) {
        p->second = Math::Vector<double>(x.begin() + c * dim, x.begin() + c * dim + dim);
        ++c;
    }

    size_t i = c * dim;
    if (u.nRows() != dim)
        u.resize(dim);

    for (size_t col = 0; col < u.nColumns(); ++col) {
        for (size_t row = 0; row < u.nRows(); row++) {
            u[row][col] = x[i];
            ++i;
        }
    }
    ensure(i - 1 < x.size());

    if (v.nRows() != dim)
        v.resize(dim);
    for (size_t col = 0; col < v.nColumns(); ++col) {
        for (size_t row = 0; row < v.nRows(); row++) {
            v[row][col] = x[i];
            ++i;
        }
    }
    ensure(i - 1 < x.size());
}

Math::Vector<double> Mm::convert2Vector(const std::map<NodeId, Math::Vector<double>>& l,
                                        const Math::Matrix<double>&                   u,
                                        const Math::Matrix<double>&                   v) {
    Math::Vector<double> result;

    for (std::map<NodeId, Math::Vector<double>>::const_iterator p = l.begin(); p != l.end(); ++p) {
        result.insert(result.end(), p->second.begin(), p->second.end());
    }

    for (size_t col = 0; col < u.nColumns(); ++col) {
        for (size_t row = 0; row < u.nRows(); ++row) {
            result.push_back(u[row][col]);
        }
    }

    for (size_t col = 0; col < v.nColumns(); ++col) {
        for (size_t row = 0; row < v.nRows(); ++row) {
            result.push_back(v[row][col]);
        }
    }

    return result;
}
