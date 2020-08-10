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
#include <Math/FastMatrix.hh>
#include <Math/FastVector.hh>
#include <Math/LinearConjugateGradient.hh>
#include <Test/UnitTest.hh>

template<typename T>
class ExplicitConjugateGradient : public Math::LinearConjugateGradient<Math::FastVector<T>, T> {
    typedef Math::LinearConjugateGradient<Math::FastVector<T>, T> Precursor;

public:
    using Math::LinearConjugateGradient<Math::FastVector<T>, T>::initializeCg;
    using Math::LinearConjugateGradient<Math::FastVector<T>, T>::rhs_;
    using Math::LinearConjugateGradient<Math::FastVector<T>, T>::iterate_;

protected:
    Math::FastMatrix<T> matrix_;

public:
    ExplicitConjugateGradient()
            : Math::LinearConjugateGradient<Math::FastVector<T>, T>() {}

    virtual void allocate(const Math::FastVector<T>& vector) {
        Math::LinearConjugateGradient<Math::FastVector<T>, T>::allocate(vector);
    }

    void setMatrix(Math::FastMatrix<T>& matrix) {
        require(matrix.nRows() == matrix.nColumns());
        matrix_.swap(matrix);
    }

    void setPreconditioner(Math::FastVector<T>& diagonalVector) {
        require(diagonalVector.size() == matrix_.nRows());
        Math::DiagonalCgPreconditioner<Math::FastVector<T>, T>* preconditioner = new Math::DiagonalCgPreconditioner<Math::FastVector<T>, T>();
        preconditioner->swap(diagonalVector);
        Precursor::setPreconditioner(preconditioner);
    }

    // not efficient, since symmetry is not exploited, but ok for testing purposes
    virtual void applyMatrix(const Math::FastVector<T>& in, Math::FastVector<T>& out) {
        matrix_.multiply(in, out);
    }
};

class TestLinearConjugateGradient : public Test::ConfigurableFixture {
public:
    void setUp();
    void tearDown();

protected:
    int                             dim_;
    Math::FastMatrix<f64>           matrix_;
    Math::FastVector<f64>           preconditioner_;
    Math::FastVector<f64>           initialization_;
    Math::FastVector<f64>           rhs_;
    Math::FastVector<f64>           solution_;
    ExplicitConjugateGradient<f64>* solver_;
};

void TestLinearConjugateGradient::setUp() {
    setParameter("*.unbuffered", "true");
    dim_ = 2;
    matrix_.resize(dim_, dim_);
    matrix_.at(0, 0) = 2.0;
    matrix_.at(0, 1) = 1.0;
    matrix_.at(1, 0) = 1.0;
    matrix_.at(1, 1) = 4.0;
    preconditioner_.resize(dim_);
    preconditioner_.at(0) = 2.0;
    preconditioner_.at(1) = 4.0;
    initialization_.resize(2);
    initialization_.at(0) = -1.0;
    initialization_.at(1) = 1.0;
    rhs_.resize(2);
    rhs_.at(0) = 7.0;
    rhs_.at(1) = 14.0;
    solution_.resize(2);
    solution_.fill(1);
    solver_ = new ExplicitConjugateGradient<f64>();
    setParameter("*.channel", "/dev/null");
}

void TestLinearConjugateGradient::tearDown() {
    delete solver_;
}

TEST_F(Test, TestLinearConjugateGradient, solvefromzero) {
    solver_->configuration.maxIterations_                            = 1000;
    solver_->configuration.terminateBasedOnResidualNorm_             = true;
    solver_->configuration.terminateBasedOnAverageObjectiveFunction_ = false;
    solver_->configuration.residualTolerance_                        = 0.0;
    solver_->configuration.verbosity_                                = 0;

    solver_->allocate(rhs_);
    solver_->setMatrix(matrix_);

    Math::FastVector<f64> emptyinit;
    u32                   nIterations = 0;
    solver_->solve(rhs_, emptyinit, solution_, nIterations);

    EXPECT_DOUBLE_EQ(solution_.at(0), 2.0, 0.000000001);
    EXPECT_DOUBLE_EQ(solution_.at(1), 3.0, 0.000000001);

    f64 obj = solver_->getCgObjectivefunction();
    EXPECT_DOUBLE_EQ(obj, -28.0, 0.000000001);
}

TEST_F(Test, TestLinearConjugateGradient, solvefromnonzero) {
    solver_->configuration.maxIterations_                            = 1000;
    solver_->configuration.terminateBasedOnResidualNorm_             = true;
    solver_->configuration.terminateBasedOnAverageObjectiveFunction_ = false;
    solver_->configuration.residualTolerance_                        = 0.0;
    solver_->configuration.verbosity_                                = 0;

    solver_->allocate(rhs_);
    solver_->setMatrix(matrix_);

    u32 nIterations = 0;
    solver_->solve(rhs_, initialization_, solution_, nIterations);
    EXPECT_DOUBLE_EQ(solution_.at(0), 2.0, 0.000000001);
    EXPECT_DOUBLE_EQ(solution_.at(1), 3.0, 0.000000001);

    f64 obj = solver_->getCgObjectivefunction();
    EXPECT_DOUBLE_EQ(obj, -28.0, 0.000000001);
}

TEST_F(Test, TestLinearConjugateGradient, objectiveFunction) {
    solver_->configuration.verbosity_ = 0;

    solver_->allocate(rhs_);
    solver_->setMatrix(matrix_);

    solver_->rhs_     = &rhs_;
    solver_->iterate_ = &solution_;
    solver_->initializeCg(initialization_);
    f64 obj = solver_->getCgObjectivefunction();
    // direct computation of objective function
    Math::FastVector<f64> Ax;
    Ax.copyStructure(initialization_);
    solver_->applyMatrix(initialization_, Ax);
    f64 objTest = 0.5 * initialization_.dot(Ax);
    objTest -= rhs_.dot(initialization_);
    EXPECT_EQ(objTest, obj);
    EXPECT_EQ(-5.0, obj);
}

TEST_F(Test, TestLinearConjugateGradient, PCGsolvefromzero) {
    solver_->configuration.maxIterations_                            = 1000;
    solver_->configuration.terminateBasedOnResidualNorm_             = true;
    solver_->configuration.terminateBasedOnAverageObjectiveFunction_ = false;
    solver_->configuration.residualTolerance_                        = 0.0;
    solver_->configuration.verbosity_                                = 0;
    solver_->configuration.usePreconditioning_                       = true;

    solver_->allocate(rhs_);
    solver_->setMatrix(matrix_);
    solver_->setPreconditioner(preconditioner_);

    Math::FastVector<f64> emptyinit;
    u32                   nIterations = 0;
    solver_->solve(rhs_, emptyinit, solution_, nIterations);
    EXPECT_DOUBLE_EQ(solution_.at(0), 2.0, 0.000000001);
    EXPECT_DOUBLE_EQ(solution_.at(1), 3.0, 0.000000001);

    f64 obj = solver_->getCgObjectivefunction();
    EXPECT_DOUBLE_EQ(obj, -28.0, 0.000000001);
}

TEST_F(Test, TestLinearConjugateGradient, PCGsolvefromnonzero) {
    solver_->configuration.maxIterations_                            = 1000;
    solver_->configuration.terminateBasedOnResidualNorm_             = true;
    solver_->configuration.terminateBasedOnAverageObjectiveFunction_ = false;
    solver_->configuration.residualTolerance_                        = 0.0;
    solver_->configuration.verbosity_                                = 0;
    solver_->configuration.usePreconditioning_                       = true;

    solver_->allocate(rhs_);
    solver_->setMatrix(matrix_);
    solver_->setPreconditioner(preconditioner_);

    u32 nIterations = 0;
    solver_->solve(rhs_, initialization_, solution_, nIterations);
    EXPECT_DOUBLE_EQ(solution_.at(0), 2.0, 0.000000001);
    EXPECT_DOUBLE_EQ(solution_.at(1), 3.0, 0.000000001);

    f64 obj = solver_->getCgObjectivefunction();
    EXPECT_DOUBLE_EQ(obj, -28.0, 0.000000001);
}
