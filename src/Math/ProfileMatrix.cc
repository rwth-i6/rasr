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
#include <cstdlib>
#include <ctime>
#include <sstream>

#include <Core/Application.hh>
#include <Core/Utility.hh>
#include <Math/CudaMatrix.hh>
#include <Math/CudaWrapper.hh>
#include <Math/Random.hh>
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include "ProfileMatrix.hh"

const Core::ParameterInt MatrixProfiler::paramNumberOfRepetitions("number-of-repetitions", "number of repetitions", 5);

template<typename T>
void MatrixProfiler::initMatrices(Math::FastMatrix<T>& C, Math::CudaMatrix<T>& G, int nRows, int nCols, bool randomInit) const {
    C.resize(nRows, nCols);
    G.resize(nRows, nCols);
    Math::randomSeed(0);
    if (randomInit) {
#pragma omp parallel for
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                C.at(i, j) = (float)(Math::rand() % 10);
                G.at(i, j) = C.at(i, j);
            }
        }
    }
    else {
        C.setToZero();
        G.setToZero();
    }
    G.initComputation();
}

template<typename T>
void MatrixProfiler::resetMatrices(Math::FastMatrix<T>& C, Math::CudaMatrix<T>& G, bool randomInit) const {
    if (randomInit) {
        if (G.isComputing())
            G.finishComputation();
        initMatrices(C, G, C.nRows(), G.nColumns(), randomInit);
    }
    else {
        if (!G.isComputing())
            G.initComputation(false);
        G.setToZero();
        C.setToZero();
    }
}

void MatrixProfiler::logTiming(double timeCpu, double timeGpu) const {
    log() << Core::XmlOpen("timer")
          << Core::XmlFull("time-on-cpu", timeCpu / nRepetitions_)
          << Core::XmlFull("time-on-gpu", timeGpu / nRepetitions_)
          << Core::XmlFull("speedup", timeCpu / timeGpu)
          << Core::XmlClose("timer");
}

MatrixProfiler::MatrixProfiler()
        : Core::Application(),
          nRepetitions_(paramNumberOfRepetitions(config)) {
    setTitle("profile");
}

void MatrixProfiler::profileExp(int nRows, int nCols) const {
    log() << "testing EXP with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> C;
    Math::CudaMatrix<f32> G;
    initMatrices(C, G, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        C.exp();
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        G.exp();
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(C, G);
    }
    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileDot(int nRows, int nCols) const {
    log() << "testing Dot with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> A_cpu;
    Math::CudaMatrix<f32> A_gpu;
    Math::FastMatrix<f32> B_cpu;
    Math::CudaMatrix<f32> B_gpu;
    initMatrices(A_cpu, A_gpu, nRows, nCols);
    initMatrices(B_cpu, B_gpu, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        A_cpu.dot(B_cpu);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        A_gpu.dot(B_gpu);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(B_cpu, B_gpu);
    }
    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileAdd(int nRows, int nCols) const {
    log() << "testing Add with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> A_cpu;
    Math::CudaMatrix<f32> A_gpu;
    Math::FastMatrix<f32> B_cpu;
    Math::CudaMatrix<f32> B_gpu;
    initMatrices(A_cpu, A_gpu, nRows, nCols);
    initMatrices(B_cpu, B_gpu, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        A_cpu.add(B_cpu);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        A_gpu.add(B_gpu);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(B_cpu, B_gpu);
    }
    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileSigmoid(int nRows, int nCols, float gamma) const {
    log() << "testing SIGMOID with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> C;
    Math::CudaMatrix<f32> G;
    initMatrices(C, G, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        C.sigmoid(gamma);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        G.sigmoid(gamma);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(C, G);
    }
    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileSoftmax(int nRows, int nCols) const {
    log() << "testing SOFTMAX with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> C;
    Math::CudaMatrix<f32> G;
    initMatrices(C, G, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        C.softmax();
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        G.softmax();
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(C, G);
    }
    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileMaxOfColumns(int nRows, int nCols) const {
    log() << "testing MAX-OF-COLUMNS with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> C;
    Math::CudaMatrix<f32> G;
    Math::FastVector<f32> v_cpu(nCols);
    Math::CudaVector<f32> v_gpu(nCols);

    initMatrices(C, G, nRows, nCols);
    v_gpu.initComputation(false);
    Math::CudaMatrix<f32> tmp(32, nCols);
    tmp.initComputation(false);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        v_cpu.getMaxOfColumns(C);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        v_gpu.getMaxOfColumns(G, tmp);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(C, G);
    }
    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileAddSummedRows(int nRows, int nCols) const {
    log() << "testing SUM-ROWS with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> C;
    Math::CudaMatrix<f32> G;
    Math::FastVector<f32> v_cpu(nCols);
    v_cpu.setToZero();
    Math::CudaVector<f32> v_gpu(nCols);

    initMatrices(C, G, nRows, nCols);
    v_gpu.initComputation(false);
    v_gpu.setToZero();
    Math::CudaMatrix<f32> tmp(32, nCols);
    tmp.initComputation(false);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        v_cpu.addSummedRows(C);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        v_gpu.addSummedRows(G);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(C, G);
    }
    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileSoftmaxDetailed(int nRows, int nCols, int tmpDimension) const {
    log() << "detailed testing of SOFTMAX with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeTmpCpu = 0.0, timeTmpGpu = 0.0;
    double  timeMaxCpu = 0.0, timeMaxGpu = 0.0;
    double  timeAddCpu = 0.0, timeAddGpu = 0.0;
    double  timeExpCpu = 0.0, timeExpGpu = 0.0;
    double  timeZeroCpu = 0.0, timeZeroGpu = 0.0;
    double  timeAddSummedRowsCpu = 0.0, timeAddSummedRowsGpu = 0.0;
    double  timeScaleCpu = 0.0, timeScaleGpu = 0.0;

    Math::FastMatrix<f32> C;
    Math::CudaMatrix<f32> G;
    initMatrices(C, G, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        // tmp memory allocation
        gettimeofday(&start, NULL);
        Math::FastVector<f32>* tmp_cpu = new Math::FastVector<f32>(nCols);
        gettimeofday(&end, NULL);
        timeTmpCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        Math::CudaVector<f32>* tmp_cuda  = new Math::CudaVector<f32>(nCols);
        Math::CudaMatrix<f32>* tmp2_cuda = new Math::CudaMatrix<f32>(tmpDimension, nCols);
        tmp_cuda->initComputation(false);
        tmp2_cuda->initComputation(false);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeTmpGpu += Core::timeDiff(start, end);

        // get max of columns
        gettimeofday(&start, NULL);
        tmp_cpu->getMaxOfColumns(C);
        gettimeofday(&end, NULL);
        timeMaxCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        tmp_cuda->getMaxOfColumns(G, *tmp2_cuda);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeMaxGpu += Core::timeDiff(start, end);

        // add to all rows
        gettimeofday(&start, NULL);
        C.addToAllRows(*tmp_cpu, -1.0);
        gettimeofday(&end, NULL);
        timeAddCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        G.addToAllRows(*tmp_cuda, -1.0);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeAddGpu += Core::timeDiff(start, end);

        // exp
        gettimeofday(&start, NULL);
        C.exp();
        gettimeofday(&end, NULL);
        timeExpCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        G.exp();
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeExpGpu += Core::timeDiff(start, end);

        // set to zero
        gettimeofday(&start, NULL);
        tmp_cpu->setToZero();
        gettimeofday(&end, NULL);
        timeZeroCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        tmp_cuda->setToZero();
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeZeroGpu += Core::timeDiff(start, end);

        // accumulate entries of each column
        gettimeofday(&start, NULL);
        tmp_cpu->addSummedRows(C);
        gettimeofday(&end, NULL);
        timeAddSummedRowsCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        tmp_cuda->addSummedRows(G, *tmp2_cuda);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeAddSummedRowsGpu += Core::timeDiff(start, end);

        // compute actual softmax output for each column
        gettimeofday(&start, NULL);
        C.divideColumnsByScalars(*tmp_cpu);
        gettimeofday(&end, NULL);
        timeScaleCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        G.divideColumnsByScalars(*tmp_cuda);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeScaleGpu += Core::timeDiff(start, end);

        // free tmp vector
        gettimeofday(&start, NULL);
        delete tmp_cpu;
        gettimeofday(&end, NULL);
        timeTmpCpu += Core::timeDiff(start, end);

        gettimeofday(&start, NULL);
        delete tmp_cuda;
        delete tmp2_cuda;
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeTmpGpu += Core::timeDiff(start, end);

        resetMatrices(C, G);
    }

    double totalTimeCpu = timeTmpCpu + timeMaxCpu + timeAddCpu +
                          timeExpCpu + timeZeroCpu + timeAddSummedRowsCpu + timeScaleCpu;
    double totalTimeGpu = timeTmpGpu + timeMaxGpu + timeAddGpu +
                          timeExpGpu + timeZeroGpu + timeAddSummedRowsGpu + timeScaleGpu;

    log("tmp");
    logTiming(timeTmpCpu, timeTmpGpu);
    log("max-of-columns");
    logTiming(timeMaxCpu, timeMaxGpu);
    log("addToAllRows");
    logTiming(timeAddCpu, timeAddGpu);
    log("exp");
    logTiming(timeExpCpu, timeExpGpu);
    log("zero");
    logTiming(timeZeroCpu, timeZeroGpu);
    log("addSummedRows");
    logTiming(timeAddSummedRowsCpu, timeAddSummedRowsGpu);
    log("scale");
    logTiming(timeScaleCpu, timeScaleGpu);

    log() << Core::XmlOpen("CPU")
          << Core::XmlFull("tmp", timeTmpCpu / totalTimeCpu)
          << Core::XmlFull("max", timeMaxCpu / totalTimeCpu)
          << Core::XmlFull("addToAllRows", timeAddCpu / totalTimeCpu)
          << Core::XmlFull("exp", timeExpCpu / totalTimeCpu)
          << Core::XmlFull("zero", timeZeroCpu / totalTimeCpu)
          << Core::XmlFull("addSummedRows", timeAddSummedRowsCpu / totalTimeCpu)
          << Core::XmlFull("scale", timeScaleCpu / totalTimeCpu)
          << Core::XmlClose("CPU");

    log() << Core::XmlOpen("GPU")
          << Core::XmlFull("tmp", timeTmpGpu / totalTimeGpu)
          << Core::XmlFull("max", timeMaxGpu / totalTimeGpu)
          << Core::XmlFull("addToAllRows", timeAddGpu / totalTimeGpu)
          << Core::XmlFull("exp", timeExpGpu / totalTimeGpu)
          << Core::XmlFull("zero", timeZeroGpu / totalTimeGpu)
          << Core::XmlFull("addSummedRows", timeAddSummedRowsGpu / totalTimeGpu)
          << Core::XmlFull("scale", timeScaleGpu / totalTimeGpu)
          << Core::XmlClose("GPU");

    log("total-speedup:") << totalTimeCpu / totalTimeGpu;
}

void MatrixProfiler::profileMatrixMultiplication(int m, int n, int k) const {
    log() << "testing MATRIX-MULTIPLICATION with dimensions: " << m << " x " << k << " x " << n;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> A_cpu;
    Math::CudaMatrix<f32> A_gpu;
    Math::FastMatrix<f32> B_cpu;
    Math::CudaMatrix<f32> B_gpu;
    Math::FastMatrix<f32> C_cpu;
    Math::CudaMatrix<f32> C_gpu;

    initMatrices(A_cpu, A_gpu, m, k);
    initMatrices(B_cpu, B_gpu, k, n);
    initMatrices(C_cpu, C_gpu, m, n);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        C_cpu.addMatrixProduct(A_cpu, B_cpu);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        C_gpu.addMatrixProduct(A_gpu, B_gpu);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(C_cpu, C_gpu);
    }

    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileMatrixMultiplicationMixed(int m, int n, int k) const {
    log() << "testing MIXED PRECISION MATRIX MULTIPLICATION with dimensions: " << m << " x " << k << " x " << n;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> A_cpu;
    Math::CudaMatrix<f32> A_gpu;
    Math::FastMatrix<f32> B_cpu;
    Math::CudaMatrix<f32> B_gpu;
    Math::FastMatrix<f64> C_cpu;
    Math::CudaMatrix<f64> C_gpu;

    initMatrices(A_cpu, A_gpu, m, k);
    initMatrices(B_cpu, B_gpu, k, n);
    initMatrices(C_cpu, C_gpu, m, n);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        C_cpu.addMatrixProduct(A_cpu, B_cpu);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        C_gpu.addMatrixProduct(A_gpu, B_gpu);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(C_cpu, C_gpu);
    }

    logTiming(timeCpu, timeGpu);
}

void MatrixProfiler::profileSync(int nRows, int nCols) const {
    log() << "testing SYNC with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeGpu = 0.0;

    Math::FastMatrix<f32> A_cpu;
    Math::CudaMatrix<f32> A_gpu;

    initMatrices(A_cpu, A_gpu, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        A_gpu.initComputation();
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(A_cpu, A_gpu);
        A_gpu.finishComputation();
    }

    logTiming(0, timeGpu);
}

void MatrixProfiler::profileCast(int nRows, int nCols) const {
    log() << "testing CAST with dimensions: " << nRows << " x " << nCols;
    timeval start, end;
    double  timeCpu = 0.0, timeGpu = 0.0;

    Math::FastMatrix<f32> A_cpu;
    Math::CudaMatrix<f32> A_gpu;
    Math::FastMatrix<f32> D_cpu;
    Math::CudaMatrix<f32> D_gpu;

    initMatrices(A_cpu, A_gpu, nRows, nCols);
    initMatrices(D_cpu, D_gpu, nRows, nCols);

    for (u32 run = 0; run < nRepetitions_; run++) {
        gettimeofday(&start, NULL);
        D_cpu.copy(A_cpu);
        gettimeofday(&end, NULL);
        timeCpu += Core::timeDiff(start, end);
        gettimeofday(&start, NULL);
        D_gpu.copy(A_gpu);
        Math::Cuda::deviceSync();  // need to synchronize device !
        gettimeofday(&end, NULL);
        timeGpu += Core::timeDiff(start, end);
        resetMatrices(A_cpu, A_gpu);
        resetMatrices(D_cpu, D_gpu);
    }

    logTiming(timeCpu, timeGpu);
}

int MatrixProfiler::main(const std::vector<std::string>& arguments) {
    // id of test
    std::string id;
    log("measuring time of GPU and CPU implementations, using CPU math library: ") << Math::getMathLibrary();
    log("averaging over ") << nRepetitions_ << " runs";
    if (arguments.size() > 0)
        id = arguments.at(0);
    else
        id = "exp";
    if (id == "exp") {
        int nRows = 2048, nCols = 2048;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        profileExp(nRows, nCols);
    }
    else if (id == "sigmoid") {
        int   nRows = 2048, nCols = 2048;
        float gamma = 1.0;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        if (arguments.size() > 3)
            std::istringstream(arguments.at(2)) >> gamma;
        profileSigmoid(nRows, nCols, gamma);
    }
    else if (id == "max-of-columns") {
        int   nRows = 2048, nCols = 2048;
        float gamma = 1.0;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        if (arguments.size() > 3)
            std::istringstream(arguments.at(2)) >> gamma;
        profileMaxOfColumns(nRows, nCols);
    }
    else if (id == "sum-rows") {
        int   nRows = 2048, nCols = 2048;
        float gamma = 1.0;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        if (arguments.size() > 3)
            std::istringstream(arguments.at(2)) >> gamma;
        profileAddSummedRows(nRows, nCols);
    }
    else if (id == "softmax") {
        int nRows = 2048, nCols = 2048;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        profileSoftmax(nRows, nCols);
    }
    else if (id == "softmax-detailed") {
        int nRows = 2048, nCols = 2048, tmpDimension = 32;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        if (arguments.size() > 3)
            std::istringstream(arguments.at(3)) >> tmpDimension;

        profileSoftmaxDetailed(nRows, nCols, tmpDimension);
    }
    else if (id == "matrix-mult") {
        int m = 2048, n = 2048, k = 2048;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> m;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> n;
        if (arguments.size() > 3)
            std::istringstream(arguments.at(3)) >> k;

        profileMatrixMultiplication(m, n, k);
    }
    else if (id == "matrix-mult-mixed") {
        int m = 2048, n = 2048, k = 2048;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> m;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> n;
        if (arguments.size() > 3)
            std::istringstream(arguments.at(3)) >> k;

        profileMatrixMultiplicationMixed(m, n, k);
    }
    else if (id == "sync") {
        int nRows = 2048, nCols = 2048;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        profileSync(nRows, nCols);
    }
    else if (id == "cast") {
        int nRows = 2048, nCols = 2048;
        if (arguments.size() > 1)
            std::istringstream(arguments.at(1)) >> nRows;
        if (arguments.size() > 2)
            std::istringstream(arguments.at(2)) >> nCols;
        profileCast(nRows, nCols);
    }

    return 0;
}

/*===========================================================================*/
// explicit template instantiation
template void MatrixProfiler::initMatrices<f32>(Math::FastMatrix<f32>&, Math::CudaMatrix<f32>&, int, int, bool) const;
template void MatrixProfiler::initMatrices<f64>(Math::FastMatrix<f64>&, Math::CudaMatrix<f64>&, int, int, bool) const;
template void MatrixProfiler::resetMatrices<f32>(Math::FastMatrix<f32>&, Math::CudaMatrix<f32>&, bool) const;
template void MatrixProfiler::resetMatrices<f64>(Math::FastMatrix<f64>&, Math::CudaMatrix<f64>&, bool) const;

APPLICATION(MatrixProfiler)
