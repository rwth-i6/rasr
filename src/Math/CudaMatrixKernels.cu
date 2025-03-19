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
#include <cuda_runtime.h>
#include <math_constants.h>
#include "CudaMatrixKernels.hh"
#include "stdio.h"

#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#endif

#define THREADS_PER_BLOCK 1024

/*
 *
 *  mixed precision axpy
 *
 */

__global__ void __cuda_axpy(int nElements, float alpha, const float* x, double* y) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        y[index] += alpha * x[index];
}

void _cuda_axpy(int nElements, float alpha, const float* x, double* y) {
    int gridSize = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    __cuda_axpy<<<gridSize, THREADS_PER_BLOCK>>>(nElements, alpha, x, y);
}

__global__ void __cuda_axpy(int nElements, double alpha, const double* x, float* y) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        y[index] += alpha * x[index];
}

void _cuda_axpy(int nElements, double alpha, const double* x, float* y) {
    int gridSize = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    __cuda_axpy<<<gridSize, THREADS_PER_BLOCK>>>(nElements, alpha, x, y);
}

__global__ void __cuda_cast(int nElements, const float* x, double* y) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        y[index] = x[index];
}

void _cuda_cast(int nElements, const float* x, double* y) {
    int gridSize = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    __cuda_cast<<<gridSize, THREADS_PER_BLOCK>>>(nElements, x, y);
}

/*
 *
 *  exp
 *
 */
template<typename T>
__global__ void __cuda_exp(T* data, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = exp(data[index]);
}

template<typename T>
void _cuda_exp(T* data, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_exp<<<gridSize, THREADS_PER_BLOCK>>>(data, nElements);
}

template __global__ void __cuda_exp<float>(float*, unsigned int);
template __global__ void __cuda_exp<double>(double*, unsigned int);
template void            _cuda_exp<float>(float*, unsigned int, unsigned int);
template void            _cuda_exp<double>(double*, unsigned int, unsigned int);

/*
 *
 *  log
 *
 */

template<typename T>
__global__ void __cuda_log(T* data, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = log(data[index]);
}

template<typename T>
void _cuda_log(T* data, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_log<<<gridSize, THREADS_PER_BLOCK>>>(data, nElements);
}

template __global__ void __cuda_log<float>(float*, unsigned int);
template __global__ void __cuda_log<double>(double*, unsigned int);
template void            _cuda_log<float>(float*, unsigned int, unsigned int);
template void            _cuda_log<double>(double*, unsigned int, unsigned int);

/*
 *
 *  pow
 *
 */

template<typename T>
__global__ void __cuda_pow(T* data, unsigned int nElements, T exponent) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = pow(data[index], exponent);
}

template<typename T>
void _cuda_pow(T* data, unsigned int nRows, unsigned int nColumns, T exponent) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    __cuda_pow<<<gridSize, THREADS_PER_BLOCK>>>(data, nElements, exponent);
}

template void _cuda_pow<float>(float*, unsigned int, unsigned int, float);
template void _cuda_pow<double>(double*, unsigned int, unsigned int, double);

/*
 *
 * tanh
 *
 *
 */

template<typename T>
__global__ void __cuda_tanh(T* data, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = tanh(data[index]);
}

template<typename T>
void _cuda_tanh(T* data, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_tanh<<<gridSize, THREADS_PER_BLOCK>>>(data, nElements);
}

template __global__ void __cuda_tanh<float>(float*, unsigned int);
template __global__ void __cuda_tanh<double>(double*, unsigned int);
template void            _cuda_tanh<float>(float*, unsigned int, unsigned int);
template void            _cuda_tanh<double>(double*, unsigned int, unsigned int);

/*
 *
 * sigmoid
 *
 */

template<typename T>
__global__ void __cuda_sigmoid1(T* data, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = 1.0 / (1.0 + exp(-data[index]));
}

template<typename T>
__global__ void __cuda_sigmoid(T gamma, T* data, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = 1.0 / (1.0 + exp(-gamma * data[index]));
}

template<typename T>
void _cuda_sigmoid(T gamma, T* data, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    if (gamma == 1.0)
        __cuda_sigmoid1<<<gridSize, THREADS_PER_BLOCK>>>(data, nElements);
    else
        __cuda_sigmoid<<<gridSize, THREADS_PER_BLOCK>>>(gamma, data, nElements);
}

template void            _cuda_sigmoid<double>(double gamma, double* data, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_sigmoid<double>(double gamma, double* data, unsigned int nElements);
template __global__ void __cuda_sigmoid1<double>(double* data, unsigned int nElements);
template void            _cuda_sigmoid<float>(float gamma, float* data, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_sigmoid<float>(float gamma, float* data, unsigned int nElements);
template __global__ void __cuda_sigmoid1<float>(float* data, unsigned int nElements);

/*
 *
 * softmax
 *
 */

template<typename T>
__global__ void __cuda_softmax(T* data, unsigned int nRows, unsigned int nColumns) {
    unsigned int column        = blockIdx.x;
    unsigned int tid           = threadIdx.x;
    unsigned int blocksize     = (unsigned int)ceil((float)nRows / blockDim.x);  // e.g. 4501/1024 = 5
    unsigned int max_thread_id = (unsigned int)floor((float)nRows / blocksize);  // e.g. 4501/5    = 900

    volatile __shared__ T tmp[THREADS_PER_BLOCK];
    volatile __shared__ T max;
    tmp[tid] = -9999999999;
    T val;

    //// step 1: find maximum in the column
    // each thread finds a maximum in its own "block"
    if (column < nColumns && tid <= max_thread_id) {
        uint beginCol = column * nRows;
        for (uint i = tid * blocksize; i < (tid + 1) * blocksize; ++i) {
            if (i >= nRows)
                break;
            val = data[beginCol + i];
            if (val > tmp[tid])
                tmp[tid] = val;
        }
    }
    __syncthreads();

    // max-reduction
    for (uint s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (tmp[tid + s] > tmp[tid])
                tmp[tid] = tmp[tid + s];
        }
        __syncthreads();
    }

    // one thread stores the maximum in the shared memory
    if (tid == 0 && column < nColumns) {
        max = tmp[0];
    }
    __syncthreads();
    tmp[tid] = 0;

    //// step 2: subtract max from each value and store the sum of
    ////         exp(x-max) in the shared memory
    if (column < nColumns && tid <= max_thread_id) {
        uint beginCol = column * nRows;
        for (uint i = tid * blocksize; i < (tid + 1) * blocksize; ++i) {
            if (i >= nRows)
                break;
            val                = exp(data[beginCol + i] - max);
            data[beginCol + i] = val;
            tmp[tid] += val;
        }
    }
    __syncthreads();

    // sum-reduction; the result is in tmp[0] (softmax normalization)
    for (uint s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            tmp[tid] += tmp[tid + s];
        }
        __syncthreads();
    }

    //// step 3: normalize each entry in the column
    if (column < nColumns && tid <= max_thread_id) {
        uint beginCol = column * nRows;
        for (uint i = tid * blocksize; i < (tid + 1) * blocksize; ++i) {
            if (i >= nRows)
                break;
            data[beginCol + i] /= tmp[0];
        }
    }
}

template<typename T>
void _cuda_softmax(T* data, unsigned int nRows, unsigned int nColumns) {
    __cuda_softmax<<<nColumns, THREADS_PER_BLOCK>>>(data, nRows, nColumns);
}

template __global__ void __cuda_softmax(double* data, unsigned int nRows, unsigned int nColumns);
template void            _cuda_softmax(double* data, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_softmax(float* data, unsigned int nRows, unsigned int nColumns);
template void            _cuda_softmax(float* data, unsigned int nRows, unsigned int nColumns);

/*
 *
 * addSummedRows
 *
 */
template<typename T>
__global__ void __cuda_addSummedRows(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale) {
    unsigned int columnIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (columnIndex < nColumns) {
        float result = 0.0;
        for (unsigned int i = 0; i < nRows; i++) {
            // result += matrix(i,columnIndex)
            result += matrixDevPtr[columnIndex * nRows + i];
        }
        vectorDevPtr[columnIndex] += scale * result;
    }
}

template<typename T>
void _cuda_addSummedRows(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale) {
    // parallelize over columns
    int gridSize = (int)ceil((float)nColumns / THREADS_PER_BLOCK);

    __cuda_addSummedRows<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale);
}

template __global__ void __cuda_addSummedRows(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const double scale);
template void            _cuda_addSummedRows(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const double scale);
template __global__ void __cuda_addSummedRows(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);
template void            _cuda_addSummedRows(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);

/*
 * slightly faster version using tmp array
 *
 */
template<typename T>
__global__ void __cuda_summedRowsTmp(const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                     T* tmpDevPtr, unsigned int tmpRows) {
    unsigned int columnIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int columnPart  = blockIdx.y;
    if (columnIndex < nColumns) {
        unsigned int nRowsDiv = nRows / tmpRows;
        unsigned int startRow = columnPart * nRowsDiv;
        if (startRow < nRows) {
            unsigned int endRow = columnPart == tmpRows - 1 ? nRows : (columnPart + 1) * nRowsDiv;
            T            result = 0.0;
            for (unsigned int i = startRow; i < endRow; i++) {
                // result += matrix(i, columnIndex)
                result += matrixDevPtr[columnIndex * nRows + i];
            }
            tmpDevPtr[columnIndex * tmpRows + columnPart] = result;
        }
    }
}

template<typename T>
void _cuda_addSummedRows(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                         T* tmpDevPtr, unsigned int tmpRows, const T scale) {
    int  gridDimx = (int)ceil((float)nColumns / THREADS_PER_BLOCK);
    int  gridDimy = tmpRows;
    dim3 gridSize(gridDimx, gridDimy);
    __cuda_summedRowsTmp<<<gridSize, THREADS_PER_BLOCK>>>(matrixDevPtr, nRows, nColumns, tmpDevPtr, tmpRows);

    _cuda_addSummedRows<T>(vectorDevPtr, tmpDevPtr, tmpRows, nColumns, scale);
}

template __global__ void __cuda_summedRowsTmp<double>(const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                                      double* tmpDevPtr, unsigned int tmpRows);
template void            _cuda_addSummedRows<double>(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                                     double* tmpDevPtr, unsigned int tmpRows, const double scale);
template __global__ void __cuda_summedRowsTmp<float>(const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                                     float* tmpDevPtr, unsigned int tmpRows);
template void            _cuda_addSummedRows<float>(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                                    float* tmpDevPtr, unsigned int tmpRows, const float scale);
/*
 *
 * addSummedColumns
 *
 */

template<typename T, typename S>
__global__ void __cuda_addSummedColumns(T* vectorDevPtr, const S* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const S scale) {
    unsigned int rowIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowIndex < nRows) {
        T result = 0.0;
        for (unsigned int i = 0; i < nColumns; i++) {
            // result += matrix(rowIndex,i)
            result += matrixDevPtr[i * nRows + rowIndex];
        }
        vectorDevPtr[rowIndex] += scale * result;
    }
}

template<typename T, typename S>
void _cuda_addSummedColumns(T* vectorDevPtr, const S* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const S scale) {
    // parallelize over rows
    int gridSize = (int)ceil((float)nRows / THREADS_PER_BLOCK);

    __cuda_addSummedColumns<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale);
}

template __global__ void __cuda_addSummedColumns<double, double>(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const double scale);
template void            _cuda_addSummedColumns<double, double>(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const double scale);
template __global__ void __cuda_addSummedColumns<float, float>(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);
template void            _cuda_addSummedColumns<float, float>(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);
template __global__ void __cuda_addSummedColumns<double, float>(double* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);
template void            _cuda_addSummedColumns<double, float>(double* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);

/*
 *
 * addSquaredSummedColumns
 *
 */

template<typename T>
__global__ void __cuda_addSquaredSummedColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale) {
    unsigned int rowIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowIndex < nRows) {
        T result = 0.0;
        for (unsigned int i = 0; i < nColumns; i++) {
            result += matrixDevPtr[i * nRows + rowIndex] * matrixDevPtr[i * nRows + rowIndex];
        }
        vectorDevPtr[rowIndex] += scale * result;
    }
}

template<typename T>
void _cuda_addSquaredSummedColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale) {
    // parallelize over rows
    int gridSize = (int)ceil((float)nRows / THREADS_PER_BLOCK);

    __cuda_addSquaredSummedColumns<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale);
}

template __global__ void __cuda_addSquaredSummedColumns(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const double scale);
template void            _cuda_addSquaredSummedColumns(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const double scale);
template __global__ void __cuda_addSquaredSummedColumns(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);
template void            _cuda_addSquaredSummedColumns(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const float scale);

/*
 *
 * elementwise multiplication
 *
 */

template<typename T>
__global__ void __cuda_elementwiseMultiplication(T* data, T* datab, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = data[index] * datab[index];
}

template<typename T>
void _cuda_elementwiseMultiplication(T* data, T* datab, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_elementwiseMultiplication<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements);
}

template __global__ void __cuda_elementwiseMultiplication<double>(double* data, double* datab, unsigned int nElements);
template __global__ void __cuda_elementwiseMultiplication<float>(float* data, float* datab, unsigned int nElements);
template void            _cuda_elementwiseMultiplication<double>(double* data, double* datab, unsigned int nRows, unsigned int nColumns);
template void            _cuda_elementwiseMultiplication<float>(float* data, float* datab, unsigned int nRows, unsigned int nColumns);

/*
 *
 * elementwise division
 *
 */

template<typename T>
__global__ void __cuda_elementwiseDivision(T* data, T* datab, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = data[index] / datab[index];
}

template<typename T>
void _cuda_elementwiseDivision(T* data, T* datab, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_elementwiseDivision<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements);
}

template __global__ void __cuda_elementwiseDivision<double>(double* data, double* datab, unsigned int nElements);
template __global__ void __cuda_elementwiseDivision<float>(float* data, float* datab, unsigned int nElements);
template void            _cuda_elementwiseDivision<double>(double* data, double* datab, unsigned int nRows, unsigned int nColumns);
template void            _cuda_elementwiseDivision<float>(float* data, float* datab, unsigned int nRows, unsigned int nColumns);

/*
 *
 * add constant elementwise
 *
 */
template<typename T>
__global__ void __cuda_addConstantElementwise(T constant, T* data, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = data[index] + constant;
}

template<typename T>
void _cuda_addConstantElementwise(T constant, T* data, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((T)nElements / THREADS_PER_BLOCK);

    __cuda_addConstantElementwise<<<gridSize, THREADS_PER_BLOCK>>>(constant, data, nElements);
}

template __global__ void __cuda_addConstantElementwise<double>(double constant, double* data, unsigned int nElements);
template void            _cuda_addConstantElementwise<double>(double constant, double* data, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_addConstantElementwise<float>(float constant, float* data, unsigned int nElements);
template void            _cuda_addConstantElementwise<float>(float constant, float* data, unsigned int nRows, unsigned int nColumns);

/*
 *
 * getMaxOfColumns
 *
 */
template<typename T>
__global__ void __cuda_getMaxOfColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    unsigned int columnIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (columnIndex < nColumns) {
        T result = 0.0;
        for (unsigned int i = 0; i < nRows; i++) {
            // result += matrix(i, columnIndex)
            T val  = matrixDevPtr[columnIndex * nRows + i];
            result = fmax(result, val);
        }
        vectorDevPtr[columnIndex] = result;
    }
}

template<typename T>
void _cuda_getMaxOfColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    // parallelize over columns
    int gridSize = (int)ceil((float)nColumns / THREADS_PER_BLOCK);

    __cuda_getMaxOfColumns<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nColumns);
}

template __global__ void __cuda_getMaxOfColumns(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns);
template void            _cuda_getMaxOfColumns(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_getMaxOfColumns(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns);
template void            _cuda_getMaxOfColumns(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns);

/*
 * slightly faster version using tmp array
 */

template<typename T>
__global__ void __cuda_getMaxOfColumnsTmp(const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                          T* tmpDevPtr, unsigned int tmpRows) {
    unsigned int columnIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int columnPart  = blockIdx.y;
    if (columnIndex < nColumns) {
        unsigned int nRowsDiv = nRows / tmpRows;
        unsigned int startRow = columnPart * nRowsDiv;
        if (startRow < nRows) {
            unsigned int endRow = columnPart == tmpRows - 1 ? nRows : (columnPart + 1) * nRowsDiv;
            T            result = 0.0;
            for (unsigned int i = startRow; i < endRow; i++) {
                // result += matrix(i, columnIndex)
                T val  = matrixDevPtr[columnIndex * nRows + i];
                result = fmax(result, val);
            }
            tmpDevPtr[columnIndex * tmpRows + columnPart] = result;
        }
    }
}

template<typename T>
void _cuda_getMaxOfColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                           T* tmpDevPtr, unsigned int tmpRows) {
    int  gridDimx = (int)ceil((float)nColumns / THREADS_PER_BLOCK);
    int  gridDimy = tmpRows;
    dim3 gridSize(gridDimx, gridDimy);

    __cuda_getMaxOfColumnsTmp<<<gridSize, THREADS_PER_BLOCK>>>(matrixDevPtr, nRows, nColumns, tmpDevPtr, tmpRows);

    _cuda_getMaxOfColumns<T>(vectorDevPtr, tmpDevPtr, tmpRows, nColumns);
}

template __global__ void __cuda_getMaxOfColumnsTmp(const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                                   double* tmpDevPtr, unsigned int tmpRows);
template void            _cuda_getMaxOfColumns(double* vectorDevPtr, const double* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                               double* tmpDevPtr, unsigned int tmpRows);
template __global__ void __cuda_getMaxOfColumnsTmp(const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                                   float* tmpDevPtr, unsigned int tmpRows);
template void            _cuda_getMaxOfColumns(float* vectorDevPtr, const float* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                                               float* tmpDevPtr, unsigned int tmpRows);
/*
 *
 * elementwiseMultiplicationWithSigmoidDerivative
 *
 */

template<typename T>
__global__ void __cuda_elementwiseMultiplicationWithSigmoidDerivative(T* data, T* datab, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = data[index] * (datab[index] * (1 - datab[index]));
}

template<typename T>
void _cuda_elementwiseMultiplicationWithSigmoidDerivative(T* data, T* datab, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_elementwiseMultiplicationWithSigmoidDerivative<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements);
}

template __global__ void __cuda_elementwiseMultiplicationWithSigmoidDerivative(double* data, double* datab, unsigned int nElements);
template void            _cuda_elementwiseMultiplicationWithSigmoidDerivative(double* data, double* datab, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_elementwiseMultiplicationWithSigmoidDerivative(float* data, float* datab, unsigned int nElements);
template void            _cuda_elementwiseMultiplicationWithSigmoidDerivative(float* data, float* datab, unsigned int nRows, unsigned int nColumns);

/*
 *
 * elementwiseMultiplicationWithTanhDerivative
 *
 */

template<typename T>
__global__ void __cuda_elementwiseMultiplicationWithTanhDerivative(T* data, T* datab, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = data[index] * (1 - pow(datab[index], 2));
}

template<typename T>
void _cuda_elementwiseMultiplicationWithTanhDerivative(T* data, T* datab, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_elementwiseMultiplicationWithTanhDerivative<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements);
}

template __global__ void __cuda_elementwiseMultiplicationWithTanhDerivative(double* data, double* datab, unsigned int nElements);
template void            _cuda_elementwiseMultiplicationWithTanhDerivative(double* data, double* datab, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_elementwiseMultiplicationWithTanhDerivative(float* data, float* datab, unsigned int nElements);
template void            _cuda_elementwiseMultiplicationWithTanhDerivative(float* data, float* datab, unsigned int nRows, unsigned int nColumns);

/*
 *
 * multiplicationWithSoftmaxDerivative
 *
 */

template<typename T>
__global__ void __cuda_multiplicationWithSoftmaxDerivative(T* data, T* datab, T* datac, unsigned int nElements, unsigned int nRows) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = datab[index] * (data[index] - datac[index / nRows]);
}

template<typename T>
void _cuda_multiplicationWithSoftmaxDerivative(T* data, T* datab, T* datac, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_multiplicationWithSoftmaxDerivative<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, datac, nElements, nRows);
}

template __global__ void __cuda_multiplicationWithSoftmaxDerivative(double* data, double* datab, double* datac, unsigned int nElements, unsigned int nRows);
template void            _cuda_multiplicationWithSoftmaxDerivative(double* data, double* datab, double* datac, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_multiplicationWithSoftmaxDerivative(float* data, float* datab, float* datac, unsigned int nElements, unsigned int nRows);
template void            _cuda_multiplicationWithSoftmaxDerivative(float* data, float* datab, float* datac, unsigned int nRows, unsigned int nColumns);

/*
 * elementwiseMultiplicationWithRectifiedDerivative
 *
 */

template<typename T>
__global__ void __cuda_elementwiseMultiplicationWithRectifiedDerivative(T* errOut, T* activations, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        if (activations[index] <= 0)
            errOut[index] = 0;
}
template<typename T>
void _cuda_elementwiseMultiplicationWithRectifiedDerivative(T* data, T* datab, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    __cuda_elementwiseMultiplicationWithRectifiedDerivative<T><<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements);
}
template __global__ void __cuda_elementwiseMultiplicationWithRectifiedDerivative<float>(float*, float*, unsigned int);
template __global__ void __cuda_elementwiseMultiplicationWithRectifiedDerivative<double>(double*, double*, unsigned int);
template void            _cuda_elementwiseMultiplicationWithRectifiedDerivative<float>(float*, float*, unsigned int, unsigned int);
template void            _cuda_elementwiseMultiplicationWithRectifiedDerivative<double>(double*, double*, unsigned int, unsigned int);

/*
 * elementwiseMultiplicationWithEluDerivative
 *
 */

template<typename T>
__global__ void __cuda_elementwiseMultiplicationWithEluDerivative(T* errOut, T* activations, T alpha, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements && activations[index] < 0) {
        errOut[index] *= (activations[index] + alpha);
    }
}
template<typename T>
void _cuda_elementwiseMultiplicationWithEluDerivative(T* data, T* datab, T alpha, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    __cuda_elementwiseMultiplicationWithEluDerivative<T><<<gridSize, THREADS_PER_BLOCK>>>(data, datab, alpha, nElements);
}
template __global__ void __cuda_elementwiseMultiplicationWithEluDerivative<float>(float*, float*, float, unsigned int);
template __global__ void __cuda_elementwiseMultiplicationWithEluDerivative<double>(double*, double*, double, unsigned int);
template void            _cuda_elementwiseMultiplicationWithEluDerivative<float>(float*, float*, float, unsigned int, unsigned int);
template void            _cuda_elementwiseMultiplicationWithEluDerivative<double>(double*, double*, double, unsigned int, unsigned int);

/*
 *
 * addToAllColumns
 *
 */

template<typename T>
__global__ void __cuda_addToAllColumns(T* data, T* datab, unsigned int nElements, unsigned int nRows, T alpha) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] += alpha * datab[index % nRows];
}

template<typename T>
void _cuda_addToAllColumns(T* data, T* datab, unsigned int nRows, unsigned int nColumns, T alpha) {
    // TODO implement kernel without % operator (slow on GPU)
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_addToAllColumns<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements, nRows, alpha);
}

template __global__ void __cuda_addToAllColumns<double>(double* data, double* datab, unsigned int nElements, unsigned int nRows, double alpha);
template void            _cuda_addToAllColumns<double>(double* data, double* datab, unsigned int nRows, unsigned int nColumns, double alpha);
template __global__ void __cuda_addToAllColumns<float>(float* data, float* datab, unsigned int nElements, unsigned int nRows, float alpha);
template void            _cuda_addToAllColumns<float>(float* data, float* datab, unsigned int nRows, unsigned int nColumns, float alpha);

// with offset (caller has to add offset to data and adjust nRowsMat)

template<typename T>
__global__ void __cuda_addToAllColumnsWithOffset(T* data, T* datab, unsigned int nElements, unsigned int nRowsMat, unsigned int nRowsVec, T alpha) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int col   = index / nRowsVec;
    unsigned int row   = index % nRowsVec;
    if (index < nElements) {
        data[col * nRowsMat + row] += alpha * datab[row];
    }
}

template<typename T>
void _cuda_addToAllColumnsWithOffset(T* data, T* datab, unsigned int nRowsMat, unsigned nRowsVec, unsigned int nColumns, T alpha) {
    // TODO implement kernel without % operator (slow on GPU)
    unsigned int nElements = nRowsVec * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_addToAllColumns<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements, nRowsMat, nRowsVec, alpha);
}

template __global__ void __cuda_addToAllColumnsWithOffset<double>(double* data, double* datab, unsigned int nElements, unsigned int nRowsMat, unsigned int nRowsVec, double alpha);
template void            _cuda_addToAllColumnsWithOffset<double>(double* data, double* datab, unsigned int nRowsMat, unsigned int nRowsVec, unsigned int nColumns, double alpha);
template __global__ void __cuda_addToAllColumnsWithOffset<float>(float* data, float* datab, unsigned int nElements, unsigned int nRowsMat, unsigned int nRowsVec, float alpha);
template void            _cuda_addToAllColumnsWithOffset<float>(float* data, float* datab, unsigned int nRowsMat, unsigned int nRowsVec, unsigned int nColumns, float alpha);

/*
 *
 * addToAllRows
 *
 */
template<typename T>
__global__ void __cuda_addToAllRows(T* data, T* datab, unsigned int nElements, unsigned int nRows, T alpha) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] += alpha * datab[index / nRows];
}
template<typename T>
void _cuda_addToAllRows(T* data, T* datab, unsigned int nRows, unsigned int nColumns, T alpha) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_addToAllRows<<<gridSize, THREADS_PER_BLOCK>>>(data, datab, nElements, nRows, alpha);
}

template __global__ void __cuda_addToAllRows<double>(double* data, double* datab, unsigned int nElements, unsigned int nRows, double alpha);
template void            _cuda_addToAllRows<double>(double* data, double* datab, unsigned int nRows, unsigned int nColumns, double alpha);
template __global__ void __cuda_addToAllRows<float>(float* data, float* datab, unsigned int nElements, unsigned int nRows, float alpha);
template void            _cuda_addToAllRows<float>(float* data, float* datab, unsigned int nRows, unsigned int nColumns, float alpha);

/*
 *
 * multiplyColumnsByScalars
 *
 */
template<typename T>
__global__ void __cuda_multiplyColumnsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nElements) {
    unsigned int index    = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int colIndex = index / nRows;
    if (index < nElements)
        matrixDevPtr[index] = matrixDevPtr[index] * vectorDevPtr[colIndex];
}
template<typename T>
void _cuda_multiplyColumnsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_multiplyColumnsByScalars<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nElements);
}

template __global__ void __cuda_multiplyColumnsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_multiplyColumnsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_multiplyColumnsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_multiplyColumnsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nColumns);

/*
 *
 * divideColumnsByScalars
 *
 */
template<typename T>
__global__ void __cuda_divideColumnsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nElements) {
    unsigned int index    = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int colIndex = index / nRows;
    if (index < nElements)
        matrixDevPtr[index] = matrixDevPtr[index] / vectorDevPtr[colIndex];
}
template<typename T>
void _cuda_divideColumnsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_divideColumnsByScalars<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nElements);
}

template __global__ void __cuda_divideColumnsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_divideColumnsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_divideColumnsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_divideColumnsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nColumns);

/*
 *
 * multiplyRowsByScalars
 *
 */
template<typename T>
__global__ void __cuda_multiplyRowsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nElements) {
    unsigned int index    = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int rowIndex = index % nRows;
    if (index < nElements)
        matrixDevPtr[index] = matrixDevPtr[index] * vectorDevPtr[rowIndex];
}
template<typename T>
void _cuda_multiplyRowsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_multiplyRowsByScalars<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nElements);
}

template __global__ void __cuda_multiplyRowsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_multiplyRowsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_multiplyRowsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_multiplyRowsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nColumns);

/*
 *
 * divideRowsByScalars
 *
 */
template<typename T>
__global__ void __cuda_divideRowsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nElements) {
    unsigned int index    = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int rowIndex = index % nRows;
    if (index < nElements)
        matrixDevPtr[index] = matrixDevPtr[index] / vectorDevPtr[rowIndex];
}
template<typename T>
void _cuda_divideRowsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_divideRowsByScalars<<<gridSize, THREADS_PER_BLOCK>>>(vectorDevPtr, matrixDevPtr, nRows, nElements);
}

template __global__ void __cuda_divideRowsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_divideRowsByScalars<double>(const double* vectorDevPtr, double* matrixDevPtr, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_divideRowsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nElements);
template void            _cuda_divideRowsByScalars<float>(const float* vectorDevPtr, float* matrixDevPtr, unsigned int nRows, unsigned int nColumns);

/*
 *
 * sign
 *
 */
template<typename T>
__global__ void __cuda_sign(T* out, const T* in, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements) {
        out[index] = in[index] == 0 ? 0 : copysignf(1.0, in[index]);
    }
}
template<typename T>
void _cuda_sign(T* out, const T* in, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_sign<<<gridSize, THREADS_PER_BLOCK>>>(out, in, nElements);
}

template __global__ void __cuda_sign<double>(double* out, const double* in, unsigned int nElements);
template void            _cuda_sign<double>(double* out, const double* in, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_sign<float>(float* out, const float* in, unsigned int nElements);
template void            _cuda_sign<float>(float* out, const float* in, unsigned int nRows, unsigned int nColumns);

/*
 *
 *  fill
 *
 */
template<typename T>
__global__ void __cuda_fill(T* data, T value, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements)
        data[index] = value;
}
template<typename T>
void _cuda_fill(T* data, T value, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_fill<<<gridSize, THREADS_PER_BLOCK>>>(data, value, nElements);
}

template __global__ void __cuda_fill<double>(double* data, double value, unsigned int nElements);
template void            _cuda_fill<double>(double* data, double value, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_fill<float>(float* data, float value, unsigned int nElements);
template void            _cuda_fill<float>(float* data, float value, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_fill<uint>(uint* data, uint value, unsigned int nElements);
template void            _cuda_fill<uint>(uint* data, uint value, unsigned int nRows, unsigned int nColumns);

/*
 *
 *  ensure minimal value
 *
 */
template<typename T>
__global__ void __cuda_ensureMinimalValue(T* data, T value, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ((index < nElements) && (data[index] < value))
        data[index] = value;
}

template<typename T>
void _cuda_ensureMinimalValue(T* data, T value, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_ensureMinimalValue<<<gridSize, THREADS_PER_BLOCK>>>(data, value, nElements);
}

template __global__ void __cuda_ensureMinimalValue(double* data, double value, unsigned int nElements);
template void            _cuda_ensureMinimalValue(double* data, double value, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_ensureMinimalValue(float* data, float value, unsigned int nElements);
template void            _cuda_ensureMinimalValue(float* data, float value, unsigned int nRows, unsigned int nColumns);

/*
 *
 *  ELU
 *
 */
template<typename T>
__global__ void __cuda_elu(T* data, T alpha, unsigned int nElements) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements && data[index] < 0)
        data[index] = alpha * (exp(data[index]) - 1);
}

template<typename T>
void _cuda_elu(T* data, T alpha, unsigned int nRows, unsigned int nColumns) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);
    __cuda_elu<<<gridSize, THREADS_PER_BLOCK>>>(data, alpha, nElements);
}

template __global__ void __cuda_elu(double* data, double alpha, unsigned int nElements);
template void            _cuda_elu(double* data, double alpha, unsigned int nRows, unsigned int nColumns);
template __global__ void __cuda_elu(float* data, float alpha, unsigned int nElements);
template void            _cuda_elu(float* data, float alpha, unsigned int nRows, unsigned int nColumns);

/*
 *
 * nClassificationErrors
 *
 *
 */
template<typename T>
__global__ void __cuda_nClassificationErrors(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* errBuf) {
    unsigned int column        = blockIdx.x;
    unsigned int tid           = threadIdx.x;
    unsigned int blocksize     = (unsigned int)ceil((float)nRows / blockDim.x);  // e.g. 4501/1024 = 5
    unsigned int max_thread_id = (unsigned int)floor((float)nRows / blocksize);  // e.g. 4501/5    = 900

    volatile __shared__ bool error_found[THREADS_PER_BLOCK];
    error_found[tid] = false;

    if (column < nColumns && tid <= max_thread_id) {
        uint beginCol  = column * nRows;
        uint c_true    = alignmentDevPtr[column];
        T    true_prob = matrixPtr[beginCol + c_true];

        for (uint i = tid * blocksize; i < (tid + 1) * blocksize; ++i) {
            if (i >= nRows)
                break;
            if (i == c_true)
                continue;
            error_found[tid] |= matrixPtr[beginCol + i] > true_prob;
        }
    }
    __syncthreads();

    for (uint s = THREADS_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s)
            error_found[tid] |= error_found[tid + s];
        __syncthreads();
    }
    if (tid == 0 && column < nColumns) {
        errBuf[column] = error_found[0] ? 1.0 : 0.0;
    }
}
template<typename T>
void _cuda_nClassificationErrors(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* errorBuf) {
    __cuda_nClassificationErrors<<<nColumns, THREADS_PER_BLOCK>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, errorBuf);
}

template __global__ void __cuda_nClassificationErrors<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* errBuf);
template void            _cuda_nClassificationErrors<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* errBuf);
template __global__ void __cuda_nClassificationErrors<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* errBuf);
template void            _cuda_nClassificationErrors<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* errBuf);

// crossEntropyObjectiveFunction
template<typename T>
__global__ void __cuda_crossEntropyObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn) {
    *objFctn = 0.0f;
    for (int column = 0; column < nColumns; column++) {
        unsigned int position = column * nRows + alignmentDevPtr[column];
        *objFctn -= log(matrixPtr[position]);
    }
}

template<typename T>
void _cuda_crossEntropyObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn) {
    // no parallelization, but probably not relevant
    __cuda_crossEntropyObjectiveFunction<<<1, 1>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, objFctn);
}

template __global__ void __cuda_crossEntropyObjectiveFunction<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn);
template void            _cuda_crossEntropyObjectiveFunction<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn);
template __global__ void __cuda_crossEntropyObjectiveFunction<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn);
template void            _cuda_crossEntropyObjectiveFunction<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn);

// weightedCrossEntropyObjectiveFunction
template<typename T>
__global__ void __cuda_weightedCrossEntropyObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn, T* weights) {
    *objFctn = 0.0f;
    for (int column = 0; column < nColumns; column++) {
        unsigned int position = column * nRows + alignmentDevPtr[column];
        *objFctn -= log(matrixPtr[position]) * weights[column];
    }
}

template<typename T>
void _cuda_weightedCrossEntropyObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn, T* weights) {
    // no parallelization, but probably not relevant
    __cuda_weightedCrossEntropyObjectiveFunction<<<1, 1>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, objFctn, weights);
}

template __global__ void __cuda_weightedCrossEntropyObjectiveFunction<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn, double* weights);
template void            _cuda_weightedCrossEntropyObjectiveFunction<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn, double* weights);
template __global__ void __cuda_weightedCrossEntropyObjectiveFunction<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn, float* weights);
template void            _cuda_weightedCrossEntropyObjectiveFunction<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn, float* weights);

// squaredErrorObjectiveFunction

template<typename T>
__global__ void __cuda_squaredErrorObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < nRows) {
        objFctn[row] = 0.0f;
        for (int column = 0; column < nColumns; column++) {
            T            kroneckerDelta = alignmentDevPtr[column] == row ? 1.0 : 0.0;
            unsigned int position       = column * nRows + row;
            objFctn[row] += (matrixPtr[position] - kroneckerDelta) * (matrixPtr[position] - kroneckerDelta);
        }
    }
}

template<typename T>
void _cuda_squaredErrorObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn) {
    unsigned int nElements = nRows;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    // no parallelization, but probably not relevant
    __cuda_squaredErrorObjectiveFunction<<<gridSize, THREADS_PER_BLOCK>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, objFctn);
}

template __global__ void __cuda_squaredErrorObjectiveFunction(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn);
template void            _cuda_squaredErrorObjectiveFunction(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn);
template __global__ void __cuda_squaredErrorObjectiveFunction(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn);
template void            _cuda_squaredErrorObjectiveFunction(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn);

// weightedSquaredErrorObjectiveFunction

template<typename T>
__global__ void __cuda_weightedSquaredErrorObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn, T* weights) {
    unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < nRows) {
        objFctn[row] = 0.0f;
        for (int column = 0; column < nColumns; column++) {
            T            kroneckerDelta = alignmentDevPtr[column] == row ? 1.0 : 0.0;
            unsigned int position       = column * nRows + row;
            objFctn[row] += (matrixPtr[position] - kroneckerDelta) * (matrixPtr[position] - kroneckerDelta) * weights[column];
        }
    }
}

template<typename T>
void _cuda_weightedSquaredErrorObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn, T* weights) {
    unsigned int nElements = nRows;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_weightedSquaredErrorObjectiveFunction<<<gridSize, THREADS_PER_BLOCK>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, objFctn, weights);
}

template __global__ void __cuda_weightedSquaredErrorObjectiveFunction(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn, double* weights);
template void            _cuda_weightedSquaredErrorObjectiveFunction(double* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, double* objFctn, double* weights);
template __global__ void __cuda_weightedSquaredErrorObjectiveFunction(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn, float* weights);
template void            _cuda_weightedSquaredErrorObjectiveFunction(float* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, float* objFctn, float* weights);

// ###########################################################################
// binaryDivergenceObjectiveFunction
template<typename T>
__global__ void __cuda_binaryDivergenceObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn) {
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column < nColumns) {
        objFctn[column] = 0.0;
        for (int row = 0; row < nRows; row++) {
            unsigned int position = column * nRows + row;
            if (alignmentDevPtr[column] == row)
                objFctn[column] -= log(matrixPtr[position]);
            else
                objFctn[column] -= log(1.0 - matrixPtr[position]);
        }
    }
}
template<typename T>
void _cuda_binaryDivergenceObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn) {
    int gridSize = (int)ceil((float)nColumns / THREADS_PER_BLOCK);
    __cuda_binaryDivergenceObjectiveFunction<T><<<gridSize, THREADS_PER_BLOCK>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, objFctn);
}
template __global__ void __cuda_binaryDivergenceObjectiveFunction<float>(float*, unsigned int, unsigned int, unsigned int*, float*);
template __global__ void __cuda_binaryDivergenceObjectiveFunction<double>(double*, unsigned int, unsigned int, unsigned int*, double*);
template void            _cuda_binaryDivergenceObjectiveFunction<float>(float*, unsigned int, unsigned int, unsigned int*, float*);
template void            _cuda_binaryDivergenceObjectiveFunction<double>(double*, unsigned int, unsigned int, unsigned int*, double*);

// ###########################################################################
// weightedBinaryDivergenceObjectiveFunction
template<typename T>
__global__ void __cuda_weightedBinaryDivergenceObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn, T* weights) {
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column < nColumns) {
        objFctn[column] = 0.0;
        for (int row = 0; row < nRows; row++) {
            unsigned int position = column * nRows + row;
            if (alignmentDevPtr[column] == row)
                objFctn[column] -= log(matrixPtr[position]) * weights[column];
            else
                objFctn[column] -= log(1.0 - matrixPtr[position]) * weights[column];
        }
    }
}
template<typename T>
void _cuda_weightedBinaryDivergenceObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* objFctn, T* weights) {
    int gridSize = (int)ceil((float)nColumns / THREADS_PER_BLOCK);
    __cuda_weightedBinaryDivergenceObjectiveFunction<T><<<gridSize, THREADS_PER_BLOCK>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, objFctn, weights);
}
template __global__ void __cuda_weightedBinaryDivergenceObjectiveFunction<float>(float*, unsigned int, unsigned int, unsigned int*, float*, float*);
template __global__ void __cuda_weightedBinaryDivergenceObjectiveFunction<double>(double*, unsigned int, unsigned int, unsigned int*, double*, double*);
template void            _cuda_weightedBinaryDivergenceObjectiveFunction<float>(float*, unsigned int, unsigned int, unsigned int*, float*, float*);
template void            _cuda_weightedBinaryDivergenceObjectiveFunction<double>(double*, unsigned int, unsigned int, unsigned int*, double*, double*);

// ###########################################################################
// binary divergence softmax gradient computation

template<typename T>
__global__ void __cuda_binaryDivergenceSoftmaxGradient(T* gradient, unsigned int nRows, unsigned int nColumns, const T* output, const unsigned int* alignment) {
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column < nColumns) {
        T constsum = 0.0;
        for (int i = 0; i < nRows; ++i) {
            unsigned int position = column * nRows + i;
            const T      y        = output[position];
            if (alignment[column] == i)
                constsum -= 1.0;
            else if (y < 1.0)
                constsum += y / (1.0 - y);
        }

        for (int i = 0; i < nRows; ++i) {
            unsigned int position = column * nRows + i;
            const T      y        = output[position];
            if (alignment[column] == i)
                gradient[position] = -1.0 - y * constsum;
            else {
                if (y < 1.0)
                    gradient[position] = y * (1.0 / (1.0 - y) - constsum);
                else
                    gradient[position] = 0.0;
            }
        }
    }
}
template<typename T>
void _cuda_binaryDivergenceSoftmaxGradient(T* matrixPtr, unsigned int nRows, unsigned int nColumns, const T* outputDevPtr, const unsigned int* alignmentDevPtr) {
    int gridSize = (int)ceil((float)nColumns / THREADS_PER_BLOCK);
    __cuda_binaryDivergenceSoftmaxGradient<T><<<gridSize, THREADS_PER_BLOCK>>>(matrixPtr, nRows, nColumns, outputDevPtr, alignmentDevPtr);
}
template __global__ void __cuda_binaryDivergenceSoftmaxGradient<float>(float*, unsigned int, unsigned int, const float*, const unsigned int*);
template __global__ void __cuda_binaryDivergenceSoftmaxGradient<double>(double*, unsigned int, unsigned int, const double*, const unsigned int*);
template void            _cuda_binaryDivergenceSoftmaxGradient<float>(float*, unsigned int, unsigned int, const float*, const unsigned int*);
template void            _cuda_binaryDivergenceSoftmaxGradient<double>(double*, unsigned int, unsigned int, const double*, const unsigned int*);

template<typename T>
__global__ void __cuda_addKroneckerDelta(T* matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int* alignmentDevPtr, const T scale) {
    unsigned int index     = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int nElements = nRows * nColumns;
    if (index < nElements) {
        unsigned int colIndex = index / nRows;
        unsigned int rowIndex = index % nRows;
        matrixPtr[index] += rowIndex == alignmentDevPtr[colIndex] ? scale : 0.0;
    }
}

template<typename T>
void _cuda_addKroneckerDelta(T* matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int* alignmentDevPtr, const T scale) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_addKroneckerDelta<<<gridSize, THREADS_PER_BLOCK>>>(matrixPtr, nRows, nColumns, alignmentDevPtr, scale);
}

template __global__ void __cuda_addKroneckerDelta<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int* alignmentDevPtr, const double scale);
template void            _cuda_addKroneckerDelta<double>(double* matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int* alignmentDevPtr, const double scale);
template __global__ void __cuda_addKroneckerDelta<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int* alignmentDevPtr, const float scale);
template void            _cuda_addKroneckerDelta<float>(float* matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int* alignmentDevPtr, const float scale);

/*
 *  appendSecondOrderFeatures
 */

template<typename T>
__global__ void __cuda_appendSecondOrderFeatures(const T* X, unsigned int nRowsX, unsigned int nColumnsX, T* Y, unsigned int nRowsY, unsigned int offset) {
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column < nColumnsX) {
        unsigned int pos = offset;
        for (unsigned int i = 0; i < nRowsX; ++i) {
            for (unsigned int j = i; j < nRowsX; ++j) {
                Y[column * nRowsY + pos] = X[column * nRowsX + i] * X[column * nRowsX + j];
                pos++;
            }
        }
    }
}

template<typename T>
void _cuda_appendSecondOrderFeatures(const T* X, unsigned int nRowsX, unsigned int nColumnsX, T* Y, unsigned int nRowsY, unsigned int offset) {
    int gridSize = (int)ceil((float)nColumnsX / THREADS_PER_BLOCK);
    __cuda_appendSecondOrderFeatures<<<gridSize, THREADS_PER_BLOCK>>>(X, nRowsX, nColumnsX, Y, nRowsY, offset);
}

template __global__ void __cuda_appendSecondOrderFeatures(const double* X, unsigned int nRowsX, unsigned int nColumnsX, double* Y, unsigned int nRowsY, unsigned int offset);
template void            _cuda_appendSecondOrderFeatures(const double* X, unsigned int nRowsX, unsigned int nColumnsX, double* Y, unsigned int nRowsY, unsigned int offset);
template __global__ void __cuda_appendSecondOrderFeatures(const float* X, unsigned int nRowsX, unsigned int nColumnsX, float* Y, unsigned int nRowsY, unsigned int offset);
template void            _cuda_appendSecondOrderFeatures(const float* X, unsigned int nRowsX, unsigned int nColumnsX, float* Y, unsigned int nRowsY, unsigned int offset);

// appendThirdOrderFeatures

template<typename T>
__global__ void __cuda_appendThirdOrderFeatures(const T* X, unsigned int nRowsX, unsigned int nColumnsX, T* Y, unsigned int nRowsY, unsigned int offset) {
    unsigned int column = threadIdx.x + blockIdx.x * blockDim.x;
    if (column < nColumnsX) {
        unsigned int pos = offset;
        for (unsigned int i = 0; i < nRowsX; ++i) {
            for (unsigned int j = i; j < nRowsX; ++j) {
                for (unsigned int k = j; k < nRowsX; ++k) {
                    Y[column * nRowsY + pos] = X[column * nRowsX + i] * X[column * nRowsX + j] * X[column * nRowsX + k];
                    pos++;
                }
            }
        }
    }
}

template<typename T>
void _cuda_appendThirdOrderFeatures(const T* X, unsigned int nRowsX, unsigned int nColumnsX, T* Y, unsigned int nRowsY, unsigned int offset) {
    int gridSize = (int)ceil((float)nColumnsX / THREADS_PER_BLOCK);
    __cuda_appendThirdOrderFeatures<<<gridSize, THREADS_PER_BLOCK>>>(X, nRowsX, nColumnsX, Y, nRowsY, offset);
}

template __global__ void __cuda_appendThirdOrderFeatures(const double* X, unsigned int nRowsX, unsigned int nColumnsX, double* Y, unsigned int nRowsY, unsigned int offset);
template void            _cuda_appendThirdOrderFeatures(const double* X, unsigned int nRowsX, unsigned int nColumnsX, double* Y, unsigned int nRowsY, unsigned int offset);
template __global__ void __cuda_appendThirdOrderFeatures(const float* X, unsigned int nRowsX, unsigned int nColumnsX, float* Y, unsigned int nRowsY, unsigned int offset);
template void            _cuda_appendThirdOrderFeatures(const float* X, unsigned int nRowsX, unsigned int nColumnsX, float* Y, unsigned int nRowsY, unsigned int offset);
/*
 *
 * dropout
 *
 */
template<typename T>
__global__ void __cuda_dropout(T* data, const T* mask, unsigned int nElements, T dropoutProbability) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ((index < nElements) && (mask[index] <= dropoutProbability))
        data[index] = 0.0;
}

template<typename T>
void _cuda_dropout(T* data, const T* mask, unsigned int nRows, unsigned int nColumns, T dropoutProbability) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_dropout<<<gridSize, THREADS_PER_BLOCK>>>(data, mask, nElements, dropoutProbability);
}

template __global__ void __cuda_dropout(double* data, const double* mask, unsigned int nElements, double dropoutProbability);
template void            _cuda_dropout(double* data, const double* mask, unsigned int nRows, unsigned int nColumns, double dropoutProbability);
template __global__ void __cuda_dropout(float* data, const float* mask, unsigned int nElements, float dropoutProbability);
template void            _cuda_dropout(float* data, const float* mask, unsigned int nRows, unsigned int nColumns, float dropoutProbability);

/*
 *
 * l1clipping
 *
 */
template<typename T>
__global__ void __cuda_l1clipping(T* data, unsigned int nElements, T value) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements) {
        if (data[index] > 0) {
            if (data[index] - value > 0)
                data[index] = data[index] - value;
            else
                data[index] = 0;
        }
        else if (data[index] < 0) {
            if (data[index] + value < 0)
                data[index] = data[index] + value;
            else
                data[index] = 0;
        }
    }
}

template<typename T>
void _cuda_l1clipping(T* data, unsigned int nRows, unsigned int nColumns, T value) {
    unsigned int nElements = nRows * nColumns;
    int          gridSize  = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_l1clipping<<<gridSize, THREADS_PER_BLOCK>>>(data, nElements, value);
}

template __global__ void __cuda_l1clipping(double* data, unsigned int nElements, double value);
template void            _cuda_l1clipping(double* data, unsigned int nRows, unsigned int nColumns, double value);
template __global__ void __cuda_l1clipping(float* data, unsigned int nElements, float value);
template void            _cuda_l1clipping(float* data, unsigned int nRows, unsigned int nColumns, float value);

/*
 *
 * clip
 *
 */
template<typename T>
__global__ void __cuda_clip(T* data, unsigned int nElements, T maxAbsValue);

template<>
__global__ void __cuda_clip(float* data, unsigned int nElements, float maxAbsValue) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements) {
        data[index] = data[index] > 0 ? fminf(data[index], maxAbsValue) : fmaxf(data[index], -maxAbsValue);
    }
}
template<>
__global__ void __cuda_clip(double* data, unsigned int nElements, double maxAbsValue) {
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < nElements) {
        data[index] = data[index] > 0 ? fmin(data[index], maxAbsValue) : fmax(data[index], -maxAbsValue);
    }
}

template<typename T>
void _cuda_clip(T* data, unsigned int nElements, T maxAbsValue) {
    int gridSize = (int)ceil((float)nElements / THREADS_PER_BLOCK);

    __cuda_clip<<<gridSize, THREADS_PER_BLOCK>>>(data, nElements, maxAbsValue);
}

// template __global__ void __cuda_clip(double *data, unsigned int nElements, double value);
template void _cuda_clip(double* data, unsigned int nElements, double value);
// template __global__ void __cuda_clip(float *data, unsigned int nElements, float value);
template void _cuda_clip(float* data, unsigned int nElements, float value);

//////////////
// maxout helper functions
// forward

template<typename T>
__global__ void __cuda_addPoolingMax(const T* input, T* output, unsigned int* argmax,
                                     unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out,
                                     unsigned int poolingSize, bool poolingAbs) {
    // unsigned  int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int pool     = blockIdx.x;
    unsigned int column   = threadIdx.x;
    unsigned int pool_max = nRows_in / poolingSize;
    // if (column < nColumns) {
    if (column < nColumns && pool < pool_max) {
        unsigned int inpos = nRows_in * column + pool * poolingSize;
        // for (unsigned int pool = 0; pool < nRows_in/poolingSize; ++pool) {
        T            maxval = -9999999999;
        T            val;
        unsigned int maxidx = 0;
        for (unsigned int in = 0; in < poolingSize; ++in, ++inpos) {
            // T val = poolingAbs ? abs(input[inpos]) : input[inpos]; // TODO: remove?
            val = input[inpos];
            if (maxval < val) {
                maxval = val;
                maxidx = inpos;
            }
        }
        unsigned int outpos = column * nRows_out + pool;
        output[outpos]      = maxval;
        argmax[outpos]      = maxidx;
        //}
    }
}

template<typename T>
void _cuda_addPoolingMax(const T* input, T* output, unsigned int* argmax,
                         unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out,
                         unsigned int poolingSize, bool poolingAbs) {
    // parallelization over columns only
    // int gridSize = (int)ceil( (float) nColumns/THREADS_PER_BLOCK);
    //__cuda_addPoolingMax <<< gridSize , THREADS_PER_BLOCK >>> (input, output, argmax, nColumns, nRows_in, nRows_out, poolingSize, poolingAbs);
    int pool_max = nRows_in / poolingSize;
    __cuda_addPoolingMax<<<pool_max, nColumns>>>(input, output, argmax, nColumns, nRows_in, nRows_out, poolingSize, poolingAbs);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 8\n");
}

template void _cuda_addPoolingMax(const double* input, double* output, unsigned int* argmax,
                                  unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out, unsigned int poolingSize, bool poolingAbs);
template void _cuda_addPoolingMax(const float* input, float* output, unsigned int* argmax,
                                  unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out, unsigned int poolingSize, bool poolingAbs);

// maxout backward

template<typename T>
__global__ void __cuda_backpropPoolingMax(T* output, const unsigned int* argmax, const T* error,
                                          unsigned int nColumns, unsigned int nRows_err) {
    // unsigned  int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int err_idx = blockIdx.x;
    unsigned int column  = threadIdx.x;

    // if (column < nColumns) {
    if (column < nColumns && err_idx < nRows_err) {
        unsigned int offset = column * nRows_err;
        // for (unsigned int pos = offset; pos < offset + nRows_err; ++pos) {
        int pos             = offset + err_idx;
        output[argmax[pos]] = error[pos];
        //}
    }
}

template<typename T>
void _cuda_backpropPoolingMax(T* output, const unsigned int* argmax, const T* error,
                              unsigned int nColumns, unsigned int nRows_err) {
    // parallelization over columns only
    // int gridSize = (int)ceil( (float) nColumns/THREADS_PER_BLOCK);
    //__cuda_backpropPoolingMax <<< gridSize , THREADS_PER_BLOCK >>> (output, argmax, error, nColumns, nRows_err);
    __cuda_backpropPoolingMax<<<nRows_err, nColumns>>>(output, argmax, error, nColumns, nRows_err);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 7\n");
}

template void _cuda_backpropPoolingMax(double* output, const unsigned int* argmax, const double* error,
                                       unsigned int nColumns, unsigned int nRows_err);
template void _cuda_backpropPoolingMax(float* output, const unsigned int* argmax, const float* error,
                                       unsigned int nColumns, unsigned int nRows_err);

//////////////
// P-norm pooling
// forward

template<typename T>
__global__ void __cuda_addPoolingPnorm(const T* input, T* output,
                                       unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out,
                                       unsigned int poolingSize, int pnorm) {
    // unsigned  int column = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int pool     = blockIdx.x;
    unsigned int column   = threadIdx.x;
    unsigned int pool_max = nRows_in / poolingSize;
    // if (column < nColumns) {
    if (column < nColumns && pool < pool_max) {
        unsigned int inpos     = nRows_in * column + pool * poolingSize;
        T            inv_pnorm = 1.0 / pnorm;
        // for (unsigned int pool = 0; pool < nRows_in/poolingSize; ++pool) {
        T val = 0;
        for (unsigned int in = 0; in < poolingSize; ++in, ++inpos) {
            // T val = poolingAbs ? abs(input[inpos]) : input[inpos]; // TODO: remove?
            val += pow(abs(input[inpos]), pnorm);
        }
        unsigned int outpos = column * nRows_out + pool;
        output[outpos]      = pow(val, inv_pnorm);
        //}
    }
}

template<typename T>
void _cuda_addPoolingPnorm(const T* input, T* output,
                           unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out,
                           unsigned int poolingSize, unsigned int pnorm) {
    // parallelization over columns only
    // int gridSize = (int)ceil( (float) nColumns/THREADS_PER_BLOCK);
    //__cuda_addPoolingMax <<< gridSize , THREADS_PER_BLOCK >>> (input, output, argmax, nColumns, nRows_in, nRows_out, poolingSize, poolingAbs);
    int pool_max = nRows_in / poolingSize;
    __cuda_addPoolingPnorm<<<pool_max, nColumns>>>(input, output, nColumns, nRows_in, nRows_out, poolingSize, pnorm);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 6\n");
}

template void _cuda_addPoolingPnorm(const double* input, double* output,
                                    unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out, unsigned int poolingSize, unsigned int pnorm);
template void _cuda_addPoolingPnorm(const float* input, float* output,
                                    unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out, unsigned int poolingSize, unsigned int pnorm);

// maxout backward

template<typename T>
__global__ void __cuda_backpropPoolingPnorm(T* output, const T* error,
                                            unsigned int nColumns, unsigned int nRows_err, unsigned int poolingSize, unsigned int pnorm) {
    unsigned int pool     = blockIdx.x;
    unsigned int column   = threadIdx.x;
    unsigned int nRows_in = nRows_err * poolingSize;
    if (column < nColumns && pool < nRows_err) {
        unsigned int inpos = nRows_in * column + pool * poolingSize;
        for (unsigned int in = 0; in < poolingSize; ++in, ++inpos) {
            int pos       = column * nRows_err + pool;
            output[inpos] = error[pos];
        }
    }
}

template<typename T>
void _cuda_backpropPoolingPnorm(T* output, const T* error,
                                unsigned int nColumns, unsigned int nRows_err,
                                unsigned int poolingSize, unsigned int pnorm) {
    // parallelization over columns only
    // int gridSize = (int)ceil( (float) nColumns/THREADS_PER_BLOCK);
    //__cuda_backpropPoolingMax <<< gridSize , THREADS_PER_BLOCK >>> (output, argmax, error, nColumns, nRows_err);
    __cuda_backpropPoolingPnorm<<<nRows_err, nColumns>>>(output, error, nColumns, nRows_err, poolingSize, pnorm);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 5\n");
}

template void _cuda_backpropPoolingPnorm(double* output, const double* error,
                                         unsigned int nColumns, unsigned int nRows_err, unsigned int poolingSize, unsigned int pnorm);
template void _cuda_backpropPoolingPnorm(float* output, const float* error,
                                         unsigned int nColumns, unsigned int nRows_err, unsigned int poolingSize, unsigned int pnorm);

// ###########################################################################
// convolutional layer helping functions

template<typename T>
__global__ void __cuda_convExtractPatches(const T* input, const int* patchIdx, T* patches,
                                          int* inverse_patches,
                                          int input_frames, int input_dim, int shifts_num, int shifts_dim, int patch_dim) {
    int t = blockIdx.x;
    // int si = blockIdx.y;
    // int s  = threadIdx.x;
    int s   = blockIdx.y;
    int tid = threadIdx.x;

    // volatile __shared__ int tgt_idx[THREADS_PER_BLOCK];
    // volatile __shared__ T   tgt_val[THREADS_PER_BLOCK];
    // tgt_idx[tid] = -1;
    // tgt_val[tid] = 0;

    int si_block = (int)ceil((float)shifts_dim / THREADS_PER_BLOCK);

    // if (t < input_frames) {
    // if (t < input_frames && s < shifts_num && si < shifts_dim) {
    if (t < input_frames && s < shifts_num) {
        int num_input_elems = input_frames * input_dim;
        int inverse_idx;
        // for (int s = 0; s < shifts_num; ++s) {
        for (int si = tid * si_block; si < (tid + 1) * si_block; ++si) {
            if (si >= shifts_dim)
                break;
            int pos_patches = (t * shifts_num + s) * shifts_dim + si;

            int i = t * input_dim + patchIdx[shifts_num * si + s];  // patchIdx.at(s, si);
            if (i < 0 || i >= num_input_elems)
                continue;
            // while (i < 0)               i += input_dim;
            // while (i >= num_input_elems) i -= input_dim;

            T val = 0;
            // if (i >= 0 && i < num_input_elems)  {

            val = input[i];
            // inverse_idx = inverse_patches_counts[i]*num_input_elems + i;
            inverse_idx                  = (si % patch_dim) * num_input_elems + i;
            inverse_patches[inverse_idx] = pos_patches;  // slow TODO
                                                         // if (pos_patches < 0 || pos_patches >= 138240)
            // printf("%d: %d -> %d, %d, %d\n", inverse_patches_counts[i], inverse_idx, pos_patches, i, t);
            // inverse_patches_counts[i] += 1;
            // atomicAdd(&(inverse_patches_counts[i]), 1);
            //}
            // patches->at(si, t*shifts_num + s) = val;

            patches[pos_patches] = val;  // slow TODO
                                         // tgt_idx[tid] = pos_patches;
            // tgt_val[tid] = val;
            // if(t==1 && s == 0) printf("t=%d s=%d si=%d i=%d patchIdx=%d pos_patches=%d, inverse_idx=%d\n", t, s, si, i, patchIdx[shifts_num*si + s], pos_patches, inverse_idx);
        }
        //}
        //__syncthreads();
        // int FACTOR = patch_dim;
        // if (tid < FACTOR) {
        //    for (int i = 0; i < THREADS_PER_BLOCK; i += FACTOR) {
        //        if(tgt_idx[i] > -1)
        //            patches[tgt_idx[i]] = tgt_val[i];
        //    }
        //}
    }
}
template<typename T>
void _cuda_convExtractPatches(const T* input, const int* patchIdx, T* patches,
                              int* inverse_patches,
                              int input_frames, int input_dim, int shifts_num, int shifts_dim, int patch_dim) {
    //    dim3 gridSize(input_frames, shifts_dim);
    //    __cuda_convExtractPatches<T> <<<gridSize, shifts_num>>> (input, patchIdx, patches,
    //	        inverse_patches,
    //		input_frames, input_dim, shifts_num, shifts_dim, patch_dim);
    dim3 gridSize(input_frames, shifts_num);
    __cuda_convExtractPatches<T><<<gridSize, THREADS_PER_BLOCK>>>(input, patchIdx, patches,
                                                                  inverse_patches,
                                                                  input_frames, input_dim, shifts_num, shifts_dim, patch_dim);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 4\n");
}

template void _cuda_convExtractPatches(const float* input, const int* patchIdx, float* patches,
                                       int* inverse_patches,
                                       int input_frames, int input_dim, int shifts_num, int shifts_dim, int patch_dim);
template void _cuda_convExtractPatches(const double* input, const int* patchIdx, double* patches,
                                       int* inverse_patches,
                                       int input_frames, int input_dim, int shifts_num, int shifts_dim, int patch_dim);

///////////////////////////////////////

template<typename T>
__global__ void __cuda_convRestoreFromPatches(T* unwarped_error, const T* warped_error,
                                              const int* patchIdx, int* inverse_patches, int num_input_elems, int patch_dim) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < num_input_elems) {
        //	T err = 0;
        int idx;
        int inv_patch_idx;
        /*
        if (i==0) {
            for(int a=0; a<1000; ++a) {
                for(int b=0; b<42; ++b) {
                    printf("%d ", inverse_patches[b*num_input_elems+a] );
                }
                printf("\n");
            }
        }
        */
        // int N = 0;
        for (int col = 0; col < patch_dim; ++col) {
            inv_patch_idx = col * num_input_elems + i;
            idx           = inverse_patches[inv_patch_idx];
            // if (i==0) printf("col=%d, inv_patch_idx=%d, idx=%d \n", col, inv_patch_idx, idx);
            /*
                        if (col>=81 || col*num_input_elems+i >= 1244160 || idx >= 131328 || idx < 0 || i >= 15360) {
                            //for (int c=0; c<inverse_patches_counts[i]; ++c)
                                printf("c=%d idx=%d i=%d \n", col, inverse_patches[col*num_input_elems+i], i);
                            //printf("\n");

                        //printf("i=%d col=%d/%d, idx=%d, err=%f\n", i, col, inverse_patches_counts[i], idx, err);
                        }
                        */

            if (idx > 0) {
                unwarped_error[i] += warped_error[idx];
                inverse_patches[inv_patch_idx] = 0;
                //++N;
            }
        }
        // if (inverse_patches_counts[i]>0) unwarped_error[i] /= inverse_patches_counts[i];
        // else printf("WTF?!");
        // unwarped_error[i] /= N;
    }
}
template<typename T>
void _cuda_convRestoreFromPatches(T* unwarped_error, const T* warped_error,
                                  const int* patchIdx, int* inverse_patches, int num_input_elems, int patch_dim) {
    int gridSize = (int)ceil((float)num_input_elems / THREADS_PER_BLOCK);
    __cuda_convRestoreFromPatches<T><<<gridSize, THREADS_PER_BLOCK>>>(unwarped_error, warped_error, patchIdx,
                                                                      inverse_patches, num_input_elems, patch_dim);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 3\n");
}

template void _cuda_convRestoreFromPatches(float* unwarped_error, const float* warped_error,
                                           const int* patchIdx, int* inverse_patches,
                                           int num_input_elems, int patch_dim);
template void _cuda_convRestoreFromPatches(double* unwarped_error, const double* warped_error,
                                           const int* patchIdx, int* inverse_patches,
                                           int num_input_elems, int patch_dim);

///////////////////////////////////////
template<typename T>
__global__ void __cuda_convUnwarpFrames(const T* output_warped, const T* bias, T* output,
                                        int output_dim, int filter_num, int shifts_num, int input_frames) {
    // int t = threadIdx.x + blockIdx.x * blockDim.x;
    int t = blockIdx.x;
    int f = threadIdx.x;
    if (t < input_frames && f < filter_num) {
        // int num_input_elems = input_frames * input_dim;
        // int row = 0;
        int row = f * shifts_num;
        // for (int f = 0; f < filter_num; ++f) {
        for (int s = 0; s < shifts_num; ++s) {
            int i = (t * shifts_num + s) * filter_num + f;

            output[t * output_dim + row] = output_warped[i] + bias[f];
            ++row;
        }
        //}
    }
}
template<typename T>
void _cuda_convUnwarpFrames(const T* output_warped, const T* bias, T* output,
                            int output_dim, int filter_num, int shifts_num, int input_frames) {
    // int gridSize = (int)ceil( (float) input_frames/THREADS_PER_BLOCK);
    //__cuda_convUnwarpFrames<T> <<<gridSize, THREADS_PER_BLOCK>>> (output_warped, bias, output, output_dim, filter_num, shifts_num, input_frames);
    __cuda_convUnwarpFrames<T><<<input_frames, filter_num>>>(output_warped, bias, output, output_dim, filter_num, shifts_num, input_frames);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 2\n");
}

template void _cuda_convUnwarpFrames(const float* output_warped, const float* bias, float* output,
                                     int output_dim, int filter_num, int shifts_num, int input_frames);
template void _cuda_convUnwarpFrames(const double* output_warped, const double* bias, double* output,
                                     int output_dim, int filter_num, int shifts_num, int input_frames);

///////////////////////////////////////

template<typename T>
__global__ void __cuda_convWarpFrames(const T* error_unwarped, T* output,
                                      int error_dim, int filter_num, int shifts_num, int input_frames) {
    // int t = threadIdx.x + blockIdx.x * blockDim.x;
    // if (t < input_frames) {
    int t = blockIdx.x;
    int f = threadIdx.x;

    if (t < input_frames && f < filter_num) {
        // int num_input_elems = input_frames * input_dim;
        int row = f * shifts_num;
        // for (int f = 0; f < filter_num; ++f) {
        for (int s = 0; s < shifts_num; ++s) {
            int i = (t * shifts_num + s) * filter_num + f;

            output[i] = error_unwarped[t * error_dim + row];
            ++row;
            /*
            if (t==0 && s==0) {
                printf("%d %d %d -> %f\n", f, i, row, output[i]);
            }
            */
        }
        //}
    }
}
template<typename T>
void _cuda_convWarpFrames(const T* error_unwarped, T* output,
                          int error_dim, int filter_num, int shifts_num, int input_frames) {
    // int gridSize = (int)ceil( (float) input_frames/THREADS_PER_BLOCK);
    //__cuda_convWarpFrames<T> <<<gridSize, THREADS_PER_BLOCK>>> (error_unwarped, output, error_dim, filter_num, shifts_num, input_frames);
    __cuda_convWarpFrames<T><<<input_frames, filter_num>>>(error_unwarped, output, error_dim, filter_num, shifts_num, input_frames);
    if (cudaSuccess != cudaGetLastError())
        printf("Error 1\n");
}

template void _cuda_convWarpFrames(const float* error_unwarped, float* output,
                                   int error_dim, int filter_num, int shifts_num, int input_frames);
template void _cuda_convWarpFrames(const double* error_unwarped, double* output,
                                   int error_dim, int filter_num, int shifts_num, int input_frames);
