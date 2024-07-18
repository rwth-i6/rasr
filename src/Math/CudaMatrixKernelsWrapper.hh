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
#ifndef CUDAMATRIXKERNELSWRAPPER_HH_
#define CUDAMATRIXKERNELSWRAPPER_HH_

#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
/**
 * Macro CUDACALL inserts the first parameter, if MODULE_CUDA is enabled.
 * Otherwise, a critical error is raised.
 */
#ifdef MODULE_CUDA
#include "CudaMatrixKernels.hh"
#define CUDACALL(function, functionName) \
    function;
#else
#define CUDACALL(function, functionName)                                           \
    const char* msg = "Calling CUDA kernel '%s' in a binary without GPU support!"; \
    Core::Application::us()->criticalError(msg, functionName);
#endif

namespace Math {

namespace Cuda {

// exponential function

template<typename T>
inline void exp(T* devPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_exp<T>(devPtr, nRows, nColumns)), "exp");
}

inline void cast(int nElements, const float* x, double* y) {
    CUDACALL(_cuda_cast(nElements, x, y), "cast");
}

// logarithm

template<typename T>
inline void log(T* devPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_log<T>(devPtr, nRows, nColumns)), "log");
}

// power

template<typename T>
inline void pow(T* devPtr, unsigned int nRows, unsigned int nColumns, T exponent) {
    CUDACALL((_cuda_pow<T>(devPtr, nRows, nColumns, exponent)), "pow");
}

// addSummedRows

template<typename T>
inline void addSummedRows(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale) {
    CUDACALL((_cuda_addSummedRows<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale)),
             "addSummedRows");
}

// faster version of addSummedRows

template<typename T>
inline void addSummedRows(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                          T* tmpDevPtr, unsigned int tmpRows, const T scale) {
    CUDACALL((_cuda_addSummedRows<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, tmpDevPtr, tmpRows, scale)),
             "addSummedRows");
}

// addSummedColumns

template<typename T, typename S>
inline void addSummedColumns(T* vectorDevPtr, const S* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const S scale) {
    CUDACALL((_cuda_addSummedColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale)),
             "addSummedColumns");
}

// addSquaredSummedColumns

template<typename T>
inline void addSquaredSummedColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns, const T scale) {
    CUDACALL((_cuda_addSquaredSummedColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, scale)),
             "addSquaredSummedColumns");
}

// tanh

template<typename T>
inline void tanh(T* devPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_tanh<T>(devPtr, nRows, nColumns)), "tanh");
}

// sigmoid

template<typename T>
inline void sigmoid(T gamma, T* devPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_sigmoid<T>(gamma, devPtr, nRows, nColumns)), "sigmoid");
}

// softmax

template<typename T>
inline void softmax(T* devPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_softmax<T>(devPtr, nRows, nColumns)), "softmax");
}

// elementwise multiplication

template<typename T>
inline void elementwiseMultiplication(T* a, T* b, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_elementwiseMultiplication<T>(a, b, nRows, nColumns)),
             "elementwiseMultiplication");
}

// elementwise division

template<typename T>
inline void elementwiseDivision(T* a, T* b, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_elementwiseDivision<T>(a, b, nRows, nColumns)), "elementwiseDivision");
}

// add constant elementwise

template<typename T>
inline void addConstantElementwise(T a, T* b, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_addConstantElementwise<T>(a, b, nRows, nColumns)), "addConstantElementwise");
}

// elementwiseMultiplicationWithSigmoidDerivative

template<typename T>
inline void elementwiseMultiplicationWithSigmoidDerivative(T* a, T* b, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_elementwiseMultiplicationWithSigmoidDerivative<T>(a, b, nRows, nColumns)),
             "elementwiseMultiplicationWithSigmoidDerivative");
}

// elementwiseMultiplicationWithTanhDerivative

template<typename T>
inline void elementwiseMultiplicationWithTanhDerivative(T* a, T* b, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_elementwiseMultiplicationWithTanhDerivative<T>(a, b, nRows, nColumns)),
             "elementwiseMultiplicationWithTanhDerivative");
}

// multiplicationWithSoftmaxDerivative

template<typename T>
inline void multiplicationWithSoftmaxDerivative(T* a, T* b, T* c, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_multiplicationWithSoftmaxDerivative<T>(a, b, c, nRows, nColumns)),
             "multiplicationWithSoftmaxDerivative");
}

// elementwiseMultiplicationWithRectifiedDerivative

template<typename T>
inline void elementwiseMultiplicationWithRectifiedDerivative(T* a, T* b, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_elementwiseMultiplicationWithRectifiedDerivative<T>(a, b, nRows, nColumns)),
             "elementwiseMultiplicationWithRectifiedDerivative");
}

// elementwiseMultiplicationWithEluDerivative

template<typename T>
inline void elementwiseMultiplicationWithEluDerivative(T* a, T* b, T alpha, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_elementwiseMultiplicationWithEluDerivative<T>(a, b, alpha, nRows, nColumns)),
             "elementwiseMultiplicationWithEluDerivative");
}

// ELU

template<typename T>
inline void elu(T* devPtr, T value, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_elu<T>(devPtr, value, nRows, nColumns)), "elu");
}

// getMaxOfColumns

template<typename T>
inline void getMaxOfColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_getMaxOfColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
             "getMaxOfColumns");
}

// faster getMaxOfColumns

template<typename T>
inline void getMaxOfColumns(T* vectorDevPtr, const T* matrixDevPtr, unsigned int nRows, unsigned int nColumns,
                            T* tmpDevPtr, unsigned int tmpRows) {
    CUDACALL((_cuda_getMaxOfColumns<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns, tmpDevPtr, tmpRows)),
             "getMaxOfColumns");
}

// addToAllColumns

template<typename T>
inline void addToAllColumns(T* a, T* b, unsigned int nRows, unsigned int nColumns, T alpha) {
    CUDACALL((_cuda_addToAllColumns<T>(a, b, nRows, nColumns, alpha)),
             "addToAllColumns");
}

// addToAllColumnsWithOffset

template<typename T>
inline void addToAllColumnsWithOffset(T* a, T* b, unsigned int nRowsMat, unsigned nRowsVec, unsigned int nColumns, T alpha) {
    CUDACALL((_cuda_addToAllColumnsWithOffset<T>(a, b, nRowsMat, nRowsVec, nColumns, alpha)),
             "addToAllColumnsWithOffset");
}

// addToAllRows

template<typename T>
inline void addToAllRows(T* a, T* b, unsigned int nRows, unsigned int nColumns, T alpha) {
    CUDACALL((_cuda_addToAllRows<T>(a, b, nRows, nColumns, alpha)),
             "addToAllRows");
}

// multiplyColumnsByScalars

template<typename T>
inline void multiplyColumnsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_multiplyColumnsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
             "multiplyColumnsByScalars");
}

// divideColumnsByScalars

template<typename T>
inline void divideColumnsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_divideColumnsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
             "divideColumnsByScalars");
}

// multiplyRowsByScalars

template<typename T>
inline void multiplyRowsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_multiplyRowsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
             "multiplyRowsByScalars");
}

// divideRowsByScalars

template<typename T>
inline void divideRowsByScalars(const T* vectorDevPtr, T* matrixDevPtr, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_divideRowsByScalars<T>(vectorDevPtr, matrixDevPtr, nRows, nColumns)),
             "divideRowsByScalars");
}

template<typename T>
inline void sign(T* out, const T* in, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_sign<T>(out, in, nRows, nColumns)), "sign");
}

// fill

template<typename T>
inline void fill(T* devPtr, T value, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_fill<T>(devPtr, value, nRows, nColumns)), "fill");
}

// ensure minimal value

template<typename T>
inline void ensureMinimalValue(T* devPtr, T value, unsigned int nRows, unsigned int nColumns) {
    CUDACALL((_cuda_ensureMinimalValue<T>(devPtr, value, nRows, nColumns)),
             "ensureMinimalValue");
}
// number of classification errors

template<typename T>
inline void nClassificationErrors(T* devPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* errorBuf) {
    CUDACALL((_cuda_nClassificationErrors<T>(devPtr, nRows, nColumns, alignmentDevPtr, errorBuf)),
             "nClassificationErrors");
}

// cross-entropy objective function
template<typename T>
inline void crossEntropyObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* resultDev) {
    CUDACALL((_cuda_crossEntropyObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev)),
             "crossEntropyObjectiveFunction");
}

// weighted cross-entropy objective function
template<typename T>
inline void weightedCrossEntropyObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* resultDev, T* weights) {
    CUDACALL((_cuda_weightedCrossEntropyObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev, weights)),
             "weightedCrossEntropyObjectiveFunction");
}

// squared error objective function
template<typename T>
inline void squaredErrorObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* resultDev) {
    CUDACALL((_cuda_squaredErrorObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev)),
             "squaredErrorObjectiveFunction");
}
// weighted squared error objective function
template<typename T>
inline void weightedSquaredErrorObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* resultDev, T* weights) {
    CUDACALL((_cuda_weightedSquaredErrorObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev, weights)),
             "weightedSquaredErrorObjectiveFunction");
}

// binary divergence objective function
template<typename T>
inline void binaryDivergenceObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* resultDev) {
    CUDACALL((_cuda_binaryDivergenceObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev)),
             "binaryDivergenceObjectiveFunction");
}

template<typename T>
inline void weightedBinaryDivergenceObjectiveFunction(T* matrixPtr, unsigned int nRows, unsigned int nColumns, unsigned int* alignmentDevPtr, T* resultDev, T* weights) {
    CUDACALL((_cuda_weightedBinaryDivergenceObjectiveFunction<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, resultDev, weights)),
             "weightedBinaryDivergenceObjectiveFunction");
}

template<typename T>
inline void binaryDivergenceSoftmaxGradient(T* matrixPtr, unsigned int nRows, unsigned int nColumns, const T* outputDevPtr, const unsigned int* alignmentDevPtr) {
    CUDACALL((_cuda_binaryDivergenceSoftmaxGradient<T>(matrixPtr, nRows, nColumns, outputDevPtr, alignmentDevPtr)),
             "binaryDivergenceSoftmaxGradient");
}

// Kronecker Delta
template<typename T>
inline void addKroneckerDelta(T* matrixPtr, unsigned int nRows, unsigned int nColumns, const unsigned int* alignmentDevPtr, const T scale) {
    CUDACALL((_cuda_addKroneckerDelta<T>(matrixPtr, nRows, nColumns, alignmentDevPtr, scale)),
             "addKroneckerDelta");
}
// second order features
template<typename T>
inline void appendSecondOrderFeatures(const T* X, unsigned int nRowsX, unsigned int nColumnsX, T* Y, unsigned int nRowsY, unsigned int offset) {
    CUDACALL((_cuda_appendSecondOrderFeatures<T>(X, nRowsX, nColumnsX, Y, nRowsY, offset)),
             "appendSecondOrderFeatures");
}
// third order features
template<typename T>
inline void appendThirdOrderFeatures(const T* X, unsigned int nRowsX, unsigned int nColumnsX, T* Y, unsigned int nRowsY, unsigned int offset) {
    CUDACALL((_cuda_appendThirdOrderFeatures<T>(X, nRowsX, nColumnsX, Y, nRowsY, offset)),
             "appendThirdOrderFeatures");
}

// dropout

template<typename T>
inline void dropout(T* X, const T* mask, unsigned int nRows, unsigned int nColumns, T dropoutProbability) {
    CUDACALL((_cuda_dropout<T>(X, mask, nRows, nColumns, dropoutProbability)), "dropout");
}

// l1 regularization with clipping
template<typename T>
inline void l1clipping(T* X, unsigned int nRows, unsigned int nColumns, T value) {
    CUDACALL((_cuda_l1clipping<T>(X, nRows, nColumns, value)), "l1clipping");
}

// clip
template<typename T>
inline void clip(T* X, unsigned int nElements, T maxAbsValue) {
    CUDACALL((_cuda_clip<T>(X, nElements, maxAbsValue)), "clip");
}

// max pooling layer
template<typename T>
inline void addPoolingMax(const T* input, T* output, unsigned int* argmax,
                          unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out,
                          unsigned int poolingSize, bool poolingAbs) {
    CUDACALL((_cuda_addPoolingMax<T>(input, output, argmax, nColumns, nRows_in, nRows_out, poolingSize, poolingAbs)),
             "addPoolingMax");
}

template<typename T>
inline void backpropPoolingMax(T* output, const unsigned int* argmax, const T* error,
                               unsigned int nColumns, unsigned int nRows_err) {
    CUDACALL((_cuda_backpropPoolingMax<T>(output, argmax, error, nColumns, nRows_err)),
             "backpropPoolingMax");
}

// pnorm pooling layer
template<typename T>
inline void addPoolingPnorm(const T* input, T* output,
                            unsigned int nColumns, unsigned int nRows_in, unsigned int nRows_out,
                            unsigned int poolingSize, unsigned int pnorm) {
    CUDACALL((_cuda_addPoolingPnorm<T>(input, output, nColumns, nRows_in, nRows_out, poolingSize, pnorm)),
             "addPoolingPnorm");
}

template<typename T>
inline void backpropPoolingPnorm(T* output, const T* error,
                                 unsigned int nColumns, unsigned int nRows_err,
                                 unsigned int poolingSize, unsigned int pnorm) {
    CUDACALL((_cuda_backpropPoolingPnorm<T>(output, error, nColumns, nRows_err, poolingSize, pnorm)),
             "backpropPoolingPnorm");
}

// convolutional processing
template<typename T>
inline void convExtractPatches(const T* input, const int* patchIdx, T* patches,
                               int* inverse_patches,
                               s32 input_frames, s32 input_dim, s32 shifts_num, s32 shifts_dim, s32 patch_dim) {
    CUDACALL((_cuda_convExtractPatches(input, patchIdx, patches, inverse_patches,
                                       input_frames, input_dim, shifts_num, shifts_dim, patch_dim)),
             "convExtractPatches");
}

template<typename T>
inline void convRestoreFromPatches(T* unwarped_error, const T* warped_error,
                                   const int* patchIdx, int* inverse_patches,
                                   s32 num_input_elems, int patch_dim) {
    CUDACALL((_cuda_convRestoreFromPatches(unwarped_error, warped_error, patchIdx,
                                           inverse_patches, num_input_elems, patch_dim)),
             "convRestoreFromPatches");
}

template<typename T>
inline void convUnwarpFrames(const T* output_warped, const T* bias, T* output,
                             s32 output_dim, s32 filter_num, s32 shifts_num, s32 input_frames) {
    CUDACALL((_cuda_convUnwarpFrames(output_warped, bias, output, output_dim, filter_num, shifts_num, input_frames)),
             "convUnwarpFrames");
}

template<typename T>
inline void convWarpFrames(const T* error_unwarped, T* output,
                           s32 error_dim, s32 filter_num, s32 shifts_num, s32 input_frames) {
    CUDACALL((_cuda_convWarpFrames(error_unwarped, output, error_dim, filter_num, shifts_num, input_frames)),
             "convWarpFrames");
}

}  // namespace Cuda

}  // namespace Math

#endif /* CUDAMATRIXKERNELSWRAPPER_HH_ */
