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
#ifndef PROFILEMATRIX_HH_
#define PROFILEMATRIX_HH_

class MatrixProfiler : public Core::Application
{
    static const Core::ParameterInt paramNumberOfRepetitions;
protected:
    u32 nRepetitions_;
protected:
    template<typename T>
    void resetMatrices(Math::FastMatrix<T> &C, Math::CudaMatrix<T> &G, bool randomInit=true) const;

    template<typename T>
    void initMatrices(Math::FastMatrix<T> &C, Math::CudaMatrix<T> &G, int nRows, int nCols, bool randomInit=true) const;

    void logTiming(double timeCpu, double timeGpu) const;

public:
    virtual std::string getUsage() const { return "short program to test Math features\n"; }

    MatrixProfiler();

    void profileExp(int nRows, int nCols) const ;

    void profileDot(int nRows, int nCols) const ;

    void profileAdd(int nRows, int nCols) const ;

    void profileSigmoid(int nRows, int nCols, float gamma) const ;

    void profileMaxOfColumns(int nRows, int nCols) const;

    void profileAddSummedRows(int nRows, int nCols) const;

    void profileSoftmax(int nRows, int nCols) const ;

    void profileSoftmaxDetailed(int nRows, int nCols, int tmpDimension) const ;

    void profileMatrixMultiplication(int l, int m, int n) const ;

    void profileMatrixMultiplicationMixed(int l, int m, int n) const ;

    void profileSync(int nRows, int nCols) const;

    void profileCast(int nRows, int nCols) const;

    int main(const std::vector<std::string> &arguments);

};

#endif /* PROFILEMATRIX_HH_ */
