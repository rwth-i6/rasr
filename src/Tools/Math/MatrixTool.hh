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
#ifndef MATRIXTOOL_H_
#define MATRIXTOOL_H_

#include <cmath>

#include <Core/Application.hh>
#include <Math/Matrix.hh>
#include <Math/Module.hh>


namespace Math {

class MatrixTool : public Core::Application {
protected:
    static const Core::ParameterString paramNewFile;
    static const Core::ParameterString paramFile;
    static const Core::ParameterString paramSummand;
    static const Core::ParameterFloat paramScalingFactor;
    static const Core::ParameterInt paramNumberOfRows;
    static const Core::ParameterInt paramNumberOfColumns;
    static const Core::ParameterInt paramMinColumn;
protected:
    // actions
    template<typename T>
    void write(const Math::Matrix<T> &matrix) const;

    template<typename T>
    void scale(Math::Matrix<T> &matrix) const;

    template<typename T>
    void max(Math::Matrix<T> &matrix) const;

    template<typename T>
    void l2norm(Math::Matrix<T> &matrix) const;

    template<typename T>
    void add(Math::Matrix<T> &matrix) const;

    template<typename T>
    void addMultiple(Math::Matrix<T> &matrix) const;

    template<typename T>
    void mult(Math::Matrix<T> &matrix) const;

    template<typename T>
    void expand(Math::Matrix<T> &matrix) const;

    template<typename T>
    void getColumns(Math::Matrix<T> &matrix) const;

    template<typename T>
    void join(Math::Matrix<T> &matrix) const;

    template<typename T>
    void exp(Math::Matrix<T> &matrix) const;

    template<typename T>
    void logarithm(Math::Matrix<T> &matrix) const;

    template<typename T>
    bool actionLoop(const std::vector<std::string> &actions, Math::Matrix<T> &matrix) const;

    void usage(std::string action = "") const;
public:
    MatrixTool();
    virtual ~MatrixTool();
    int main(const std::vector<std::string> &arguments);
};

template<typename T>
void MatrixTool::write(const Math::Matrix<T> &matrix) const {
    std::string filename = paramNewFile(select("write"));
    require_ne(filename, "");
    if (!Math::Module::instance().formats().write(filename, matrix))
        error("could not write matrix to file ") << filename;
    else
        log("matrix written to file ") << filename;
}

template<typename T>
void MatrixTool::scale(Math::Matrix<T> &matrix) const {
    T factor = paramScalingFactor(select("scale"));
    matrix = factor * matrix;
    log("matrix scaled by ") << factor;
}

template<typename T>
void MatrixTool::max(Math::Matrix<T> &matrix) const {
    T maxElement = matrix.maxElement();
    log("maximum element: ") << maxElement;
}

template<typename T>
void MatrixTool::l2norm(Math::Matrix<T> &matrix) const {
    log("l2 norm: ") << matrix.l2Norm();
}


template<typename T>
void MatrixTool::add(Math::Matrix<T> &matrix) const {
    Math::Matrix<T> summand;
    if (!Math::Module::instance().formats().read(paramFile(select("add")), summand))
        error("could not read matrix from file ") << paramFile(select("add"));
    else
        log("matrix ") << paramFile(select("add")) << " added";
    matrix += summand;
}

template<typename T>
void MatrixTool::addMultiple(Math::Matrix<T> &matrix) const {
    std::vector<std::string> filenames = Core::split(paramFile(select("add")), ",");
    Math::Matrix<T> summand;
    for (std::vector<std::string>::const_iterator fn = filenames.begin(); fn != filenames.end(); fn++){
        if (!Math::Module::instance().formats().read(*fn, summand))
            error("could not read matrix from file ") << *fn;
        else
            log("matrix ") << *fn << " added";
        matrix += summand;
    }
}

template<typename T>
void MatrixTool::mult(Math::Matrix<T> &matrix) const {
    Math::Matrix<T> rfactor;
    Math::Matrix<T> old = matrix;
    if (!Math::Module::instance().formats().read(paramFile(select("mult")), rfactor))
        error("could not read matrix from file ") << paramFile(select("mult"));
    else
        log("matrix ") << paramFile(select("mult")) << " multiplied (from right hand side)";

    matrix = old * rfactor;
}


template<typename T>
void MatrixTool::exp(Math::Matrix<T> &matrix) const {
    for (uint i = 0; i < matrix.nRows(); ++i){
        for (uint j = 0; j < matrix.nColumns(); ++j){
            matrix[i][j] = std::exp(matrix[i][j]);
        }
    }
}

template<typename T>
void MatrixTool::logarithm(Math::Matrix<T> &matrix) const {
    for (uint i = 0; i < matrix.nRows(); ++i){
        for (uint j = 0; j < matrix.nColumns(); ++j){
            matrix[i][j] = std::log(matrix[i][j]);
        }
    }
}

template<typename T>
void MatrixTool::expand(Math::Matrix<T> &matrix) const {
    matrix.resize(paramNumberOfRows(select("expand")), paramNumberOfColumns(select("expand")));
}

template<typename T>
void MatrixTool::getColumns(Math::Matrix<T> &matrix) const {
    int cLo = paramMinColumn(select("get-columns"));
    Math::Matrix<T> tmp = matrix;
    matrix.resize(tmp.nRows(), paramNumberOfColumns(select("get-columns")));
    for (u32 i = 0; i < tmp.nRows(); ++i){
        for (u32 j = 0; j < tmp.nColumns(); ++j)
            matrix[i][j] = tmp[i][cLo + j];
    }
}

template<typename T>
void MatrixTool::join(Math::Matrix<T> &matrix) const {
    Math::Matrix<T> B;
    if (!Math::Module::instance().formats().read(paramFile(select("join")), B))
        error("could not read matrix from file ") << paramFile(select("join"));

    if (B.nRows() != matrix.nRows()){
        error("dimension mismatch");
    }
    u32 oldColumns = matrix.nColumns();
    matrix.resize(matrix.nRows(), matrix.nColumns() + B.nColumns());
    for (u32 i = 0; i < B.nRows(); ++i){
        for (u32 j = 0; j < B.nColumns(); ++j)
            matrix[i][j + oldColumns] = B[i][j];
    }
    log("matrix ") << paramFile(select("join")) << " joined (to the right)";
}


template<typename T>
bool MatrixTool::actionLoop(const std::vector<std::string> &actions, Math::Matrix<T> &matrix) const {
    log("processing matrix of size: ") << matrix.nRows() << "x" << matrix.nColumns();
    for (std::vector<std::string>::const_iterator action = actions.begin(); action != actions.end(); action++){
        log("action: ") << *action;
        if (*action == "write"){
            write(matrix);
        }
        else if (*action == "scale"){
            scale(matrix);
        }
        else if (*action == "max"){
            max(matrix);
        }
        else if (*action == "add"){
            add(matrix);
        }
        else if (*action == "add-multiple"){
            addMultiple(matrix);
        }
        else if (*action == "mult"){
            mult(matrix);
        }
        else if (*action == "l2-norm"){
            l2norm(matrix);
        }
        else if (*action == "exp"){
            exp(matrix);
        }
        else if (*action == "log"){
            logarithm(matrix);
        }
        else if (*action == "get-columns"){
            getColumns(matrix);
        }
        else if (*action == "join"){
            join(matrix);
        }
        else if (*action == "expand"){
            expand(matrix);
        }
        else{
            error("unknown action: ") << *action;
            return false;
        }
    }
    return true;
}

}

#endif /* MATRIXTOOL_H_ */
