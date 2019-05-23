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
#include "MatrixTool.hh"
#include <Core/StringUtilities.hh>

using namespace Math;

const Core::ParameterString MatrixTool::paramNewFile("new-file", "new filename", "");

const Core::ParameterString MatrixTool::paramFile("file", "filename", "");

const Core::ParameterFloat MatrixTool::paramScalingFactor("scaling-factor", "scaling factor");

const Core::ParameterInt MatrixTool::paramNumberOfRows("number-of-rows", "number of rows");

const Core::ParameterInt MatrixTool::paramNumberOfColumns("number-of-columns", "number of columns");

const Core::ParameterInt MatrixTool::paramMinColumn("min-column", "lower column index");

MatrixTool::MatrixTool() {
    INIT_MODULE(Math);
    setTitle("matrix-tool");
}

MatrixTool::~MatrixTool() {}

void MatrixTool::usage(std::string action) const {
    if (action == "write") {
        std::cout << "action write: write parameter to file" << std::endl;
        std::cout << "parameter: \"new-file\"" << std::endl;
    }
    else if (action == "scale") {
        std::cout << "action scale: scale matrix" << std::endl;
        std::cout << "parameter: \"scaling-factor\"" << std::endl;
    }
    else if (action == "max") {
        std::cout << "action max: compute abs max of matrix" << std::endl;
        std::cout << " no parameters" << std::endl;
    }
    else if (action == "l2norm") {
        std::cout << "action l2norm: compute l2norm of matrix" << std::endl;
        std::cout << " no parameters" << std::endl;
    }
    else if (action == "add") {
        std::cout << "action add:  add another matrix" << std::endl;
        std::cout << "parameter \"file"
                     "\" (name of summand matrix file)"
                  << std::endl;
    }
    else if (action == "expand") {
        std::cout << "action expand:  expand matrix with zeros" << std::endl;
        std::cout << "parameters \"number-of-rows\", \"number-of-columns\"" << std::endl;
    }
    else if (action == "exp") {
        std::cout << "action exp: take exp of all entries" << std::endl;
    }
    else if (action == "log") {
        std::cout << "action exp: take log of all entries" << std::endl;
    }
    else {
        std::cout << "usage: " << basename_ << " [filename] [precision] [actions]" << std::endl;
        std::cout << "\t where" << std::endl;
        std::cout << "\t precision is \"f32\" or \"f64\"" << std::endl;
        std::cout << "\t actions is a comma-separated sequence of commands" << std::endl;
        std::cout << "\t available actions:" << std::endl;
        std::cout << "\t\twrite, scale, max, l2norm, add, expand, exp, log" << std::endl;
    }
}

int MatrixTool::main(const std::vector<std::string>& arguments) {
    if (arguments.size() == 2 && arguments.at(0) == "help") {
        usage(arguments.at(1));
        return 1;
    }
    else if (arguments.size() < 3) {
        usage();
        return 1;
    }
    std::string              filename  = arguments.at(0);
    std::string              precision = arguments.at(1);
    std::vector<std::string> actions   = Core::split(arguments.at(2), ",");

    bool success = true;
    if (precision == "f32") {
        Math::Matrix<f32> matrix;
        if (!Math::Module::instance().formats().read(filename, matrix)) {
            error("could not read matrix from file ") << matrix;
            success = 1;
        }
        else {
            success = actionLoop(actions, matrix);
        }
    }
    else if (precision == "f64") {
        Math::Matrix<f64> matrix;
        if (!Math::Module::instance().formats().read(filename, matrix)) {
            error("could not read matrix from file ") << matrix;
            success = 1;
        }
        else {
            success = actionLoop(actions, matrix);
        }
    }
    else {
        error("unknown precision: ") << precision;
        return 1;
    }
    int returnVal = success ? 0 : 1;
    return returnVal;
}

APPLICATION(MatrixTool)
