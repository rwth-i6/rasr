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
#ifndef _COMPLEX_VECTOR_FUNCTION_HH
#define _COMPLEX_VECTOR_FUNCTION_HH

#include <Core/Extensions.hh>
#include <Core/Types.hh>
#include <Flow/Vector.hh>
#include <Math/Complex.hh>
#include "Node.hh"

namespace Signal {

/** returns the amplitude vector of the input alternating complex vector
 *  length of output is set according to the number of input complex numbers
 */
template<class T>
struct alternatingComplexVectorAmplitude {
    using argument_type = T;
    using result_type   = T;

    static std::string name() {
        return std::string("vector-alternating-complex-") + Core::Type<T>::name + "-amplitude";
    }

    /** x and result can be one object */
    void operator()(const std::vector<T>& x, std::vector<T>& result) {
        if (&x != &result)
            result.resize(x.size() / 2);

        Math::transformAlternatingComplex(x.begin(), x.end(), result.begin(), Math::pointerAbs<T>());

        result.resize(x.size() / 2);
    }
};

/** returns the absolute square vector of the input alternating complex vector
 *  length of output is set according to the number of input complex numbers
 */
template<class T>
struct alternatingComplexVectorAbsoluteSquare {
    using argument_type = T;
    using result_type   = T;

    static std::string name() {
        return std::string("vector-alternating-complex-") + Core::Type<T>::name + "-absolute-square";
    }

    /** x and result can be one object */
    void operator()(const std::vector<T>& x, std::vector<T>& result) {
        if (&x != &result) {
            result.resize(x.size() / 2);
        }

        Math::transformAlternatingComplex(x.begin(), x.end(), result.begin(), Math::pointerAbsSqr<T>());

        result.resize(x.size() / 2);
    }
};

/** returns the phase vector of the input alternating complex vector
 *  length of output is set according to the number of input complex numbers
 */
template<class T>
struct alternatingComplexVectorPhase {
    using argument_type = T;
    using result_type   = T;

    static std::string name() {
        return std::string("vector-alternating-complex-") + Core::Type<T>::name + "-phase";
    }

    /** x and result can be one object */
    void operator()(const std::vector<T>& x, std::vector<T>& result) {
        if (&x != &result)
            result.resize(x.size() / 2);

        Math::transformAlternatingComplex(x.begin(), x.end(), result.begin(), Math::pointerArg<T>());

        result.resize(x.size() / 2);
    }
};

/** returns the real part vector of the input alternating complex vector
 *  length of output is set according to the number of input complex numbers
 */
template<class T>
struct alternatingComplexVectorRealPart {
    using argument_type = T;
    using result_type   = T;

    static std::string name() {
        return std::string("vector-alternating-complex-") + Core::Type<T>::name + "-real-part";
    }

    /** x and result can be one object */
    void operator()(const std::vector<T>& x, std::vector<T>& result) {
        if (&x != &result)
            result.resize(x.size() / 2);

        Math::transformAlternatingComplex(x.begin(), x.end(), result.begin(), Math::pointerReal<T>());

        result.resize(x.size() / 2);
    }
};

/** returns the imaginary part vector of the input alternating complex vector
 *  length of output is set according to the number of input complex numbers
 */
template<class T>
struct alternatingComplexVectorImaginaryPart {
    using argument_type = T;
    using result_type   = T;

    static std::string name() {
        return std::string("vector-alternating-complex-") + Core::Type<T>::name + "-imaginary-part";
    }

    /** x and result can be one object */
    void operator()(const std::vector<T>& x, std::vector<T>& result) {
        if (&x != &result)
            result.resize(x.size() / 2);

        Math::transformAlternatingComplex(x.begin(), x.end(), result.begin(), Math::pointerImag<T>());

        result.resize(x.size() / 2);
    }
};

/** converts vector of real parts into alternating complex vector
 *  Imaginary parts are set to 0.
 *  length of output is set according to the size of input.
 */
template<class T>
struct vectorToAlternatingComplexVector {
    using argument_type = T;
    using result_type   = T;

    static std::string name() {
        return std::string("vector-") + Core::Type<T>::name +
               "-to-vector-alternating-complex-" + Core::Type<T>::name;
    }
    void operator()(const std::vector<T>& x, std::vector<T>& result) {
        result.resize(x.size() * 2);

        Math::transformToAlternatingComplex(x.begin(), x.end(), result.begin(), std::bind(Math::makeComplex<T>(), std::placeholders::_1, 0));
    }
};

/** converts alternating complex vector to vector of objects of the class complex<T>
 *  length of output is set according to the number of input complex numbers.
 */
template<class T>
struct alternatingComplexVectorToComplexVector {
    using argument_type = T;
    using result_type   = std::complex<T>;

    static std::string name() {
        return std::string("vector-alternating-complex-") + Core::Type<T>::name +
               "-to-vector-complex-" + Core::Type<T>::name;
    }

    void operator()(const std::vector<T>& x, std::vector<std::complex<T>>& result) {
        result.resize(x.size() / 2);
        Math::transformAlternatingComplex(x.begin(), x.end(), result.begin(), Core::identity<std::complex<T>>());
    }
};

/** converts vector of objects of class complex<T> to alternating complex vector
 *  length of output is set according to the number size of input.
 */
template<class T>
struct complexVectorToAlternatingComplexVector {
    using argument_type = std::complex<T>;
    using result_type   = T;

    static std::string name() {
        return std::string("vector-complex-") + Core::Type<T>::name +
               "-to-vector-alternating-complex-" + Core::Type<T>::name;
    }

    void operator()(const std::vector<std::complex<T>>& x, std::vector<T>& result) {
        result.resize(x.size() * 2);
        Math::transformToAlternatingComplex(x.begin(), x.end(), result.begin(), Core::identity<std::complex<T>>());
    }
};

/** ComplexVectorFunctionNode is a node class
 *  for different complex unary functions
 */
template<class Function>
class ComplexVectorFunctionNode : public SleeveNode {
public:
    typedef typename Function::argument_type ArgumentType;
    typedef typename Function::result_type   ResultType;

private:
    Function function_;

public:
    static std::string filterName() {
        return "signal-" + Function::name();
    }

    ComplexVectorFunctionNode(const Core::Configuration& c)
            : Core::Component(c),
              SleeveNode(c) {}

    virtual ~ComplexVectorFunctionNode() {}

    virtual bool configure() {
        Core::Ref<const Flow::Attributes> a = getInputAttributes(0);
        if (!configureDatatype(a, Flow::Vector<ResultType>::type()))
            return false;
        return putOutputAttributes(0, a);
    }

    virtual bool setParameter(const std::string& name, const std::string& value) {
        return false;
    }

    virtual bool work(Flow::PortId p) {
        Flow::DataPtr<Flow::Vector<ArgumentType>> in;
        if (!getData(0, in))
            return putData(0, in.get());

        Flow::Vector<ResultType>* out = new Flow::Vector<ResultType>;
        function_(*in, *out);
        out->setTimestamp(*in);
        return putData(0, out);
    }
};

}  // namespace Signal

#endif  // _COMPLEX_VECTOR_FUNCTION_HH
