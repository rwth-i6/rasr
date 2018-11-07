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
#ifndef _SIGNAL_VECTOR_CUT_HH
#define _SIGNAL_VECTOR_CUT_HH


#include <Core/Parameter.hh>
#include <Core/Utility.hh>

#include <Flow/Data.hh>
#include <Flow/Vector.hh>
#include <Flow/Node.hh>

namespace Signal {

    /** VectorCutLength cuts out [first..last] interval of input vector
     *
     * first: first element of the interval in continous unit depending on previous nodes
     * last: last element of the interval in continous unit depending on previous nodes
     */

    template<class T>
    class VectorCutLength {
    public:

        typedef T Value;

    protected:

        f32 first_;
        f32 last_;

        f64 sampleRate_;

    protected:

        /** first
         * @return first index within the interval to send
         */

        virtual u32 first() const {
            return (u32)rint(first_ * sampleRate_);
        }

        /** last
         * @return last index within the interval to send
         */

        virtual u32 last() const {
            return (u32)rint(last_ * sampleRate_);
        }

        virtual void init(const std::vector<Value> &in) {}

    public:
        static std::string name() {
            return Core::Type<Value>::name + std::string("-cut-length");
        }

        VectorCutLength() : first_(0), last_(0), sampleRate_(0) {}

        virtual ~VectorCutLength() {}

        void apply(std::vector<Value> &in) {

            init(in);

            u32 f = first();
            u32 l = std::min(last(), u32(in.size()) - 1);
            hope(!in.empty() && f <= l);

            if (f == 0) { // do not copy 'in' if first == 0

                in.resize(l + 1);

            } else {

                std::vector<Value> out(in.begin() + f, in.begin() + l + 1);
                in.swap(out);

            }
        }

        void setFirst(const f32 first) { first_ = first; }

        void setLast(const f32 last) { last_ = last; }

        void setSampleRate(const f64 sampleRate) { sampleRate_ = sampleRate; }
    };


    /** VectorCutRelativeLength cuts out [first..last] interval of input vector
     *
     * first: first element in percents of lenght of input vector
     * last: last element in percents of length of input vector
     */

    template<class T>
    class VectorCutRelativeLength : public VectorCutLength<T> {

    public:

        typedef T Value;

    private:

        u32 inputSize_;

    protected:

        /** first
         * @return first index within the interval to send
         */

        virtual u32 first() const {
            ensure(this->first_ >= 0.0 && this->first_ <= 1.0);
            return (u32)rint((inputSize_ - 1) * this->first_);
        }

        /** last
         * @return last index within the interval to send
         */

        virtual u32 last() const {
            ensure(this->last_ >= 0.0 && this->last_ <= 1.0);
            return (u32)rint((inputSize_ - 1) * this->last_);
        }

        virtual void init(const std::vector<Value> &in) { inputSize_ = in.size(); }

    public:
        static std::string name() {
            return Core::Type<Value>::name + std::string("-cut-relative-length");
        }

        VectorCutRelativeLength() : inputSize_(0) {}

        virtual ~VectorCutRelativeLength() {}
    };


    /** VectorCutRelativeSurface cuts out [first..last] interval of input vector
     *
     * first: first element in percents of surface of input vector
     * last: last first element in percents of surface of input vector
     */

    template<class T>
    class VectorCutRelativeSurface : public VectorCutLength<T> {

    public:

        typedef T Value;

    private:

        std::vector<Value> integral_;

    protected:

        virtual void init(const std::vector<Value> &in) {

            integral_.resize(in.size());

            std::transform(in.begin(), in.end(), integral_.begin(), Core::absoluteValue<Value>());

            partial_sum(integral_.begin(), integral_.end(), integral_.begin());
        }

        /** first
         * @return first index within the interval to send
         */

        virtual u32 first() const {
            ensure(this->first_ >= 0.0 && this->first_ <= 1.0);

            if (this->first_ == 0.0)
                return 0;

            T limit = integral_.back() * this->first_;
            return std::distance(integral_.begin(),
                                 find_if(integral_.begin(), integral_.end(),
                                         std::bind2nd(std::greater_equal<Value>(), limit)));
        }

        /** last
         * @return last index within the interval to send
         */

        virtual u32 last() const {
            ensure(this->last_ >= 0.0 && this->last_ <= 1.0);

            if (this->last_ == 1.0)
                return integral_.size() - 1;

            T limit = integral_.back() * this->last_;
            return std::distance(integral_.begin(),
                                 find_if(integral_.begin(), integral_.end(),
                                         std::bind2nd(std::greater_equal<Value>(), limit)));
        }

    public:
        static std::string name() {
            return Core::Type<Value>::name + std::string("-cut-relative-surface");
        }

        VectorCutRelativeSurface() {}

        virtual ~VectorCutRelativeSurface() {}
    };

    /** VectorCutNode */

    const Core::ParameterFloat paramVectorCutFirst("first", "first element", 0, 0);
    const Core::ParameterFloat paramVectorCutLast("last", "last element", 0, 0);

    template<class Algorithm>
    class VectorCutNode : public Flow::SleeveNode, public Algorithm {

    public:

        static std::string filterName() { return "signal-vector-" + Algorithm::name(); }

        VectorCutNode(const Core::Configuration &c) :
            Core::Component(c), SleeveNode(c)
        {
            this->setFirst(paramVectorCutFirst(c));
            this->setLast(paramVectorCutLast(c));
        }

        virtual ~VectorCutNode() {}

        virtual bool setParameter(const std::string &name, const std::string &value) {

            if (paramVectorCutFirst.match(name))
                this->setFirst(paramVectorCutFirst(value));
            else if (paramVectorCutLast.match(name))
                this->setLast(paramVectorCutLast(value));
            else
                return false;

            return true;
        }

        virtual bool configure() {

            Core::Ref<const Flow::Attributes> a = getInputAttributes(0);
            if (!configureDatatype(a, Flow::Vector<f32>::type()))
                return false;

            Algorithm::setSampleRate(atof(a->get("sample-rate").c_str()));

            return putOutputAttributes(0, a);
        }

        virtual bool work(Flow::PortId p) {

            Flow::DataPtr<Flow::Vector<typename Algorithm::Value> > in;
            if (!getData(0, in))
                return SleeveNode::putData(0, in.get());

            in.makePrivate();
            Algorithm::apply(*in);

            return putData(0, in.get());
        }
    };



}

#endif // _SIGNAL_VECTOR_CUT_HH
