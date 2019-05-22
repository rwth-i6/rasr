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
#ifndef _SIGNAL_HISTOGRAM_HH
#define _SIGNAL_HISTOGRAM_HH

#include <Core/BinaryStream.hh>
#include <Core/XmlBuilder.hh>
#include <Flow/Vector.hh>
#include <numeric>
#include "LookupTable.hh"

namespace Signal {

/**
 *  Histogram: supports
 *   estimation of histograms
 *   calculation of probability distribution function (pdf)
 *   calculation of cummulative probability distribution function (cdf)
 *   calculation of percentiles
 */
template<class Value>
class Histogram : public LookupTable<f32, Value> {
    typedef LookupTable<f32, Value> Precursor;

public:
    typedef typename Precursor::ValueType Weight;
    typedef typename Precursor::ValueType Probability;

public:
    Histogram(const Value bucketSize = 0)
            : Precursor(bucketSize) {}
    Histogram(const Value bucketSize, const Value min, const Value max)
            : Precursor(bucketSize, min, max) {}
    void accumulate(const Value v, Weight weight = 1) {
        *Precursor::insert(v, 0) += weight;
    }
    Value percentile(const Probability percent) const;
    void  getPdf(LookupTable<Probability, Value>& pdf) const;
    void  getCdf(LookupTable<Probability, Value>& cdf) const;
};

// Implementation Histogram
// ===========================================================================
template<typename Value>
Value Histogram<Value>::percentile(const Probability percent) const {
    Probability                          p = percent * this->sum();
    typename Precursor::ConstantIterator b;

    for (b = this->begin(); b != this->end() && p > 0; ++b)
        p -= *b;
    return index(b - this->begin());
}

template<typename Value>
void Histogram<Value>::getPdf(LookupTable<Probability, Value>& pdf) const {
    require(this->sum() != 0);
    pdf = *this;
    pdf.normalizeSurface();
}

template<typename Value>
void Histogram<Value>::getCdf(LookupTable<Probability, Value>& cdf) const {
    require(this->sum() != 0);
    cdf = *this;
    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                   std::bind2nd(std::divides<Probability>(), this->sum()));
}

/**
 *  Histogram Vector
 */
template<typename T>
class HistogramVector : public std::vector<Histogram<T>> {
public:
    typedef std::vector<Histogram<T>>      Precursor;
    typedef typename Precursor::value_type HistogramType;
    typedef typename HistogramType::Weight Weight;

public:
    explicit HistogramVector(size_t size = 0, T bucketSize = 0)
            : Precursor(size, HistogramType(bucketSize)) {}

    void accumulate(const std::vector<T>& v, Weight weight = 1);

    T minimalBucketSize() const;

    void read(Core::BinaryInputStream& is);
    void write(Core::BinaryOutputStream& os) const;
    void dump(Core::XmlWriter& os) const;
};

// Implementation HistogramVector
// ===========================================================================
template<typename T>
void HistogramVector<T>::accumulate(const std::vector<T>& v, Weight weight) {
    verify_(v.size() == Precursor::size());
    for (u32 i = 0; i < v.size(); i++)
        this->operator[](i).accumulate(v[i], weight);
}

template<typename T>
T HistogramVector<T>::minimalBucketSize() const {
    T result = Core::Type<T>::max;
    for (typename Precursor::const_iterator i = this->begin(); i != this->end(); ++i)
        result = std::min(result, i->bucketSize());
    return result;
}

template<typename T>
void HistogramVector<T>::read(Core::BinaryInputStream& is) {
    u32 s;
    is >> s;
    this->resize(s);
    for (u32 i = 0; i < this->size(); ++i)
        is >> this->operator[](i);
}

template<typename T>
void HistogramVector<T>::write(Core::BinaryOutputStream& os) const {
    os << (u32)this->size();
    std::copy(this->begin(), this->end(), Core::BinaryOutputStream::Iterator<HistogramType>(os));
}

template<typename T>
void HistogramVector<T>::dump(Core::XmlWriter& o) const {
    o << Core::XmlOpen("histogram-vector") + Core::XmlAttribute("size", this->size());
    for (u32 i = 0; i < this->size(); ++i)
        o << "\n"
          << this->operator[](i);
    o << "\n"
      << Core::XmlClose("histogram-vector");
}

}  // namespace Signal

#endif  // _SIGNAL_HISTOGRAM_HH
