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
#ifndef HELPERS_HH
#define HELPERS_HH

#include <Core/ReferenceCounting.hh>
#include <Core/StopWatch.hh>
#include <Core/Types.hh>
#include <string>
#include "SearchSpaceStatistics.hh"

namespace Search {
class SearchSpaceStatistics;
}

namespace Bliss {
class LemmaPronunciation;
};

namespace Core {
class Configuration;
};

bool isBackwardRecognition(const Core::Configuration& config);

class PerformanceCounter {
public:
    PerformanceCounter(Search::SearchSpaceStatistics& stats, const std::string& name, bool start = true)
            : stopWatch_(), timeStats_(stats.customStatistics("Profiling: " + name + ": Centiseconds")) {
        if (start) {
            stopWatch_.start();
        }
    }

    ~PerformanceCounter() {
        stopAndYield();
    }

    void start() {
        stopWatch_.stop();
        stopWatch_.start();
    }

    void stop() {
        stopWatch_.stop();
    }

    /// Prints the current instruction count to the statistics object
    void stopAndYield(bool print = false) {
        stop();
        timeStats_ += stopWatch_.elapsedCentiseconds();
        if (print) {
            std::cout << " time: " << stopWatch_.elapsedCentiseconds() << std::endl;
        }
        stopWatch_.reset();
    }

private:
    Core::StopWatch        stopWatch_;
    Core::Statistics<f32>& timeStats_;
};

inline f32 scaledLogAdd(f32 a, f32 b, f32 scale, f32 invertedScale) {
    if (b == Core::Type<f32>::max)
        return a;
    if (a == Core::Type<f32>::max)
        return b;
    a *= invertedScale;
    b *= invertedScale;
    return scale * (std::min(a, b) - ::log1p(::exp(std::min(a, b) - std::max(a, b))));
}

inline bool approximatelyEqual(double a, double b, const double threshold = 0.001) {
    double diff = a - b;
    return diff > -threshold && diff < threshold;
}

// Helper function
template<class T>
inline void overwriteOrPush(u32& size, std::vector<T>& vec, const T& value) {
    if (size == vec.size()) {
        vec.push_back(value);
    }
    else {
        vec[size] = value;
    }
    ++size;
}

/// A simple helper function that parses a standard "[a, b, c, ...]" array into a vector containing the values a,b,c,...
inline std::vector<float> parsePythonArray(const std::string& dp) {
    std::vector<float> ret;
    verify(dp[0] == '[');
    int pos = 1;
    while (pos < dp.length()) {
        size_t next = dp.find(",", pos);
        if (next == std::string::npos)
            next = dp.find("]", pos);
        verify(next != std::string::npos);

        if (pos == next)
            break;

        double val = strtod(dp.c_str() + pos, 0);
        ret.push_back(val);

        pos = next + 1;
    }
    return ret;
}

template<class T>
inline std::string dumpPythonArray(std::vector<T> array) {
    std::ostringstream txt;
    txt << "{";
    bool first = true;
    for (u32 a = 0; a < array.size(); ++a) {
        if (first)
            first = false;
        else
            txt << ", ";

        txt << a << " : " << array[a];
    }
    txt << "}";

    return txt.str();
}

template<class T>
struct SetHash {
    size_t operator()(const std::set<T>& set) const {
        size_t a = set.size();
        a        = (a ^ 0xc761c23c) ^ (a >> 19);
        a        = (a + 0xfd7046c5) + (a << 3);
        for (typename std::set<T>::const_iterator it = set.begin(); it != set.end(); ++it)
            a += (*it << a) + a * *it + (*it ^ 0xb711a53c);
        return a;
    }
};

bool pronunciationHasEvaluationTokens(const Bliss::LemmaPronunciation* pron);

class GaussianDensity {
public:
    GaussianDensity(f32 bias = 1.0)
            : mean_(0),
              variance_(0),
              sigma_(0),
              offset_(0),
              energySum_(0),
              energySquareSum_(0),
              energyWeight(0),
              bias_(bias) {
    }

    // Returns the score regarding the gaussian distribution
    f64 score(f64 value) const {
        f64 d = (value - mean_) / sigma_;
        d     = (d * d) * 0.5;
        return (d + offset_) / bias_;
    }

    void add(f64 energy, f32 weight = 1) {
        energySum_ += energy * weight;
        energyWeight += weight;
        energySquareSum_ += (energy * energy) * weight;
    }

    void estimate() {
        if (energyWeight) {
            mean_     = energySum_ / energyWeight;
            variance_ = (energySquareSum_ - 2 * mean_ * energySum_ + energyWeight * mean_ * mean_) / energyWeight;
            sigma_    = sqrt(variance_);
            offset_   = sigma_ * sqrt(2 * M_PI);
        }
    }

    void reset() {
        energySum_       = 0;
        energySquareSum_ = 0;
        energyWeight     = 0;
    }

    f64 average() const {
        return sum() / count();
    }

    f64 sum() const {
        return energySum_;
    }

    f64 mean() const {
        return mean_;
    }

    f64 sigma() const {
        return sigma_;
    }

    u64 count() const {
        return energyWeight;
    }

private:
    f64 mean_;
    f64 variance_;
    f64 sigma_;
    f64 offset_;

    f64 energySum_;
    f64 energySquareSum_;
    f64 energyWeight;

    f32 bias_;
};

template<class T>
class AsymmetricIntersectionIterator {
public:
    AsymmetricIntersectionIterator(const std::vector<T>& array1, const std::vector<T>& array2)
            : a_(array1.size() < array2.size() ? array1 : array2), b_(array1.size() < array2.size() ? array2 : array1), ready_(false) {
        stack_.reserve(50);
        assert(a_.size() <= b_.size());

        if (a_.empty() || b_.empty()) {
            ready_ = true;
            return;
        }
        currentA_.start = 0;
        currentA_.end   = a_.size() - 1;

        currentB_.start = 0;
        currentB_.end   = b_.size() - 1;

        match();
    }

    operator bool() const {
        return !ready_;
    }

    const T& operator*() const {
        return a_[currentA_.start];
    }

    struct Range {
        int start, end;  // Inclusive!
    };

    AsymmetricIntersectionIterator& operator++() {
        ++currentA_.start;
        ++currentB_.start;
        if (currentA_.start > currentA_.end || currentB_.start > currentB_.end)
            pop();

        match();

        return *this;
    }

private:
    inline void pop() {
        if (stack_.empty()) {
            ready_ = true;
            return;
        }
        currentA_ = stack_.back().first;
        currentB_ = stack_.back().second;
        stack_.pop_back();
    }

    void match() {
    rematch:
        if (currentA_.start > currentA_.end || currentB_.start > currentB_.end)
            pop();

        if (ready_ || a_[currentA_.start] == b_[currentB_.start])
            return;  // Match

        bool matched;
        int  splitStartA, splitStartB;

        /// @todo Think out some optimized inverted logic when lengthA > lengthB

        // Split and match from a
        splitStartA = (currentA_.end - currentA_.start) / 2 + currentA_.start;

        typename std::vector<T>::const_iterator it = std::lower_bound(b_.begin() + currentB_.start, b_.begin() + currentB_.end + 1, a_[splitStartA]);
        if (it == b_.end()) {
            currentA_.end = splitStartA - 1;
            goto rematch;
        }

        matched = *it == a_[splitStartA];

        splitStartB = it - b_.begin();

        if (splitStartA == currentA_.start || splitStartB == currentB_.start) {
            // The left split is empty, throw it away
            if (matched)
                currentA_.start = splitStartA;  // Next round will be a match
            else
                currentA_.start = splitStartA + 1;
            currentB_.start = splitStartB;
        }
        else if (splitStartA >= currentA_.end || (matched && splitStartB >= currentB_.end)) {
            // The right split is empty, throw it away
            if (matched) {
                currentA_.end = splitStartA;
                currentB_.end = splitStartB;
            }
            else {
                // Discard last element, since we had a mismatch
                currentA_.end = splitStartA - 1;
                currentB_.end = splitStartB - 1;
            }
        }
        else {
            stack_.push_back(std::make_pair(currentA_, currentB_));
            if (matched)
                stack_.back().first.start = splitStartA;
            else
                stack_.back().first.start = splitStartA + 1;
            stack_.back().second.start = splitStartB;
            currentA_.end              = splitStartA - 1;
            currentB_.end              = splitStartB - 1;
        }

        goto rematch;
    }

    const std::vector<T>&                a_;
    const std::vector<T>&                b_;
    Range                                currentA_, currentB_;
    std::vector<std::pair<Range, Range>> stack_;
    bool                                 ready_;
};

#endif
