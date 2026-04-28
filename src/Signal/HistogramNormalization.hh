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
#ifndef _SIGNAL_HISTOGRAM_NORMALIZATION_HH
#define _SIGNAL_HISTOGRAM_NORMALIZATION_HH

#include <Core/ObjectCache.hh>
#include <Flow/DataAdaptor.hh>
#include "Histogram.hh"
#include "Node.hh"

namespace Signal {

/** HistogramNormalization */
class HistogramNormalization {
public:
    typedef f32                           Value;
    typedef Histogram<Value>::Probability Probability;
    typedef Histogram<Value>::Weight      HistogramWeight;

    typedef LookupTable<Probability, Value> Cdf;
    typedef std::vector<Cdf>                Cdfs;
    typedef LookupTable<Value, Probability> InverseCdf;
    typedef std::vector<InverseCdf>         InverseCdfs;

private:
    Cdfs        testCdfs_;
    InverseCdfs inverseTrainingCdfs_;

public:
    HistogramNormalization() {}

    void apply(const std::vector<Value>& in, std::vector<Value>& out);

    /** @param probabilityBucketSize is bucket size of the inverse training cumulative density functions
     */
    void   setTrainingHistograms(const std::vector<Histogram<Value>>& trainingHistograms,
                                 Probability                          probabilityBucketSize);
    void   setTrainingHistograms(const std::vector<HistogramVector<Value>>& trainingHistograms,
                                 const std::vector<Value>&                  scales,
                                 Probability                                probabilityBucketSize);
    size_t nTrainingHistograms() const {
        return inverseTrainingCdfs_.size();
    }

    void   setTestHistograms(const std::vector<Histogram<Value>>& testHistograms);
    size_t nTestHistograms() const {
        return testCdfs_.size();
    }

    /** @return is false if one of the weights is smaller than larger than one. */
    static bool areScalesWellDefined(const std::vector<HistogramWeight>& scales);
    /** @return is true if the sum of weights is one. */
    static bool areScalesNormalized(const std::vector<HistogramWeight>& scales);
    /** Inserts a scale at the beginning, which makes the sum of scales to be zero. */
    static void normalizeScales(std::vector<HistogramWeight>& scales);
};

/** HistogramNormalizationNode */
class HistogramNormalizationNode : public Flow::Node,
                                   public HistogramNormalization {
private:
    static const Core::ParameterFloat        paramProbabilityBucketSize;
    static const Core::ParameterStringVector paramTrainingHistogramsFilenames;
    static const Core::ParameterString       paramCorpusKey;

private:
    Flow::PortId             firstScalePortId_;
    static const std::string scalePortname_;

    typedef HistogramNormalization::Probability Probability;
    Probability                                 probabilityBucketSize_;

    std::vector<std::string>                        trainingHistogramFilenames_;
    std::vector<HistogramVector<Value>>             trainingHistograms_;
    std::vector<Flow::DataAdaptor<HistogramWeight>> histogramScales_;

    typedef HistogramNormalization::Value Value;
    typedef Core::ObjectCache<Core::MruObjectCacheList<
            std::string, HistogramVector<Value>, Core::StringHash, Core::StringEquality>>
                   HistogramCache;
    HistogramCache testHistograms_;

    bool needInit_;

private:
    void init(size_t featureDimension);
    void reset();

    void setProbabilityBucketSize(Probability r) {
        if (probabilityBucketSize_ != r) {
            probabilityBucketSize_ = r;
            needInit_              = true;
        }
    }
    void setTrainingHistogramFilenames(const std::vector<std::string>& n) {
        if (trainingHistogramFilenames_ != n) {
            trainingHistogramFilenames_ = n;
            needInit_                   = true;
        }
    }
    void loadTrainingHistograms(size_t featureDimension);
    /**
     *   Configures the normalization algorithm with new training histograms scales.
     */
    bool updateTrainingHistograms();
    void setTestHistograms(const std::string& corpusKey);

    /**
     *  Histogram scale ports are read if necessary.
     *  Only those ports are read where the current histogram scale (@see histogramScales_) does
     *   not contain the timestamp. In this way the scales are synchornized to the main input stream.
     */
    bool updateScales(const Flow::Timestamp& timestamp);

public:
    static std::string filterName() {
        return "signal-histogram-normalization";
    }
    HistogramNormalizationNode(const Core::Configuration& c);
    virtual ~HistogramNormalizationNode() {}

    virtual bool         setParameter(const std::string& name, const std::string& value);
    virtual bool         configure();
    virtual Flow::PortId getInput(const std::string& name);
    virtual Flow::PortId getOutput(const std::string& name) {
        return 0;
    }
    virtual bool work(Flow::PortId p);
};
}  // namespace Signal

#endif  // _SIGNAL_HISTOGRAM_NORMALIZATION_HH
