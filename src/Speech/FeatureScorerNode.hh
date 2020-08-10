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
#ifndef _SPEECH_FEATURESCORERNODE_HH
#define _SPEECH_FEATURESCORERNODE_HH

#include <Flow/Node.hh>
#include <Flow/Timestamp.hh>
#include <Mm/FeatureScorer.hh>
#include <deque>

namespace Speech {

class FeatureScorerNode : public Flow::SleeveNode {
    typedef Flow::SleeveNode Precursor;

public:
    typedef Mm::FeatureType FeatureType;

private:
    Core::Ref<Mm::FeatureScorer> fs_;
    bool                         needInit_;
    bool                         aggregatedFeatures_;  // features are aggregated (multiple input streams)
    std::deque<Flow::Timestamp>  timeStamps_;          // flow timestamps corresponding to feature vectors
public:
    FeatureScorerNode(const Core::Configuration& config);
    virtual ~FeatureScorerNode();
    static std::string filterName() {
        return "feature-scorer";
    }

    virtual bool configure();
    virtual bool work(Flow::PortId p);

protected:
    // checks whether we have aggregated or simple Flow features
    bool configureDataType(Core::Ref<const Flow::Attributes> a, const Flow::Datatype* d);
    bool putData(Mm::FeatureScorer::Scorer scorer);
    template<class T>
    bool work();
};

}  // namespace Speech

#endif  // _SPEECH_FEATURESCORERNODE_HH
