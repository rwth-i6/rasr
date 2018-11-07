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
#ifndef _SPEECH_STATE_POSTERIOR_FEATURE_SCORER_NODE_HH_
#define _SPEECH_STATE_POSTERIOR_FEATURE_SCORER_NODE_HH_

#include <Core/Types.hh>
#include <Core/BinaryStream.hh>
#include <Core/Component.hh>
#include <Core/Configuration.hh>
#include <Flow/Node.hh>
#include <Mm/StatePosteriorFeatureScorer.hh>
#include <Sparse/Vector.hh>

namespace Speech {

    class StatePosteriorFeatureScorerNode : public Flow::SleeveNode {
        typedef Flow::SleeveNode Precursor;
        typedef Mm::StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer StatePosteriorScorer;
        typedef Mm::StatePosteriorFeatureScorer::PosteriorsAndDensities PosteriorsAndDensities;
    public:
        typedef Sparse::Vector<f32>  FlowScoreVector;
        typedef Sparse::SingleValueSparseVector<f32>  ScoreVector;
    private:
        Mm::StatePosteriorFeatureScorer* fs_;
    public:
        StatePosteriorFeatureScorerNode(const Core::Configuration& config);
        virtual ~StatePosteriorFeatureScorerNode() {delete fs_; }
        static std::string filterName() {return "state-posterior-feature-scorer"; }

        virtual bool configure();
        virtual bool work(Flow::PortId p);
    };

} // namespace Speech

#endif //  _SPEECH_STATE_POSTERIOR_FEATURE_SCORER_NODE_HH_
