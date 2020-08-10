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
#include "StatePosteriorFeatureScorerNode.hh"
#include <Core/XmlStream.hh>
#include <Flow/Vector.hh>
#include <Mm/Module.hh>
#include <Speech/Feature.hh>

using namespace Speech;

StatePosteriorFeatureScorerNode::StatePosteriorFeatureScorerNode(const Core::Configuration& config)
        : Component(config),
          Precursor(config) {
    fs_ = new Mm::StatePosteriorFeatureScorer(select("posterior-feature-scorer"),
                                              Mm::Module::instance().readAbstractMixtureSet(select("mixture-set")));
}

bool StatePosteriorFeatureScorerNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    getInputAttributes(0, *attributes);
    if (!configureDatatype(attributes, Feature::FlowFeature::type()))
        return false;
    attributes->set("datatype", Sparse::Vector<f32>::type()->name());
    return putOutputAttributes(0, attributes);
}

bool StatePosteriorFeatureScorerNode::work(Flow::PortId) {
    Flow::DataPtr<Speech::Feature::FlowFeature> in;
    if (!getData(0, in)) {
        return putData(0, in.get());
    }
    fs_->setDefaultFilter();
    Core::Ref<const Speech::Feature>      featureRef = Core::ref(new Speech::Feature(in));
    Core::Ref<const StatePosteriorScorer> scorer(required_cast(const StatePosteriorScorer*,
                                                               fs_->getAssigningScorer(featureRef).get()));
    const PosteriorsAndDensities&         p = scorer->posteriorsAndDensities();
    std::vector<Mm::DensityIndex>         sortedP;
    for (PosteriorsAndDensities::const_iterator it = p.begin(); it != p.end(); ++it) {
        sortedP.push_back(it->first);
    }
    std::sort(sortedP.begin(), sortedP.end());
    ScoreVector sv(fs_->nDensities());
    for (std::vector<Mm::DensityIndex>::const_iterator it = sortedP.begin(); it != sortedP.end(); ++it) {
        sv.push_back(*it, p.find(*it)->second);
    }
    Core::TsRef<FlowScoreVector> out = Core::tsRef(new FlowScoreVector(sv));
    out->setTimestamp(*in);
    return putData(0, out.get());
}
