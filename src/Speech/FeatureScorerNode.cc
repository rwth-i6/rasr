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
#include "FeatureScorerNode.hh"
#include <Mm/Module.hh>
#include <Flow/Vector.hh>
#include <Flow/TypedAggregate.hh>
#include <Flow/Registry.hh>


namespace Speech {

FeatureScorerNode::FeatureScorerNode(const Core::Configuration& config)
:
    Component(config),
    Precursor(config),
    needInit_(true),
    aggregatedFeatures_(false)
{
    fs_ = Mm::Module::instance().createFeatureScorer(
        select("feature-scorer"),
        Mm::Module::instance().readAbstractMixtureSet(select("mixture-set")));
    require(fs_);
}

FeatureScorerNode::~FeatureScorerNode() {
}

// overrides Flow::Node::configure
// input of the node is a vector or an aggregate vector
// output of the node is a single vector stream
bool FeatureScorerNode::configure() {
    fs_->reset();
    timeStamps_.clear();

    // get the attributes
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    getInputAttributes(0, *attributes);

    // check the allowed data types (Vector + aggregate Vector)
    if (! (configureDataType(attributes, Flow::Vector<FeatureType>::type()) ||
            configureDataType(attributes, Flow::TypedAggregate<Flow::Vector<FeatureType> >::type())) ) {
        return false;
    }

    // return attributes (single vector stream)
    attributes->set("datatype", Flow::Vector<FeatureType>::type()->name());
    return putOutputAttributes(0, attributes);
}


/* same as Flow::Node::configureDatatype but without the error message
 * used for checking whether aggregated features or single feature stream is received
 * */
bool FeatureScorerNode::configureDataType(Core::Ref<const Flow::Attributes> a, const Flow::Datatype *d) {
    // check for valid attribute reference
    if (! a) {
        return false;
    }

    std::string dtn(a->get("datatype"));
    const Flow::Datatype* datatype(0);
    if (! dtn.empty()) { // get the data type from the attributes
        datatype = Flow::Registry::instance().getDatatype(dtn);
    }

    // data type from attribute and given data type do not match
    if (datatype != d) {
        return false;
    }

    // default is true
    return true;
}


static const Mm::FeatureVector& convertFeature(const Flow::Vector<FeatureScorerNode::FeatureType>& feature) {
    return (const Mm::FeatureVector&) feature;  // Mm::FeatureVector is just std::vector. we can just cast
}

static Core::Ref<Mm::Feature> convertFeature(const Flow::TypedAggregate<Flow::Vector<FeatureScorerNode::FeatureType> >& feature) {
    Core::Ref<Mm::Feature> fs_feature(new Mm::Feature(feature.size())); // multi-stream feature
    for (u32 i = 0; i < feature.size(); ++i) {
        // Flow::DataPtr is incompatible with Core::Ref. Thus we need to make a copy.
        fs_feature->set(i, Mm::Feature::convert((const Mm::FeatureVector&) *feature[i]));
    }
    return fs_feature;
}


bool FeatureScorerNode::putData(Mm::FeatureScorer::Scorer scorer) {
    Flow::Vector<FeatureType> *out = NULL;
    out = new Flow::Vector<FeatureType>(scorer->nEmissions());
    for(u32 i = 0; i < scorer->nEmissions(); ++i) {
        // A FeatureScorer returns the scores in -log space.
        // This Flow node is expected to return the scores in +log space.
        out->at(i) = -scorer->score(i);
    }

    // It's always related to the *front* of timeStamps_, in case there was buffering, which refers to the oldest frame.
    require(!timeStamps_.empty());
    out->setTimestamp(timeStamps_.front());
    timeStamps_.pop_front();

    // putData() will overtake out.
    return Precursor::putData(0, out);
}


template<class T>  // T is Flow::Vector<FeatureType> or Flow::TypedAggregate<Flow::Vector<FeatureType> >
bool FeatureScorerNode::work() {
    // The FeatureScorer interface, without buffering (!isBuffered()):
    //   For every input, get a scorer via getScorer().
    // With buffering (isBuffered()):
    //   Fill buffer until full (bufferFilled()). When we get more input, use getScorer().
    //   (The TrainerFeatureScorer has an infinite buffer, i.e. it's never full.)
    // Then, while not bufferEmpty(), call flush() to get a scorer for each remaining frame.
    // Thus, in every case, we can read as much input as we can here.

    while(true) {
        // pull feature from incoming connections
        Flow::DataPtr<T> ptrFeatures; // features from the flow network (single feature stream or aggregated features)
        if(!getData(0, ptrFeatures)) {
            require(ptrFeatures == Flow::Data::eos()); // or what else can it be?
            break;
        }
        timeStamps_.push_back((Flow::Timestamp) *ptrFeatures.get());

        if(fs_->isBuffered() && !fs_->bufferFilled()) {
            // fill buffer
            fs_->addFeature(convertFeature(*ptrFeatures.get()));
        }
        else {
            Mm::FeatureScorer::Scorer scorer = fs_->getScorer(convertFeature(*ptrFeatures.get()));
            if(!putData(scorer))
                return false;
        }
    }
    // We get out of the loop only when there is no more input data.

    // In case of buffering, wait until we get all the input data,
    // and only then get out the scores. That is how the FeatureScores expects it.
    // In case of bidir-RNNs (e.g. via TrainerFeatureScorer via PythonTrainer),
    // it's important that way, because the NN forward will happen at the first flush() call.
    if(fs_->isBuffered()) {
        while(!fs_->bufferEmpty()) {
            Mm::FeatureScorer::Scorer scorer = fs_->flush();
            if(!putData(scorer))
                return false;
        }
    }

    require(timeStamps_.empty());
    // There might be different behavior for different FeatureScorer's.
    fs_->finalize(); // finalize this segment
    fs_->reset(); // for the next round

    return Precursor::putData(0, Flow::Data::eos());
}


// overrides Flow::Node::work
bool FeatureScorerNode::work(Flow::PortId /*output, expected to be zero*/) {
    if(needInit_) {
        // get data type of the flow stream
        Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
        getInputAttributes(0, *attributes);
        aggregatedFeatures_ = configureDataType(attributes, Flow::TypedAggregate<Flow::Vector<FeatureType> >::type());
        needInit_ = false;
    }

    if(aggregatedFeatures_)
        return work<Flow::TypedAggregate<Flow::Vector<FeatureType> > >();
    else
        return work<Flow::Vector<FeatureType> >();
}


}
