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
#include "AcousticModelTrainer.hh"
#include <Am/DecisionTreeStateTying.hh>
#include <Am/Module.hh>
#include <Core/Hash.hh>
#include <Core/ProgressIndicator.hh>
#include <Core/TextStream.hh>

using namespace Speech;

// ===========================================================================
AcousticModelTrainer::AcousticModelTrainer(const Core::Configuration& c, Am::AcousticModel::Mode mode)
        : Component(c),
          Precursor(c) {
    lexicon_ = Bliss::Lexicon::create(select("lexicon"));
    if (!lexicon_)
        criticalError("Failed to initialize lexicon.");

    acousticModel_ = Am::Module::instance().createAcousticModel(select("acoustic-model"), lexicon_, mode);
    if (!acousticModel_)
        criticalError("Failed to initialize acoustic model.");
}

AcousticModelTrainer::~AcousticModelTrainer() {}

void AcousticModelTrainer::signOn(CorpusVisitor& corpusVisitor) {
    Precursor::signOn(corpusVisitor);
    acousticModel_->signOn(corpusVisitor);
}
// ===========================================================================
TextDependentMixtureSetTrainer::TextDependentMixtureSetTrainer(const Core::Configuration& c)
        : Core::Component(c),
          AcousticModelTrainer(c, Am::AcousticModel::noEmissions),
          MlMixtureSetTrainer(c),
          featureDescription_(*this),
          initialized_(false) {}

void TextDependentMixtureSetTrainer::setFeatureDescription(const Mm::FeatureDescription& description) {
    if (!initialized_) {
        featureDescription_ = description;

        size_t dimension;
        featureDescription_.mainStream().getValue(Mm::FeatureDescription::nameDimension, dimension);

        initializeAccumulation(acousticModel()->nEmissions(), dimension);
        initialized_ = true;
    }
    else {
        if (featureDescription_ != description) {
            criticalError("Change of features is not allowed.");
        }
    }
    AcousticModelTrainer::setFeatureDescription(description);
}

// ===========================================================================

const Core::ParameterFloat TiedTextDependentMixtureSetTrainer::paramTyingFactor(
        "tying-factor",
        "weight factor",
        0.7, 0.0, 1.0);

const Core::ParameterFloat TiedTextDependentMixtureSetTrainer::paramTyingMinFactor(
        "tying-min-factor",
        "minimum cut-off factor",
        0.001, 0.0, 1.0);

const Core::ParameterInt TiedTextDependentMixtureSetTrainer::paramTyingMinDepth(
        "tying-min-depth",
        "minimum depth of tied models in the cart tree",
        2, 0);

const Core::ParameterInt TiedTextDependentMixtureSetTrainer::paramTyingMaxEmissions(
        "tying-max-emissions",
        "maximum number of emissions samples are distributed over",
        Core::Type<s32>::max, 0);

TiedTextDependentMixtureSetTrainer::TiedTextDependentMixtureSetTrainer(const Core::Configuration& c)
        : Core::Component(c),
          AcousticModelTrainer(c, Am::AcousticModel::noEmissions),
          MlMixtureSetTrainer(c),
          featureDescription_(*this),
          initialized_(false),
          tyingFactor_(paramTyingFactor(c)),
          minTyingFactor_(paramTyingMinFactor(c)),
          minDepth_(paramTyingMinDepth(c)),
          maxEmissions_(paramTyingMaxEmissions(c)) {}

void TiedTextDependentMixtureSetTrainer::setFeatureDescription(const Mm::FeatureDescription& description) {
    if (!initialized_) {
        featureDescription_ = description;

        size_t dimension;
        featureDescription_.mainStream().getValue(Mm::FeatureDescription::nameDimension, dimension);

        initializeAccumulation(acousticModel()->nEmissions(), dimension);
        initialized_ = true;

        Am::ClassicAcousticModel* classicAm = dynamic_cast<Am::ClassicAcousticModel*>(acousticModel().get());
        verify(classicAm);
        Am::ClassicStateTyingRef          classicTying = classicAm->stateTying();
        const Am::DecisionTreeStateTying* treeTying    = dynamic_cast<const Am::DecisionTreeStateTying*>(classicTying.get());
        verify(treeTying);
        std::vector<const Cart::BinaryTree::Node*> nodes(acousticModel()->nEmissions(), 0);
        {
            std::vector<const Cart::BinaryTree::Node*> stack;
            stack.push_back(&treeTying->decisionTree().root());
            while (!stack.empty()) {
                const Cart::BinaryTree::Node* node = stack.back();
                stack.pop_back();
                if (!node->isLeaf()) {
                    stack.push_back(node->leftChild_);
                    stack.push_back(node->rightChild_);
                    verify(node->leftChild_ && node->rightChild_);
                }
                else {
                    verify(node->leftChild_ == 0 && node->rightChild_ == 0);
                    verify(node->id() < nodes.size());
                    verify(nodes[node->id()] == 0);
                    nodes[node->id()] = node;
                }
            }
        }
        for (u32 i = 0; i < nodes.size(); ++i) {
            if (!nodes[i])
                std::cout << "missing at " << i << std::endl;
        }

        struct Collector {
            Collector(const std::vector<const Cart::BinaryTree::Node*>& _nodes, float _factor, float _minFactor, u32 _minDepth)
                    : nodes(_nodes),
                      factor(_factor),
                      minFactor(_minFactor),
                      minDepth(_minDepth),
                      totalEmissions(0) {
                collectEmissions();
            }

            static u32 getDepth(const Cart::BinaryTree::Node* currentNode) {
                u32 depth = 0;
                while (currentNode->father_) {
                    ++depth;
                    currentNode = currentNode->father_;
                }
                return depth;
            }

            void collectEmissions() {
                for (int e = 0; e < nodes.size(); ++e) {
                    verify(nodes[e]);
                    verify(!nodes[e]->leftChild_ && !nodes[e]->rightChild_);
                    emissions_.push_back(std::vector<std::pair<u32, float>>());
                    std::set<const Cart::BinaryTree::Node*> visiting;
                    visiting.insert(nodes[e]);
                    visit(nodes[e]->father_, getDepth(nodes[e]), visiting, emissions_.back());
                }
            }

            void visit(const Cart::BinaryTree::Node* node, u32 lowestDepth, std::set<const Cart::BinaryTree::Node*>& visiting, std::vector<std::pair<u32, float>>& emissions) {
                if (!node || visiting.count(node))
                    return;

                u32 depth = getDepth(node);

                if (depth < minDepth)
                    return;

                if (depth < lowestDepth)
                    lowestDepth = depth;

                visiting.insert(node);

                if (node->isLeaf()) {
                    verify(depth > lowestDepth);
                    float fac = pow(factor, depth - lowestDepth);
                    if (fac >= minFactor) {
                        verify(node->id_ < nodes.size() && nodes[node->id_] == node);
                        emissions.push_back(std::make_pair(node->id_, fac));
                        ++totalEmissions;
                    }
                }

                visit(node->father_, lowestDepth, visiting, emissions);
                visit(node->leftChild_, lowestDepth, visiting, emissions);
                visit(node->rightChild_, lowestDepth, visiting, emissions);

                visiting.erase(node);
            }

            const std::vector<const Cart::BinaryTree::Node*>& nodes;
            float                                             factor, minFactor;
            u32                                               minDepth;
            std::vector<std::vector<std::pair<u32, float>>>   emissions_;
            u32                                               totalEmissions;
        } collector(nodes, tyingFactor_, minTyingFactor_, minDepth_);

        /// @todo Allow specifying the training-data multiplication factor directly (probably needs binary search)

        verify(collector.emissions_.size() == acousticModel()->nEmissions());
        tiedEmissions_.swap(collector.emissions_);

        u32 removedEmissions = 0;
        u32 totalEmissions   = 0;
        f64 totalWeight      = 0;

        for (u32 i = 0; i < tiedEmissions_.size(); ++i) {
            std::sort(tiedEmissions_[i].begin(), tiedEmissions_[i].end(),
                      Core::composeBinaryFunction(std::less<f32>(),
                                                  Core::select2nd<std::pair<u32, f32>>(),
                                                  Core::select2nd<std::pair<u32, f32>>()));

            if (tiedEmissions_[i].size() > maxEmissions_) {
                removedEmissions += maxEmissions_ - tiedEmissions_[i].size();
                tiedEmissions_[i].resize(maxEmissions_);
            }
            totalEmissions += tiedEmissions_[i].size();

            for (u32 e = 0; e < tiedEmissions_[i].size(); ++e)
                totalWeight += tiedEmissions_[i][e].second;
        }

        log() << "average number of tied emissions for each emission: " << float(totalEmissions / tiedEmissions_.size());
        log() << "average number of removed tied emissions for each emission: " << float(removedEmissions / tiedEmissions_.size());
        log() << "average tied emission weight: " << totalWeight / totalEmissions;
        log() << "training data multiplication factor: " << ((totalWeight / totalEmissions) * tiedEmissions_.size()) + 1.0;
    }
    else {
        if (featureDescription_ != description) {
            criticalError("Change of features is not allowed.");
        }
    }
    AcousticModelTrainer::setFeatureDescription(description);
}

void TiedTextDependentMixtureSetTrainer::processAlignedFeature(Core::Ref<const Feature> f, Am::AllophoneStateIndex e) {
    processAlignedFeature(f, e, 1.0);
}

void TiedTextDependentMixtureSetTrainer::processAlignedFeature(Core::Ref<const Feature> f, Am::AllophoneStateIndex e, Mm::Weight w) {
    Am::AcousticModel::EmissionIndex emission = acousticModel()->emissionIndex(e);
    accumulate(f->mainStream(), emission, w);
    verify(emission < tiedEmissions_.size());
    for (u32 i = 0; i < tiedEmissions_[emission].size(); ++i)
        accumulate(f->mainStream(), tiedEmissions_[emission][i].first, tiedEmissions_[emission][i].second * w);
}

