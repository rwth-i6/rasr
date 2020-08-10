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
#include "AcousticLookAhead.hh"
#include <Mm/Module.hh>
#include "SearchNetworkTransformation.hh"
#include "SearchSpace.hh"

using namespace AdvancedTreeSearch;
using namespace Search;

static const StateId debugState = 1;

static const int invalidLookAheadModel = Core::Type<u32>::max;

const Core::ParameterInt paramAcousticLookaheadDepth(
        "acoustic-lookahead-depth",
        "state depth of the acoustic look-ahead. If this is zero, acoustic look-ahead stays disabled. Good value: 1",
        0);

const Core::ParameterInt paramAcousticLookAheadModelCount(
        "acoustic-lookahead-model-count",
        "desired number of acoustic look-ahead models",
        1500);

const Core::ParameterInt paramAcousticLookAheadIterations(
        "acoustic-lookahead-iterations",
        "number of iterations of acoustic look-ahead node generation",
        3, 1);

const Core::ParameterString paramAcousticLookAheadCacheArchive(
        "acoustic-lookahead-cache-archive",
        "",
        "global-cache");

const Core::ParameterString paramAcousticLookaheadMixtureSet(
        "acoustic-lookahead-mixture-set",
        "",
        "");

const Core::ParameterBool paramAcousticLookAheadConsiderLabels(
        "acoustic-lookahead-consider-labels",
        "",
        true);

const Core::ParameterBool paramAcousticLookAheadUseAverage(
        "acoustic-lookahead-use-average",
        "",
        true);

const Core::ParameterInt paramAcousticLookAheadSplitsPerState(
        "acoustic-lookahead-splits-per-state",
        "",
        0);

const Core::ParameterInt paramAcousticLookAheadStartSplittingAtIteration(
        "acoustic-lookahead-start-splitting-at-iteration",
        "",
        0);

const Core::ParameterFloat paramAcousticLookAheadScale(
        "acoustic-lookahead-scale",
        "",
        2.5);

const Core::ParameterFloat paramAcousticLookaheadPerDepthFactor(
        "acoustic-lookahead-per-depth-factor",
        "",
        1.0f);

const Core::ParameterBool paramAcousticLookAheadPerfect(
        "acoustic-lookahead-use-perfect-lookahead",
        "use the full acoustic model to do the acoustic look-ahead (very slow, only for testing purposes)",
        false);

const Core::ParameterBool paramAcousticLookAheadIncludeCurrentStateModel(
        "acoustic-lookahead-include-current-state-model",
        "",
        false);

const Core::ParameterFloat paramSystematicSplittingThreshold(
        "acoustic-lookahead-splitting-threshold",
        "",
        0.0, 0.0, 0.3);

const Core::ParameterBool paramSplitEmpty(
        "acoustic-lookahead-split-empty",
        "",
        true);

const Core::ParameterBool paramApplyQuantizationScaling(
        "acoustic-lookahead-apply-quantization-scaling",
        "whether the effect of quantization in the mixture-set should be reverted the equal way it is done in the real scorers",
        true);

const Core::ParameterBool paramConsiderMultiplicity(
        "acoustic-lookahead-consider-multiplicity",
        "",
        true);

std::string AcousticLookAhead::archiveEntry() const {
    return isBackwardRecognition(config_) ? "backward-acoustic-look-ahead" : "acoustic-look-ahead";
}

int AcousticLookAhead::getDepth(const Core::Configuration& config) {
    return paramAcousticLookaheadDepth(select(config));
}

Score AcousticLookAhead::getScale(const Core::Configuration& config) {
    return paramAcousticLookAheadScale(select(config));
}

std::string AcousticLookAhead::getMixtureSetFilename(const Core::Configuration& config) {
    return paramAcousticLookaheadMixtureSet(select(config));
}

AcousticLookAhead::AcousticLookAhead(const Core::Configuration& _config, u32 checksum, bool load)
        : loaded_(false), checksum_(checksum), config_(select(_config)) {
    minCacheKey_ = nextCacheKey_ = 1;

    perDepthFactor_      = paramAcousticLookaheadPerDepthFactor(config_);
    lookaheadModelCount_ = paramAcousticLookAheadModelCount(config_);
    iterations_          = paramAcousticLookAheadIterations(config_);
    multiplicity_        = paramConsiderMultiplicity(config_);

    acousticLookAheadScorer_  = 0;
    includeCurrentStateModel_ = paramAcousticLookAheadIncludeCurrentStateModel(config_);

    acousticLookaheadDepth_ = paramAcousticLookaheadDepth(config_);
    acousticLookAheadScale_ = paramAcousticLookAheadScale(config_);

    splittingThreshold_ = paramSystematicSplittingThreshold(config_);
    splitEmpty_         = paramSplitEmpty(config_);

    Core::Application::us()->log() << "initializing acoustic look-ahead with depth " << acousticLookaheadDepth_ << " and scale " << acousticLookAheadScale_;

    if (isEnabled()) {
        mixtureSet_ = Core::Ref<Mm::MixtureSet>(dynamic_cast<Mm::MixtureSet*>(Mm::Module::instance().readAbstractMixtureSet(paramAcousticLookaheadMixtureSet(config_), config_).get()));
        verify(mixtureSet_.get());
        verify(mixtureSet_->nCovariances() == 1);

        acousticLookAheadScorer_ = new Mm::SimdGaussDiagonalMaximumFeatureScorer(Core::Configuration(config_, "acoustic-look-ahead-scorer"), mixtureSet_);

        if (paramApplyQuantizationScaling(config_)) {
            Mm::FeatureType factor = acousticLookAheadScorer_->inverseQuantizationFactor();
            acousticLookAheadScale_ *= factor;
            Core::Application::us()->log() << "Applying revert-quantization factor " << factor << ", changed scale from " << (acousticLookAheadScale_ / factor) << " to " << acousticLookAheadScale_;
        }
    }

    useAverage_ = paramAcousticLookAheadUseAverage(config_);

    considerLabels_ = paramAcousticLookAheadConsiderLabels(config_);

    currentTimeFrame_ = -1;

    for (int a = 0; a < length(); ++a)
        cachesForTimeframes_.push_back(new CacheForTimeframe(0));

    if (load)
        loaded_ = loadModels();
}

AcousticLookAhead::~AcousticLookAhead() {
    delete acousticLookAheadScorer_;
    for (int a = 0; a < length(); ++a)
        delete cachesForTimeframes_[a];
    cachesForTimeframes_.clear();
}

AcousticLookAhead::AcousticLookAheadModel::AcousticLookAheadModel(Core::MappedArchiveReader reader) {
    reader >> means_;
}

void AcousticLookAhead::AcousticLookAheadModel::write(Core::MappedArchiveWriter writer) {
    writer << means_;
}

Score AcousticLookAhead::calculateDistance(const AcousticLookAhead::AcousticFeatureVector& mean1, const AcousticLookAhead::AcousticFeatureVector& mean2) {
    int df, score = 0;
    u32 cmp       = 0;
    u32 dimension = mean1.size();
    verify(mean1.size() == mean2.size());
    switch ((dimension - 1) % 8) {
        while (cmp < dimension) {
            case 7:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 6:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 5:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 4:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 3:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 2:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 1:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 0:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
        }
    }
    verify_(cmp == dimension);
    return score;
}

Score AcousticLookAhead::calculateDistance(const Mm::SimdGaussDiagonalMaximumFeatureScorer::QuantizedType* mean1,
                                           const Mm::SimdGaussDiagonalMaximumFeatureScorer::QuantizedType* mean2,
                                           u32                                                             dimension) {
    int df, score = 0;
    u32 cmp = 0;
    switch ((dimension - 1) % 8) {
        while (cmp < dimension) {
            case 7:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 6:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 5:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 4:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 3:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 2:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 1:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
            case 0:
                df = (mean1[cmp] - mean2[cmp]);
                score += df * df;
                ++cmp;
        }
    }
    return score;
}

Score AcousticLookAhead::AcousticLookAheadModel::distance(const AcousticFeatureVector& mean) const {
    Score best = Core::Type<Score>::max;
    for (std::vector<AcousticFeatureVector>::const_iterator it = means_.begin(); it != means_.end(); ++it) {
        Score dist = calculateDistance(*it, mean);
        if (dist < best)
            best = dist;
    }
    return best;
}

void AcousticLookAhead::AcousticLookAheadModel::estimate(const std::vector<u32>&                   assigned,
                                                         const std::vector<AcousticFeatureVector>& means,
                                                         u32                                       splits) {
    if (assigned.empty())
        return;

    means_.clear();

    const u32 dimension = means[assigned[0]].size();

    std::vector<f64> accumulator;
    accumulator.resize(dimension, 0);
    for (u32 a = 0; a < assigned.size(); ++a)
        for (u32 d = 0; d < dimension; ++d)
            accumulator[d] += means[assigned[a]][d];

    AcousticFeatureVector mean;
    mean.resize(dimension, 0);

    for (u32 d = 0; d < dimension; ++d)
        mean[d] = accumulator[d] / assigned.size();

    means_.push_back(mean);

    std::vector<u32> assignedToMean(assigned.size(), 0);

    for (int s = 0; s < splits; ++s) {
        /// Step 1: Split all means
        u32 splitMeans = means_.size();
        for (u32 m = 0; m < splitMeans; ++m) {
            means_.push_back(means_[m]);
            for (u32 d = 0; d < dimension; ++d) {
                means_[m][d] -= 0.000001;
                means_.back()[d] += 0.000001;
            }
        }

        /// Step 2: Assign every observation to the closest mean
        for (u32 a = 0; a < assigned.size(); ++a) {
            Score bestDist = Core::Type<Score>::max;
            for (u32 m = 0; m < means_.size(); ++m) {
                Score dist = calculateDistance(means[assigned[a]], means_[m]);
                if (dist < bestDist) {
                    bestDist          = dist;
                    assignedToMean[a] = m;
                }
            }
        }

        /// Step 3: Accumulate means

        std::vector<std::vector<f64>> accumulators(means_.size(), std::vector<f64>(dimension, 0));
        std::vector<u32>              countAssignedToMean(means_.size(), 0);

        for (u32 a = 0; a < assigned.size(); ++a) {
            countAssignedToMean[assignedToMean[a]] += 1;
            for (u32 d = 0; d < dimension; ++d)
                accumulators[assignedToMean[a]][d] += means[assigned[a]][d];
        }

        for (int m = means_.size() - 1; m >= 0; --m) {
            if (countAssignedToMean[m] == 0) {
                // Remove means that are not required
                means_.erase(means_.begin() + m);
            }
            else {
                for (u32 d = 0; d < dimension; ++d)
                    means_[m][d] = accumulators[m][d] / countAssignedToMean[m];
            }
        }
    }
}

bool AcousticLookAhead::loadModels() {
    std::string archive = paramAcousticLookAheadCacheArchive(config_);

    if (archive.empty())
        return false;

    Core::MappedArchiveReader in = Core::Application::us()->getCacheArchiveReader(archive, archiveEntry());
    if (!in.good())
        return false;

    if (!in.check<u32>(checksum_, "network checksum") ||
        !in.check<u32>(acousticLookaheadDepth_, "depth") ||
        !in.check<u32>(lookaheadModelCount_, "model count") ||
        !in.check<u32>(iterations_, "iterations") ||
        !in.check<bool>(considerLabels_, "consider-labels") ||
        !in.check<bool>(includeCurrentStateModel_, "include-current") ||
        !in.check<bool>(multiplicity_, "consider-multiplicity") ||
        !in.check<f32>(splittingThreshold_, "splitting-threshold") ||
        !in.check<bool>(splitEmpty_, "split-empty"))
        return false;

    u64 lookAheadModels = 0ul;
    in >> lookAheadModels;
    for (u32 a = 0; a < lookAheadModels; ++a)
        acousticLookAheadModels_.push_back(AcousticLookAheadModel(in));
    in >> means_ >> modelForIndex_;

    verify(in.good());
    return true;
}

void AcousticLookAhead::saveModels() {
    std::string archive = paramAcousticLookAheadCacheArchive(config_);

    if (archive.empty())
        return;

    Core::MappedArchiveWriter out = Core::Application::us()->getCacheArchiveWriter(archive, archiveEntry());
    if (!out.good())
        return;

    out << (u32)checksum_ << (u32)acousticLookaheadDepth_ << (u32)lookaheadModelCount_;
    out << (u32)iterations_ << (bool)considerLabels_ << (bool)includeCurrentStateModel_;
    out << (bool)multiplicity_ << (f32)splittingThreshold_;
    out << (bool)splitEmpty_;

    out << (u64)acousticLookAheadModels_.size();
    for (u32 a = 0; a < acousticLookAheadModels_.size(); ++a)
        acousticLookAheadModels_[a].write(out);
    out << means_ << modelForIndex_;
}

void AcousticLookAhead::initializeModelsFromNetwork(const PersistentStateTree& network) {
    EmissionSetCounter sets;

    for (StateId state = 1; state < network.structure.stateCount(); ++state) {
        std::set<Am::AcousticModel::EmissionIndex> successorMixtures;

        if (includeCurrentStateModel_)
            getSuccessorMixtures(network, state, successorMixtures, acousticLookaheadDepth_ - 1, true);
        else
            getSuccessorMixtures(network, state, successorMixtures, acousticLookaheadDepth_);

        sets.get(successorMixtures, state);
    }

    initializeModels(sets);
}

void AcousticLookAhead::initializeModels(EmissionSetCounter sets) {
    if (acousticLookaheadDepth_ == 0)
        return;
    int splitsPerState            = paramAcousticLookAheadSplitsPerState(config_);
    int startSplittingAtIteration = paramAcousticLookAheadStartSplittingAtIteration(config_);

    Core::Application::us()->log() << "computing acoustic lookahead on " << sets.list.size() << " sets";
    // Step 1: Load single-gaussian models

    verify(modelForIndex_.empty());
    verify(acousticLookAheadModels_.empty());

    std::set<Mm::DensityIndex> hadDensity;

    std::vector<AcousticFeatureVector> means;
    for (u32 mean = 0; mean < mixtureSet_->nMeans(); ++mean)
        means.push_back(acousticLookAheadScorer_->multiplyAndQuantize(*mixtureSet_->mean(mean))[0]);

    // Actually the rand() is pseudo-random because we didn't seed the random number generator yet
    //   srand(0);

    // Step 2: Initialize each lookahead-model with a random single-gaussian distribution from the single-gaussian models
    for (int a = 0; a < lookaheadModelCount_; ++a) {
        Mm::DensityIndex pick = 0;
        while (hadDensity.count(pick))
            pick = rand() % mixtureSet_->nDensities();
        hadDensity.insert(pick);
        acousticLookAheadModels_.push_back(AcousticLookAheadModel(means[mixtureSet_->density(pick)->meanIndex()]));
    }

    std::cout << "used " << lookaheadModelCount_ << " out of " << mixtureSet_->nDensities() << " densities" << std::endl;

    if (!multiplicity_)
        for (u32 m = 0; m < sets.list.size(); ++m)
            sets.list[m].second = 1;

    // Step 3: Iterate:
    //    -  Assign each state to the best-matching look-ahead state
    //    -  Re-compute a new single-gaussian distribution for the look-ahead state

    Core::Application::us()->log() << "shared look-ahead sets: " << sets.list.size();

    std::vector<u32> modelForSet(sets.list.size(), invalidLookAheadModel);

    for (int i = 0; i < iterations_; ++i) {
        std::vector<u32> assignedSetsPerModel(acousticLookAheadModels_.size(), 0);
        std::vector<u32> assignedWeightPerModel(acousticLookAheadModels_.size(), 0);

        f64 totalDistance = 0;

        // Maps each look-ahead model to all component mixtures that make up this model
        // and the multiplicity of each component.
        std::vector<std::map<Am::AcousticModel::EmissionIndex, u32>> assignments(acousticLookAheadModels_.size());

        Core::Application::us()->log() << "Acoustic lookahead iteration " << i + 1 << " of " << iterations_;

        u32 totalCount = 0;

        for (u32 m = 0; m < sets.list.size(); ++m) {
            const std::set<Am::AcousticModel::EmissionIndex>& emissions(sets.list[m].first);
            u32                                               mult = sets.list[m].second;
            totalCount += mult;

            u32   bestLookahead      = invalidLookAheadModel;
            Score bestLookAheadScore = Core::Type<Score>::max;

            for (u32 lookahead = 0; lookahead < acousticLookAheadModels_.size(); ++lookahead) {
                Score score = 0;
                for (std::set<Am::AcousticModel::EmissionIndex>::const_iterator it = emissions.begin(); it != emissions.end(); ++it) {
                    verify(*it != StateTree::invalidAcousticModel);
                    u32 nDensities = mixtureSet_->mixture(*it)->nDensities();
                    verify(nDensities == 1);
                    Mm::GaussDensity*            dens = mixtureSet_->density(mixtureSet_->mixture(*it)->densityIndex(0));
                    const AcousticFeatureVector& mean(means[dens->meanIndex()]);

                    score += acousticLookAheadModels_.at(lookahead).distance(mean);
                }

                if (score < bestLookAheadScore) {
                    bestLookahead      = lookahead;
                    bestLookAheadScore = score;
                }
            }

            verify(bestLookahead != invalidLookAheadModel);
            assignedSetsPerModel[bestLookahead] += 1;
            assignedWeightPerModel[bestLookahead] += mult;
            totalDistance += bestLookAheadScore * mult;
            modelForSet[m] = bestLookahead;

            // Update the assignment
            for (std::set<Am::AcousticModel::EmissionIndex>::const_iterator it = emissions.begin(); it != emissions.end(); ++it) {
                if (assignments[bestLookahead].count(*it))
                    assignments[bestLookahead][*it] += mult;
                else
                    assignments[bestLookahead][*it] = mult;
            }
        }

        Core::Application::us()->log() << "assignment distance: " << totalDistance / totalCount;

        Core::Application::us()->log() << "Estimating";

        for (u32 lookahead = 0; lookahead < acousticLookAheadModels_.size(); ++lookahead) {
            std::vector<u32> assignedMeans;

            for (std::map<Mm::MixtureIndex, u32>::const_iterator assignIt = assignments[lookahead].begin(); assignIt != assignments[lookahead].end(); ++assignIt) {
                Mm::MixtureIndex  mix  = (*assignIt).first;
                Mm::GaussDensity* dens = mixtureSet_->density(mixtureSet_->mixture(mix)->densityIndex(0));
                for (u32 m = 0; m < (*assignIt).second; ++m)
                    assignedMeans.push_back(dens->meanIndex());
            }

            /// @todo Introduce weights for the splits, try out the better models again
            acousticLookAheadModels_[lookahead].estimate(assignedMeans, means, i >= startSplittingAtIteration ? splitsPerState : 0);
        }

        if ((splittingThreshold_ != 0.0 || splitEmpty_) && i + 1 != iterations_) {
            std::vector<std::pair<u32, u32>> sorted;
            for (u32 lookahead = 0; lookahead < acousticLookAheadModels_.size(); ++lookahead)
                sorted.push_back(std::make_pair(lookahead, assignedWeightPerModel[lookahead]));

            std::sort(sorted.begin(), sorted.end(),
                      Core::composeBinaryFunction(std::less<u32>(),
                                                  Core::select2nd<std::pair<u32, u32>>(),
                                                  Core::select2nd<std::pair<u32, u32>>()));

            u32 splitted = 0;

            u32 splitUntil = sorted.size() * splittingThreshold_;
            while (splitEmpty_ && splitUntil < sorted.size() && sorted[splitUntil].second == 0)
                ++splitUntil;

            std::vector<u32> splitPotential = assignedSetsPerModel;

            for (u32 i = 0; i < splitUntil; ++i) {
                u32 eliminate = sorted[i].first;

                u32 split = eliminate;
                for (u32 s = sorted.size() - 1; s > i; --s) {
                    u32 n = sorted[s].first;
                    if (splitPotential[n] > 0) {
                        split = n;
                        splitPotential[split] -= 1;
                        break;
                    }
                }

                if (split == eliminate)
                    break;  // Found nothing to split

                ++splitted;
                acousticLookAheadModels_[eliminate].split(acousticLookAheadModels_[split]);
            }
            Core::Application::us()->log() << "splitted " << splitted;
        }

        {
            Core::HistogramStatistics stats("emission-sets assigned per look-ahead node");
            for (u32 lookahead = 0; lookahead < acousticLookAheadModels_.size(); ++lookahead)
                stats += assignedSetsPerModel[lookahead];
            stats.write(Core::Application::us()->log());
        }

        {
            Core::HistogramStatistics stats("weight assigned per look-ahead node");
            for (u32 lookahead = 0; lookahead < acousticLookAheadModels_.size(); ++lookahead)
                stats += assignedWeightPerModel[lookahead];
            stats.write(Core::Application::us()->log());
        }
    }

    modelForIndex_.resize(sets.setForIndex.size(), invalidLookAheadModel);
    for (u32 index = 0; index < sets.setForIndex.size(); ++index)
        if (sets.setForIndex[index] != Core::Type<u32>::max)
            modelForIndex_[index] = modelForSet[sets.setForIndex[index]];

    // More efficient representation in means_
    ///@todo Make this the only representation
    verify(means_.empty());
    for (std::vector<AcousticLookAheadModel>::const_iterator it = acousticLookAheadModels_.begin(); it != acousticLookAheadModels_.end(); ++it) {
        verify((*it).means_.size() == 1);  // Splits are not supported (not useful)
        means_.insert(means_.end(), (*it).means_.front().begin(), (*it).means_.front().end());
    }

    saveModels();
}

int AcousticLookAhead::length() const {
    return acousticLookaheadDepth_;
}

const Core::Configuration& AcousticLookAhead::config() const {
    return config_;
}

void AcousticLookAhead::getSuccessorMixtures(const PersistentStateTree&                  network,
                                             StateId                                     state,
                                             std::set<Am::AcousticModel::EmissionIndex>& target,
                                             int                                         depth,
                                             bool                                        includeCurrent) {
    verify(state > 0 && state < network.structure.stateCount());
    if (includeCurrent) {
        Am::AcousticModel::EmissionIndex mixture = network.structure.state(state).stateDesc.acousticModel;
        verify(mixture != StateTree::invalidAcousticModel);
        target.insert(mixture);
    }

    if (depth == 0)
        return;

    for (HMMStateNetwork::SuccessorIterator successor = network.structure.successors(state); successor; ++successor) {
        if (not successor.isLabel()) {
            if (depth > 0) {
                StateId succState = *successor;
                verify(succState > 0 && succState < network.structure.stateCount());
                Mm::MixtureIndex mixture = network.structure.state(succState).stateDesc.acousticModel;
                if (mixture == StateTree::invalidAcousticModel) {
                    // Skip
                    getSuccessorMixtures(network, succState, target, depth);
                }
                else {
                    verify(mixture != StateTree::invalidAcousticModel);
                    target.insert(mixture);
                    getSuccessorMixtures(network, succState, target, depth - 1);
                }
            }
        }
        else {
            // Epsilon-transition
            verify(successor.label() < network.exits.size());
            getSuccessorMixtures(network, network.exits[successor.label()].transitState, target, depth);
        }
    }
}

void AcousticLookAhead::setLookAhead(std::vector<Mm::FeatureVector> lookahead) {
    if (!acousticLookAheadScorer_)
        return;

    if (lookahead.size() > acousticLookaheadDepth_)
        lookahead.resize(acousticLookaheadDepth_);

    acousticLookAhead_.clear();

    for (u32 a = 0; a < lookahead.size(); ++a) {
        acousticLookAhead_.push_back(std::make_pair(acousticLookAheadScorer_->multiplyAndQuantize(lookahead.at(a))[0],
                                                    acousticLookAheadScorer_->getScorer(lookahead.at(a))));
        verify(acousticLookAhead_[a].first.size());
    }
}

struct Stats {
    size_t cached, computed;
    size_t cacheSizeBefore, cacheSize, cacheSizeSamples;
    Stats()
            : cached(0), computed(0), cacheSizeBefore(0), cacheSize(0), cacheSizeSamples(0) {
    }
    ~Stats() {
        if (cached || computed) {
            std::cout << "acoustic look-ahead items cached: " << cached << " computed " << computed << std::endl;
            if (cacheSizeSamples) {
                std::cout << "average cache size before cleanup: " << (cacheSizeBefore / cacheSizeSamples) << " after: " << (cacheSize / cacheSizeSamples) << std::endl;
            }
        }
    }
} stats;

void AcousticLookAhead::startLookAhead(int timeframe, bool computeAll) {
    if (!isEnabled())
        return;

    if (currentLookAheadScores_.empty())  // Eventually initialize cache
    {
        currentLookAheadScores_.resize(acousticLookAheadModels_.size(),
                                       std::make_pair(-1, (Score)0));
        preCachedLookAheadScores_.resize(currentLookAheadScores_.size());
    }

    for (int a = 0; a < length(); ++a)
        if (cachesForTimeframes_[a]->acousticScoreCache_.empty()) {
            // Eventually initialize cache
            cachesForTimeframes_[a]->acousticScoreCache_.resize(acousticLookAheadModels_.size(), std::make_pair(-1, (Score)0));
            cachesForTimeframes_[a]->simpleScoreCache_.resize(acousticLookAheadModels_.size(), 0.0f);
        }

    if (currentTimeFrame_ + 1 == timeframe) {
        // Push the caches by one column to the left, so that we find the cached values
        CacheForTimeframe* first = cachesForTimeframes_.front();
        for (int a = 1; a < cachesForTimeframes_.size(); ++a)
            cachesForTimeframes_[a - 1] = cachesForTimeframes_[a];
        cachesForTimeframes_.back() = first;
    }

    currentTimeFrame_ = timeframe;

    stats.cacheSizeBefore += nextCacheKey_ - minCacheKey_;
    stats.cacheSize += nextCacheKey_ - minCacheKey_;
    ++stats.cacheSizeSamples;

    if (computeAll)
        computeAllLookAheadScores();
}

void AcousticLookAhead::clear() {
    // We have to clear the caches, because in future, timeframes may be different
    currentLookAheadScores_.clear();
    for (int a = 0; a < length(); ++a)
        cachesForTimeframes_[a]->clear();
    currentTimeFrame_ = -1;
    minCacheKey_ = nextCacheKey_ = 1;
}

void AcousticLookAhead::fillCacheForTimeframe(int timeframe) {
    verify(timeframe >= currentTimeFrame_ && timeframe < currentTimeFrame_ + acousticLookaheadDepth_);
    CacheForTimeframe& cache     = cacheForTimeframe(timeframe);
    u32                dimension = acousticLookAhead_.front().first.size();

    if (cache.simpleCacheTimeframe_ != timeframe) {
        cache.simpleCacheTimeframe_ = timeframe;
        // Fill the cache
        for (int lookaheadId = 0; lookaheadId < acousticLookAheadModels_.size(); ++lookaheadId)
            cache.simpleScoreCache_[lookaheadId] = calculateDistance(means_.data() + (dimension * lookaheadId),
                                                                     acousticLookAhead_[timeframe - currentTimeFrame_].first.data(),
                                                                     dimension);
    }
}

template<bool useAverage>
void AcousticLookAhead::computeAllLookAheadScoresInternal() {
    if (acousticLookAhead_.empty()) {
        preCachedLookAheadScores_.assign(preCachedLookAheadScores_.size(), 0);
        return;
    }

    uint len = acousticLookaheadDepth_;
    if (acousticLookAhead_.size() < len)
        len = acousticLookAhead_.size();

    float divFac = len != 0 ? (acousticLookAheadScale_ / len) : 0;

    if (useAverage) {
        // Special-case for depth 1: Most efficient

        if (len == 1) {
            u32 dimension = acousticLookAhead_[0].first.size();

            for (int lookaheadId = 0; lookaheadId < acousticLookAheadModels_.size(); ++lookaheadId)
                preCachedLookAheadScores_[lookaheadId] = calculateDistance(means_.data() + (dimension * lookaheadId),
                                                                           acousticLookAhead_[0].first.data(),
                                                                           dimension) *
                                                         divFac;
            return;
        }

        // Quick computation with cache

        fillCacheForTimeframe(currentTimeFrame_);

        {  // Initialize
            const CacheForTimeframe& cache = cacheForTimeframe(currentTimeFrame_);
            for (int lookaheadId = 0; lookaheadId < acousticLookAheadModels_.size(); ++lookaheadId)
                preCachedLookAheadScores_[lookaheadId] = cache.simpleScoreCache_[lookaheadId];
        }

        // Add intermediate timeframes
        for (uint a = 1; a < len - 1; ++a) {
            fillCacheForTimeframe(currentTimeFrame_ + a);
            const CacheForTimeframe& cache = cacheForTimeframe(currentTimeFrame_ + a);
            for (int lookaheadId = 0; lookaheadId < acousticLookAheadModels_.size(); ++lookaheadId)
                preCachedLookAheadScores_[lookaheadId] += cache.simpleScoreCache_[lookaheadId];
        }

        {
            // Finalize: Add last timeframe and apply averaging-factor
            fillCacheForTimeframe(currentTimeFrame_ + len - 1);
            const CacheForTimeframe& cache = cacheForTimeframe(currentTimeFrame_ + len - 1);

            for (int lookaheadId = 0; lookaheadId < acousticLookAheadModels_.size(); ++lookaheadId)
                preCachedLookAheadScores_[lookaheadId] = (preCachedLookAheadScores_[lookaheadId] + cache.simpleScoreCache_[lookaheadId]) * divFac;
        }

        return;
    }

    std::vector<Score>::iterator end = preCachedLookAheadScores_.end();
    for (std::vector<Score>::iterator it = preCachedLookAheadScores_.begin(); it != end; ++it) {
        u32    lookaheadId = it - preCachedLookAheadScores_.begin();
        Score& score(*it);

        if (useAverage) {
            verify(0);  // Handled above
        }
        else {
            score = Core::Type<Score>::max;

            uint len = acousticLookaheadDepth_;
            if (acousticLookAhead_.size() < len)
                len = acousticLookAhead_.size();

            for (uint a = 0; a < len; ++a) {
                Score localScore = getCachedScaledScore(a + currentTimeFrame_, lookaheadId);
                if (localScore < score)
                    score = localScore;
            }
        }
    }
}

void AcousticLookAhead::computeAllLookAheadScores() {
    if (useAverage_)
        computeAllLookAheadScoresInternal<true>();
    else
        computeAllLookAheadScoresInternal<false>();
}

Core::Configuration AcousticLookAhead::select(const Core::Configuration& config) {
    return Core::Configuration(config, "acoustic-lookahead");
}
