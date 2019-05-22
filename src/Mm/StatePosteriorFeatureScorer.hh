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
#ifndef _MM_STATE_POSTERIOR_FEATURE_SCORER_HH
#define _MM_STATE_POSTERIOR_FEATURE_SCORER_HH

#include <Core/Assertions.hh>
#include <Core/Channel.hh>
#include <Core/Component.hh>
#include <Core/Hash.hh>
#include <Core/Parameter.hh>
#include <Core/Types.hh>
#include "AssigningFeatureScorer.hh"
#include "DensityToWeightMap.hh"
#include "MixtureFeatureScorerElement.hh"
#include "MixtureSet.hh"
#include "Types.hh"

namespace Mm {

/** Base feature-scorer for MixtureSet of log linear densities.
 *  Scores are calculated with maximum approximation.
 */
class StatePosteriorFeatureScorer : public CachedAssigningFeatureScorer {
    typedef CachedAssigningFeatureScorer Precursor;

public:
    typedef DensityToWeightMap                       PosteriorsAndDensities;
    typedef std::unordered_map<MixtureIndex, Weight> PosteriorsAndMixtures;
    typedef PosteriorsAndMixtures                    LikelihoodAndMixtures;
    typedef std::vector<DensityIndex>                Topology;
    class Filter : public Core::ReferenceCounted {
        typedef std::unordered_map<MixtureIndex, Weight> PriorMap;

    public:
        typedef PriorMap::const_iterator const_iterator;

    protected:
        PriorMap map_;

    public:
        Filter() {}
        virtual ~Filter() {}

        void clear() {
            map_.clear();
        }
        size_t erase(const MixtureIndex& mixtureIndex) {
            return map_.erase(mixtureIndex);
        }
        Weight& operator[](const MixtureIndex& mixtureIndex) {
            return map_[mixtureIndex];
        }
        const_iterator begin() const {
            return map_.begin();
        }
        const_iterator end() const {
            return map_.end();
        }
        const_iterator find(MixtureIndex mixtureIndex) const {
            return map_.find(mixtureIndex);
        }
        size_t size() const {
            return map_.size();
        }
    };
    typedef Core::Ref<Filter>       FilterRef;
    typedef Core::Ref<const Filter> ConstFilterRef;
    struct IndexAndWeight {
        MixtureIndex e;
        Weight       w;  // -log(x)
        IndexAndWeight(MixtureIndex _e, Weight _w)
                : e(_e), w(_w) {}
    };
    typedef std::vector<IndexAndWeight> IndicesAndWeights;

public:
    class CachedStatePosteriorContextScorer : public CachedAssigningContextScorer {
        friend class StatePosteriorFeatureScorer;

    public:
        typedef PosteriorsAndDensities ScoresAndDensities;
        typedef PosteriorsAndDensities ScoresAndMixtures;

    private:
        enum IndexType {
            none             = 0,
            densityScore     = 1,
            densityPosterior = 2,
            density          = 3,
            mixtureScore     = 4,
            mixturePosterior = 8,
            mixture          = 12
        };

    protected:
        Core::Ref<const Feature>   feature_;
        mutable ScoresAndDensities scores_;
        mutable ScoresAndDensities p_;
        mutable Weight             logZ_;
        mutable DensityIndex       minimumIndex_;
        mutable Weight             minimumScore_;
        mutable FilterRef          filter_;
        Weight                     scale_;
        mutable int                initialize_;

    protected:
        CachedStatePosteriorContextScorer(Core::Ref<const Feature>           feature,
                                          const StatePosteriorFeatureScorer* featureScorer,
                                          size_t                             cacheSize);

        void            _workDensityScores(MixtureIndex marginMixture = invalidMixture) const;
        void            workMixtureScores() const;
        void            pruneScores() const;
        virtual void    workPosteriors() const;
        virtual void    workLikelihoods() const;
        void            workDensityScores(MixtureIndex marginMixture = invalidMixture) const;
        void            workDensityPosteriors(MixtureIndex marginMixture = invalidMixture) const;
        void            workMixturePosteriors() const;
        void            workMixtureLikelihoods() const;
        const Topology& topology(MixtureIndex mixtureIndex) const {
            return featureScorer()->topology(mixtureIndex);
        }

    private:
        const StatePosteriorFeatureScorer* featureScorer() const {
            return required_cast(const StatePosteriorFeatureScorer*, featureScorer_);
        }

    public:
        void reset() const;
        void setFilter(FilterRef filter, const std::vector<s32>& disregardDensities) {
            filter_ = filter;
            reset();
            for (std::vector<s32>::const_iterator it = disregardDensities.begin(); it != disregardDensities.end(); ++it) {
                filter_->erase(*it);
            }
        }
        void setScale(Weight scale) {
            scale_ = scale;
        }
        const ScoresAndDensities&     scoresAndDensities() const;
        const ScoresAndMixtures&      scoresAndMixtures() const;
        const PosteriorsAndDensities& posteriorsAndDensities(MixtureIndex marginMixture = invalidMixture) const;
        const PosteriorsAndMixtures&  posteriorsAndMixtures() const;
        const LikelihoodAndMixtures&  likelihoodAndMixtures() const;
        Weight                        minimumScore() const {
            return minimumScore_;
        }
        DensityIndex minimumIndex() const {
            return minimumIndex_;
        }
        // online calculation of posteriors
        void   posteriorsAndMixtures(IndicesAndWeights& priors) const;
        Weight logZ() const {
            return logZ_;
        }
    };

private:
    static const Core::ParameterFloat     paramScale;
    static const Core::ParameterFloat     paramPruningThreshold;
    static const Core::ParameterBool      paramViterbi;
    static const Core::ParameterFloat     paramMargin;
    static const Core::ParameterIntVector paramDisregardDensities;

protected:
    Core::Ref<const AssigningFeatureScorer> fs_;
    Weight                                  scale_;
    Weight                                  pruningThreshold_;
    bool                                    viterbi_;
    bool                                    contextPriors_;
    mutable FilterRef                       defaultFilter_;
    FilterRef                               filter_;
    mutable Core::XmlChannel                statisticsChannel_;
    Weight                                  margin_;
    std::vector<s32>                        disregardDensities_;

protected:
    virtual AssigningFeatureScorer::ScoreAndBestDensity calculateScoreAndDensity(
            const CachedAssigningFeatureScorer::CachedAssigningContextScorer* cs, MixtureIndex mixtureIndex) const;
    FilterRef defaultFilter() const;

public:
    StatePosteriorFeatureScorer(const Core::Configuration&);
    StatePosteriorFeatureScorer(const Core::Configuration&, Core::Ref<const AbstractMixtureSet>);

    virtual Core::Ref<const AssigningContextScorer> getAssigningScorer(Core::Ref<const Feature>) const;
    virtual Core::Ref<const AssigningContextScorer> getAssigningScorer(const FeatureVector&) const;

    virtual MixtureIndex nMixtures() const {
        return fs_->nMixtures();
    }
    virtual ComponentIndex dimension() const {
        return fs_->dimension();
    }
    virtual DensityIndex nDensities() const {
        return fs_->nDensities();
    }
    virtual const std::vector<DensityIndex>& densitiesInMixture(MixtureIndex mix) const {
        return fs_->densitiesInMixture(mix);
    }

    void setDefaultFilter() {
        setFilter(defaultFilter());
    }
    void setFilter(const MixtureIndex& mixtureIndex);

    void setFilter(FilterRef filter) {
        filter_ = filter;
    }
    void setFeatureScorer(Core::Ref<const AssigningFeatureScorer> fs) {
        require(fs);
        fs_ = fs;
    }
    Core::Ref<const AssigningFeatureScorer> getFeatureScorer() const {
        require(fs_);
        return fs_;
    }
    bool useViterbi() const {
        return viterbi_;
    }
    const Topology& topology(MixtureIndex mixtureIndex) const {
        return densitiesInMixture(mixtureIndex);
    }
};

}  // namespace Mm

#endif  // _MM_STATE_POSTERIOR_FEATURE_SCORER_HH
