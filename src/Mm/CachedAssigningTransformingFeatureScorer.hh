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
#ifndef _MM_CACHED_ASSIGNING_TRANSFORMING_FEATURE_SCORER_HH_
#define _MM_CACHED_ASSIGNING_TRANSFORMING_FEATURE_SCORER_HH_

#include "AssigningFeatureScorer.hh"

namespace Mm
{

/** Abstract feature scorer interface with cached scores and assignment using feature transforming distance functions . */
class CachedAssigningTransformingFeatureScorer: public CachedAssigningFeatureScorer
{
    typedef CachedAssigningFeatureScorer Precursor;
public:
    struct ScoringResult : AssigningFeatureScorer::ScoreAndBestDensity {
        Score score;
        DensityInMixture bestDensity;
        FeatureVector transformedFeature;
    };

    //@todo: should be protected
    /** Implement emission independent precalculations for feature vector */
    class CachedAssigningTransformingContextScorer: public CachedAssigningFeatureScorer::CachedAssigningContextScorer
    {
        typedef CachedAssigningFeatureScorer::CachedAssigningContextScorer ContextPrecursor;
    protected:
        const CachedAssigningTransformingFeatureScorer *featureScorer_;
        mutable Cache<FeatureVector> transformedFeatureCache_;
    protected:
        CachedAssigningTransformingContextScorer(const CachedAssigningTransformingFeatureScorer *featureScorer, EmissionIndex nEmissions) :
            ContextPrecursor(featureScorer, nEmissions), featureScorer_(featureScorer), transformedFeatureCache_(nEmissions)
        {
        }
    public:
        virtual ~CachedAssigningTransformingContextScorer()
        {
        }

        virtual Score score(EmissionIndex e, FeatureVector *transformedFeature = 0) const
        {
            require_(0 <= e && e < nEmissions());
            if (!ContextPrecursor::cache_.isCalculated(e)) {
                ScoringResult r = featureScorer_->calculateScoreAndDensityAndFeature(this, e, transformedFeature);
                transformedFeatureCache_.set(e, r.transformedFeature);
                return ContextPrecursor::cache_.set(e, r).score;
            }
            *transformedFeature = transformedFeatureCache_[e]; //copy from cache
            return ContextPrecursor::cache_[e].score;
        }
        virtual DensityInMixture bestDensity(EmissionIndex e, FeatureVector *transformedFeature = 0) const
        {
            require_(0 <= e && e < nEmissions());
            if (!ContextPrecursor::cache_.isCalculated(e)) {
                ScoringResult r = featureScorer_->calculateScoreAndDensityAndFeature(this, e, transformedFeature);
                transformedFeatureCache_.set(e, r.transformedFeature);
                return ContextPrecursor::cache_.set(e, r).bestDensity;
            }
            *transformedFeature = transformedFeatureCache_[e]; //copy from cache
            return ContextPrecursor::cache_[e].bestDensity;
        }
    };
    virtual ScoringResult calculateScoreAndDensityAndFeature(const CachedAssigningContextScorer*, MixtureIndex, FeatureVector *transformedFeature) const = 0;
public:
    CachedAssigningTransformingFeatureScorer(const Core::Configuration &c) :
        Core::Component(c), Precursor(c)
    {
    }
    virtual ~CachedAssigningTransformingFeatureScorer()
    {
    }

};

}

#endif /* _MM_CACHED_ASSIGNING_TRANSFORMING_FEATURE_SCORER_HH_ */
