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
#ifndef ADVANCEDTREESEARCH_ACOUSTICLOOKAHEAD_HH
#define ADVANCEDTREESEARCH_ACOUSTICLOOKAHEAD_HH

#include <Mm/SimdFeatureScorer.hh>
#include <Search/Types.hh>
#include "Helpers.hh"
#include "SearchSpace.hh"
#include "TreeStructure.hh"
// #include "LinearMiniHash.hh"

namespace Search {
class SearchSpace;
}

struct EmissionSetCounter {
    // Returns the model, and increaases the counter
    u32 get(const std::set<u32>& emissions, u32 index, bool record = true) {
        u32 model;

        std::unordered_map<std::set<u32>, u32, SetHash<u32>>::iterator it = assignment.find(emissions);
        if (it == assignment.end()) {
            model = list.size();
            list.push_back(std::make_pair(emissions, 0u));
            assignment.insert(std::make_pair(emissions, model));
        }
        else
            model = it->second;

        if (record) {
            list[model].second += 1;

            extendIndex(index + 1);

            setForIndex[index] = model;
        }

        return model;
    }

    void clear(u32 index) {
        extendIndex(index + 1);
        setForIndex[index] = Core::Type<u32>::max;
    }

    void extendIndex(u32 minSize) {
        if (setForIndex.size() < minSize)
            setForIndex.resize(minSize, Core::Type<u32>::max);
    }

    bool contains(u32 index) {
        return setForIndex.size() > index && setForIndex[index] != Core::Type<u32>::max;
    }

    std::unordered_map<std::set<u32>, u32, SetHash<u32>> assignment;
    // Pair of the emissions and the count
    std::vector<std::pair<std::set<u32>, u32>> list;
    // The set which was assigned to each index
    std::vector<u32> setForIndex;
};

namespace AdvancedTreeSearch {
class AcousticLookAhead {
public:
    AcousticLookAhead(const Core::Configuration& config, u32 checksum, bool load = true);
    virtual ~AcousticLookAhead();

    bool loaded() const {
        return loaded_;
    }

    /// If loading failed,
    void initializeModelsFromNetwork(const Search::PersistentStateTree& network);
    void initializeModels(EmissionSetCounter sets);

    typedef Search::Score Score;

    static int         getDepth(const Core::Configuration& config);
    static Score       getScale(const Core::Configuration& config);
    static std::string getMixtureSetFilename(const Core::Configuration& config);

    static bool isEnabled(const Core::Configuration& config) {
        return getDepth(config) != 0 && getScale(config) != 0 && getMixtureSetFilename(config).size() != 0;
    }

    /// Whether acoustic look-ahead has been enabled through the configuration
    bool isEnabled() const {
        return acousticLookaheadDepth_ != 0 && acousticLookAheadScale_ != 0;
    }

    struct ApplyNoLookahead {
        ApplyNoLookahead(const AcousticLookAhead&) {
        }

        inline Search::Score operator()(u32, Search::StateId) const {
            return 0.0f;
        }
    };

    struct ApplyPreCachedLookAheadForId {
        ApplyPreCachedLookAheadForId(const AcousticLookAhead& lah)
                : lah_(lah) {
        }

        inline Search::Score operator()(u32 id, Search::StateId) const {
            return lah_.getPreCachedLookAheadScoreForId(id);
        }
        const AcousticLookAhead& lah_;
    };

    /// Has to be called at each timeframe, after the look-ahead feature vectors have been added,
    /// and before getLookAheadScore() is called for the nodes.
    /// @param computeAll If this is true, all look-ahead scores are computed immediately, and can be retrieved through getPreCachedLookAheadScore
    void startLookAhead(int timeframe, bool computeAll = false);

    inline void getPreCachedLookAheadScores(Score* target, u32 index, u32 count) const {
        const u32* model    = modelForIndex_.data() + index;
        const u32* endModel = model + count;
        for (; model != endModel; ++model, ++target)
            *target = preCachedLookAheadScores_[*model];
    }

    inline Search::Score getPreCachedLookAheadScore(u32 index) {
        return preCachedLookAheadScores_[modelForIndex_[index]];
    }

    void setDefaultModel(u32 defaultModel, u32 minSize = 0) {
        if (modelForIndex_.size() < minSize)
            modelForIndex_.resize(minSize);
        for (u32 i = 0; i < modelForIndex_.size(); ++i)
            if (modelForIndex_[i] == Core::Type<u32>::max)
                modelForIndex_[i] = defaultModel;
    }

    inline Search::Score getPreCachedLookAheadScoreForId(u32 id) const {
        return preCachedLookAheadScores_[id];
    }

    inline u32 getLookaheadId(u32 index) {
        return modelForIndex_[index];
    }

    inline Search::Score getCachedLookAheadScore(u32 index) {
        u32 lookaheadId = modelForIndex_[index];
        return currentLookAheadScores_[lookaheadId].second;
    }

    inline Search::Score getLookAheadScore(u32 index) {
        u32 lookaheadId = modelForIndex_[index];

        verify_(lookaheadId < currentLookAheadScores_.size());

        std::pair<int, Score>& timeAndScore(currentLookAheadScores_[lookaheadId]);

        Score& score(timeAndScore.second);

        if (timeAndScore.first != currentTimeFrame_) {
            // We have to compute the look-ahead score
            timeAndScore.first = currentTimeFrame_;

            if (useAverage_) {
                u32 len = acousticLookaheadDepth_;
                if (acousticLookAhead_.size() < len)
                    len = acousticLookAhead_.size();
                score   = 0;
                int cnt = 0;
                for (u32 a = 0; a < len; ++a) {
                    score += getCachedScaledScore(a + currentTimeFrame_, lookaheadId);
                    ++cnt;
                }
                if (cnt)
                    score /= cnt;
            }
            else {
                score = Core::Type<Score>::max;

                u32 len = acousticLookaheadDepth_;
                if (acousticLookAhead_.size() < len)
                    len = acousticLookAhead_.size();

                for (u32 a = 0; a < len; ++a) {
                    Score localScore = getCachedScaledScore(a + currentTimeFrame_, lookaheadId);
                    if (localScore < score)
                        score = localScore;
                }
            }
        }

        return score;
    }

    /// Depth of the acoustic look-ahead (how many timeframes does it look into the future?)
    /// A matching number of features must be fed into the acoustic look-ahead before it can
    /// compute look-ahead scores
    int length() const;

    void setLookAhead(std::vector<Mm::FeatureVector> lookahead);

    void clear();

private:
    void computeAllLookAheadScores();

    template<bool useAverage>
    void computeAllLookAheadScoresInternal();

    void fillCacheForTimeframe(int currentTimeFrame_);

    // Pairs of time-frame and score (only current if same timeframe)
    typedef std::vector<std::pair<int, Search::Score>> CacheVector;
    struct CacheForTimeframe {
        CacheVector                acousticScoreCache_;
        std::vector<Search::Score> simpleScoreCache_;
        int                        maxDepth_;
        int                        simpleCacheTimeframe_;

        CacheForTimeframe(int maxDepth)
                : maxDepth_(maxDepth), simpleCacheTimeframe_(-1) {
            clear();
        }

        struct CacheValue {
            CacheValue()
                    : timeframe(-1), cacheKey(0), score(0.0f) {
            }
            CacheValue(int t, u32 c, Score s)
                    : timeframe(t), cacheKey(c), score(s) {
            }
            int   timeframe;
            u32   cacheKey;
            Score score;
        };
        std::vector<std::vector<CacheValue>> cachesPerDepth_;

        void clear() {
            acousticScoreCache_.clear();
            simpleScoreCache_.clear();
            cachesPerDepth_.clear();
            cachesPerDepth_.resize(maxDepth_ + 1);
            simpleCacheTimeframe_ = -1;
        }

        CacheValue& cacheValue(int depth, u32 cacheKey, u32 minCacheKey) {
            verify_(depth < cachesPerDepth_.size());
            verify_(cacheKey >= minCacheKey);
            u32                      address = cacheKey - minCacheKey;
            std::vector<CacheValue>& depthCache(cachesPerDepth_[depth]);
            if (depthCache.size() <= address)
                depthCache.resize(100 + cacheKey + (cacheKey / 3));
            verify_(address < depthCache.size());
            return depthCache[address];
        }
    };

    inline CacheForTimeframe& cacheForTimeframe(int timeframe) {
        int offset = timeframe - currentTimeFrame_;
        verify_(offset < cachesForTimeframes_.size());
        return *cachesForTimeframes_[offset];
    }

    inline Score getCachedScaledScore(int timeframe, u32 model) {
        int offset = timeframe - currentTimeFrame_;
        verify_(offset < cachesForTimeframes_.size());
        verify_(model < acousticLookAheadModels_.size());
        std::pair<int, Score>& timeAndScore(cachesForTimeframes_[offset]->acousticScoreCache_.at(model));

        if (timeAndScore.first != timeframe) {
            timeAndScore.first  = timeframe;
            timeAndScore.second = acousticLookAheadModels_[model].distance(acousticLookAhead_[offset].first) * acousticLookAheadScale_;
        }
        return timeAndScore.second;
    }

    const Core::Configuration& config() const;

    typedef Mm::SimdGaussDiagonalMaximumFeatureScorer::PreparedFeatureVector AcousticFeatureVector;

    static Score calculateDistance(const AcousticFeatureVector& mean1, const AcousticFeatureVector& mean2);

    static Score calculateDistance(const Mm::SimdGaussDiagonalMaximumFeatureScorer::QuantizedType* mean1,
                                   const Mm::SimdGaussDiagonalMaximumFeatureScorer::QuantizedType* mean2, u32 dimension);

    class AcousticLookAheadModel {
    public:
        AcousticLookAheadModel(const AcousticFeatureVector& mean = AcousticFeatureVector()) {
            if (!mean.empty())
                means_.push_back(mean);
        }

        AcousticLookAheadModel(Core::MappedArchiveReader reader);

        Score distance(const AcousticFeatureVector& mean) const;

        u32 dimension() const {
            verify(!means_.empty());
            return means_[0].size();
        }

        void split(const AcousticLookAheadModel& other) {
            means_ = other.means_;
            for (std::vector<AcousticFeatureVector>::iterator it = means_.begin(); it != means_.end(); ++it)
                for (u32 a = 0; a < (*it).size(); ++a)
                    (*it)[a] += ((rand() % 2 == 0) ? 0.01 : -0.01);
        }

        void estimate(const std::vector<u32>& assigned, const std::vector<AcousticFeatureVector>& means, u32 splits);

        void write(Core::MappedArchiveWriter writer);

        std::vector<AcousticFeatureVector> means_;
    };

    // Convenience function that collects a specific set of successor mixtures of the given state
    void getSuccessorMixtures(const Search::PersistentStateTree& network, Search::StateId state, std::set<Am::AcousticModel::EmissionIndex>& target, int depth, bool includeCurrent = false);

    bool loadModels();
    void saveModels();

    f32  acousticLookAheadScale_;
    f32  splittingThreshold_;
    bool splitEmpty_;

    std::vector<AcousticLookAheadModel> acousticLookAheadModels_;
    AcousticFeatureVector               means_;  // More efficient linear representation of acousticLookAheadModels_
    // If time-conditioned acoustic look-ahead is used, then this maps to the simplified
    // Otherwise, it directly maps to the models in acousticLookAheadModels_.
    std::vector<u32> modelForIndex_;

    // Dynamic:
    std::vector<std::pair<AcousticFeatureVector, Mm::FeatureScorer::Scorer>> acousticLookAhead_;

    CacheVector                currentLookAheadScores_;
    std::vector<Search::Score> preCachedLookAheadScores_;

    std::vector<CacheForTimeframe*> cachesForTimeframes_;
    // Mixture-set used for the look-ahead (not necessarily the original mixture-set)
    Core::Ref<Mm::MixtureSet>                  mixtureSet_;
    Mm::SimdGaussDiagonalMaximumFeatureScorer* acousticLookAheadScorer_;

    Search::SearchSpace* ss_;
    s32                  acousticLookaheadDepth_;
    s32                  currentTimeFrame_;
    bool                 useAverage_;
    bool                 considerLabels_;
    s32                  lookaheadModelCount_;
    s32                  iterations_;
    bool                 multiplicity_;
    bool                 loaded_;

    // A cache key is only valid if its value is higher or equal to currentMinCacheKey
    u32 minCacheKey_;
    u32 nextCacheKey_;
    u32 checksum_;
    // Assigns a cache-key to each StateId
    std::vector<u32> cacheKeys_;

    std::string archiveEntry() const;

    inline u32 getCacheKeyForState(Search::StateId state) {
        u32& key(cacheKeys_[state]);
        if (key <= minCacheKey_) {
            key = nextCacheKey_++;
            // TODO: This may happen on wrap
            verify(key >= minCacheKey_);
        }
        return key;
    }

    static Core::Configuration select(const Core::Configuration& config);

    Core::Configuration config_;

    // Whether the model of a mapped state is included in the simplified target model.
    // If this is true, one less future feature vector is part of the model (as the used depth is always acousticLookAheadDepth_)
    // If this is false, then the look-ahead model represents strictly the successors of the state.
    bool includeCurrentStateModel_;
    f64  perDepthFactor_;
};
}  // namespace AdvancedTreeSearch

#endif  // ADVANCEDTREESEARCH_ACOUSTICLOOKAHEAD_HH
