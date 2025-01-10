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
#ifndef SEARCH_V2_HH
#define SEARCH_V2_HH

#include <Am/AcousticModel.hh>
#include <Bliss/CorpusDescription.hh>
#include <Bliss/Lexicon.hh>
#include <Nn/LabelScorer/LabelScorer.hh>
#include <Nn/Types.hh>
#include <Search/LatticeAdaptor.hh>
#include <Search/Types.hh>
#include <Speech/Feature.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Types.hh>

namespace Search {

// Abstract base class for search algorithms that can work in an online
// or offline manner.
class SearchAlgorithmV2 : public virtual Core::Component {
public:
    // Struct to keep track of a collection of scores
    struct ScoreVector {
        Score acoustic, lm;

        ScoreVector(Score am, Score lm)
                : acoustic(am), lm(lm) {}
        operator Score() const {
            return acoustic + lm;
        };
        ScoreVector operator+(ScoreVector const& other) const {
            return ScoreVector(acoustic + other.acoustic, lm + other.lm);
        }
        ScoreVector operator-(ScoreVector const& other) const {
            return ScoreVector(acoustic - other.acoustic, lm - other.lm);
        }
        ScoreVector& operator+=(ScoreVector const& other) {
            acoustic += other.acoustic;
            lm += other.lm;
            return *this;
        }
        ScoreVector& operator-=(ScoreVector const& other) {
            acoustic -= other.acoustic;
            lm -= other.lm;
            return *this;
        }
    };

    // Struct to store data for a single traceback entry
    struct TracebackItem {
    public:
        const Bliss::LemmaPronunciation* pronunciation;  // pronunciation for lattice creation
        const Bliss::Lemma*              lemma;          // possible no pronunciation
        // Flow::Timestamp                  time;           // start-/end-time of current traceback item
        Speech::TimeframeIndex time;    // TODO: Proper word boundaries with Flow::Timestamp instead of indices
        ScoreVector            scores;  // Absolute score
        // TracebackItem(const Bliss::LemmaPronunciation* p, const Bliss::Lemma* l, Flow::Timestamp t, ScoreVector s)
        TracebackItem(const Bliss::LemmaPronunciation* p, const Bliss::Lemma* l, Speech::TimeframeIndex t, ScoreVector s)
                : pronunciation(p), lemma(l), time(t), scores(s) {}
    };

    // List of TracebackItems representing a full traceback path
    class Traceback : public std::vector<TracebackItem>, public Core::ReferenceCounted {
    public:
        void                    write(std::ostream& os, Core::Ref<const Bliss::PhonemeInventory>) const;
        Fsa::ConstAutomatonRef  lemmaAcceptor(Core::Ref<const Bliss::Lexicon>) const;
        Fsa::ConstAutomatonRef  lemmaPronunciationAcceptor(Core::Ref<const Bliss::Lexicon>) const;
        Lattice::WordLatticeRef wordLattice(Core::Ref<const Bliss::Lexicon>) const;
    };

public:
    SearchAlgorithmV2(const Core::Configuration&)
            : Core::Component(config) {}
    virtual ~SearchAlgorithmV2() = default;

    // ModelCombination to set important Modules in recognition
    virtual Speech::ModelCombination::Mode modelCombinationNeeded() const = 0;

    // Needed mode for acoustic model (only in case it's required in the model combination)
    virtual Am::AcousticModel::Mode acousticModelNeeded() const {
        return Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTying | Am::AcousticModel::noStateTransition;
    }

    virtual bool setModelCombination(const Speech::ModelCombination& modelCombination) = 0;

    // Call before starting a new recognition. Clean up existing data structures
    // from the previous run.
    virtual void reset() = 0;

    // Call at the beginning of a new segment.
    // A segment can be one recording segment in a corpus for offline recognition
    // or one chunk of audio for online recognition.
    virtual void enterSegment()                            = 0;
    virtual void enterSegment(Bliss::SpeechSegment const*) = 0;

    // Call after all features of the current segment have been passed
    virtual void finishSegment() = 0;

    // Pass a single feature vector
    virtual void addFeature(std::shared_ptr<const f32[]> const& data, size_t F) = 0;
    virtual void addFeature(std::vector<f32> const& data)                       = 0;

    // Pass feature vectors for multiple time steps
    virtual void addFeatures(std::shared_ptr<const f32[]> const& data, size_t T, size_t F) = 0;

    // Return the current best traceback. May contain unstable results.
    virtual Core::Ref<const Traceback> getCurrentBestTraceback() const = 0;

    // Similar to `getCurrentBestTraceback` but return the lattice instead of traceback.
    virtual Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const = 0;

    virtual void resetStatistics()     = 0;
    virtual void logStatistics() const = 0;

    // Try to decode one more step. Return bool indicates whether a step could be made.
    virtual bool decodeStep() = 0;

    // Decode as much as possible given the currently available features. Return bool indicates whether any steps could be made.
    virtual bool decodeMore() {
        bool success = false;
        while (decodeStep()) {
            // Set to true of at least one iteration is successful
            success = true;
        }

        return success;
    }
};

}  // namespace Search

#endif  // SEARCH_V2_HH
