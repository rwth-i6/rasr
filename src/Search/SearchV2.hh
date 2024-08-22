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

#include <Bliss/CorpusDescription.hh>
#include <Search/LatticeAdaptor.hh>
#include <Speech/Feature.hh>

namespace Search {

// Abstract base class for search algorithms that can work in an online
// or offline manner.
class SearchAlgorithmV2 : public virtual Core::Component {
public:
    typedef Speech::Score          Score;
    typedef Speech::TimeframeIndex TimeframeIndex;

    // Struct to keep track of a collection of scores
    // E.g. AM score and LM score
    struct ScoreMap : std::unordered_map<std::string, Score> {  // TODO: unordered_map is bad, keep old ScoreVector with fixed entries
        using Precursor = std::unordered_map<std::string, Score>;
        ScoreMap()      = default;
        ScoreMap(std::initializer_list<value_type> init)
                : Precursor(init) {}

        Score    sum() const;
        ScoreMap operator+(ScoreMap const& other) const;
        ScoreMap operator-(ScoreMap const& other) const;
        ScoreMap operator+=(ScoreMap const& other);
        ScoreMap operator-=(ScoreMap const& other);
    };

    // Struct to store data for a single traceback entry
    struct TracebackItem {
    public:
        typedef Lattice::WordBoundary::Transit Transit;

    public:
        std::string    pronunciation;  // TODO: Use lexicon or not?
        TimeframeIndex time;           // Ending time
        ScoreMap       scores;         // Absolute score
        Transit        transit;        // Final transition description
        TracebackItem(const std::string& p, TimeframeIndex t, ScoreMap s, Transit te)
                : pronunciation(p), time(t), scores(s), transit(te) {}
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
    SearchAlgorithmV2(const Core::Configuration&);
    virtual ~SearchAlgorithmV2() = default;

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

    // Call to finalize the results of the current recognition run.
    // This can be after finishing a corpus segment or after finishing
    // a stream in online recognition.
    virtual void finalize() = 0;

    // Pass a single feature vector
    virtual void addFeature(Core::Ref<const Speech::Feature>) = 0;

    // TODO: Simplify interface, one function, bool in TracebackItem that shows if it's stable or not
    // TODO: Mechanism to tell the search to remove some context for very long form recognition

    // Return the longest partial traceback that is known to become a prefix
    // of the final best sequence. Only required for online recognition.
    virtual Core::Ref<const Traceback> stablePartialTraceback() = 0;

    // Like `stablePartialTraceback` but only return the most recent part
    // that's new after the last `stablePartialTraceback` or `recentStablePartialTraceback`
    // call. Only required for online recognition.
    virtual Core::Ref<const Traceback> recentStablePartialTraceback() = 0;

    // Return the part of the current best sequence after the stable traceback,
    // i.e. the part that may still change in the final best.
    // Only required for online recognition.
    virtual Core::Ref<const Traceback> unstablePartialTraceback() const = 0;

    // Return the current best sequence which is the composition of stable and
    // unstable partial traceback.
    virtual Core::Ref<const Traceback> getCurrentBestTraceback() const = 0;

    // Similar to `stablePartialTraceback` but return the partial lattice instead of
    // traceback.
    virtual Core::Ref<const LatticeAdaptor> getPartialWordLattice() = 0;

    // Similar to `getCurrentBestTraceback` but return the lattice instead of traceback.
    virtual Core::Ref<const LatticeAdaptor> getCurrentBestWordLattice() const = 0;

    virtual void resetStatistics()     = 0;
    virtual void logStatistics() const = 0;
};

}  // namespace Search

#endif  // SEARCH_V2_HH
