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
// $Id$

#ifndef _SPEECH_ALIGNMENT_HH
#define _SPEECH_ALIGNMENT_HH

#include <Am/AcousticModel.hh>
#include <Am/ClassicStateModel.hh>
#include <Core/BinaryStream.hh>
#include <Core/XmlStream.hh>
#include <Fsa/Semiring.hh>
#include <Mc/Types.hh>
#include <Mm/Types.hh>
#include "Types.hh"

namespace Speech {

struct AlignmentItem {
    TimeframeIndex time;
    Fsa::LabelId   emission;
    Mm::Weight     weight;
    AlignmentItem()
            : time(0), emission(0), weight(0) {}
    AlignmentItem(TimeframeIndex t, Am::AllophoneStateIndex e, Mm::Weight w = 1.0)
            : time(t), emission(e), weight(w) {}
    bool operator==(const AlignmentItem& other) const {
        return (time == other.time && emission == other.emission && weight == other.weight);
    }
};

// ================================================================================

class Alignment;
Core::BinaryInputStream&  operator>>(Core::BinaryInputStream&, Alignment&);
Core::BinaryOutputStream& operator<<(Core::BinaryOutputStream&, const Alignment&);

/*
 * Alignments are usually stored as a sequence of allophone state ids.
 * This is useful, because it makes the alignment independent of the state tying.
 * For some purposes, the state tying can be assumed to be fix. Mapping the alignment
 * to the emission label can cause some overhead with our implementation.
 * In particular, reading the lattice-alignment
 * in sequence-discriminative training is expensive.
 * Therefore, we also allow writing the emission indices directly.
 */
class Alignment : public std::vector<AlignmentItem> {
public:
    typedef std::pair<Alignment::iterator, Alignment::iterator> Frame;
    enum LabelType { allophoneStateIds,
                     emissionIds };

private:
    typedef std::vector<AlignmentItem> Precursor;
    static const char*                 magic;
    static const char*                 magic_alphabet;
    static const char*                 magic_emission;
    static const size_t                magicSize;
    Score                              score_;
    Core::Ref<const Fsa::Alphabet>     alphabet_;
    // If the archive was read, then this contains the read alphabet information (cleared when the mapping is applied)
    std::map<u32, std::string> archiveAlphabet_;
    LabelType                  labelType_;

public:
    Alignment();

    Alignment(const Alignment& alignment);

    void setScore(Score score) {
        score_ = score;
    }

    /** Returns the score of the alignment, if available. */
    Score score() const {
        return score_;
    }

    /** Set a mapping alphabet. When this is set, the alphabet is used to robustly map
     *  allophone indices between different alphabets.
     *  If skipMismatch is true, then alignment items which could not be mapped into the new
     *  alphabet are simply removed. Otherwise an error is raised for such items. */
    bool setAlphabet(Core::Ref<const Fsa::Alphabet> alphabet, bool skipMismatch = false);

    /** Returns true, if at least one alignment item has a weight different from one. */
    bool hasWeights() const;

    /** Sorts the alignment items, such that timeframes have ascending order and
     *  secondly weight have descending order. */
    void sortItems(bool byDecreasingWeight = true);
    void sortStableItems();

    /** Combines all alignment items which differ only in their weight. */
    void combineItems(Fsa::ConstSemiringRef sr = Fsa::TropicalSemiring);

    /** Builds weights from negative logarithm of item weights. */
    void expm();

    /** Adds @param weight to each item weight. */
    void addWeight(Mm::Weight weight);

    /** Filter alignment items by their weights. (>= min && <= max) */
    void filterWeights(Mm::Weight minWeight, Mm::Weight maxWeight = 1.0);

    /** Remove alignment items by their weights. (> min) */
    void filterWeightsGT(Mm::Weight minWeight);

    /** Normalize weights, such that for each timeframe the sum of weights is one. */
    void normalizeWeights();

    /** shift per time such that for each timeframe the min of weights is 0. */
    void shiftMinToZeroWeights();

    /** Clip all weights into the interval [a..b] */
    void clipWeights(Mm::Weight a = 0.0, Mm::Weight b = 1.0);

    /** */
    void resetWeightsSmallerThan(Mm::Weight a = 0.0, Mm::Weight b = 0.0);

    /** */
    void resetWeightsLargerThan(Mm::Weight a = 1.0, Mm::Weight b = 1.0);

    /** Multiply all weights with the specified value */
    void multiplyWeights(Mm::Weight c);

    /** Raise all weights to the given power gamma. Note that the weights are not normalized afterwards.
     *  This function is intended to approximate a re-alignment with a different acoustic-model scale. */
    void gammaCorrection(Mc::Scale gamma);

    /** For each time frame, get a pair of begin and end iterator */
    void getFrames(std::vector<Frame>& rows);

    /** Component-wise weight multiplication */
    Alignment& operator*=(const Alignment&);

    /** get alignment labelType */
    LabelType labelType() const {
        return labelType_;
    }

    /** set alignment labelType, only allowed for empty alignments, no mapping is performed */
    void setLabelType(LabelType labeltype) {
        require_eq(this->size(), 0);
        labelType_ = labeltype;
    }

    /** change alignment label type to emission id,
     * all allophone state ids are mapped to emission ids */
    void mapToEmissionIdLabels(const Core::Ref<Am::AcousticModel>& acousticModel);

    friend Core::BinaryInputStream&  operator>>(Core::BinaryInputStream&, Alignment&);
    friend Core::BinaryOutputStream& operator<<(Core::BinaryOutputStream&, const Alignment&);
    void                             write(std::ostream&) const;
    void                             writeXml(Core::XmlWriter&) const;
    friend Core::XmlWriter&          operator<<(Core::XmlWriter&, const Alignment&);
    void                             addTimeOffset(TimeframeIndex);

private:
    void mapAlphabet(bool skipMismatch);
};

}  // namespace Speech

namespace Core {

template<>
class NameHelper<Speech::Alignment> : public std::string {
public:
    NameHelper()
            : std::string("flow-alignment") {}
};

}  // namespace Core

#endif  // _SPEECH_ALIGNMENT_HH
