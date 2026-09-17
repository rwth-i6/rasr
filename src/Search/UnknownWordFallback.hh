/** Copyright 2026 RWTH Aachen University. All rights reserved.
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
#ifndef SEARCH_UNKNOWN_WORD_FALLBACK_HH
#define SEARCH_UNKNOWN_WORD_FALLBACK_HH

#include <unordered_set>
#include <vector>

#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Core/Hash.hh>
#include <Core/Parameter.hh>
#include <Search/Types.hh>

namespace Search {

/**
 * Policy object for the open-vocabulary subword fallback.
 *
 * It resolves the configured fallback mode and tokenization together with the
 * special lemma groups that describe the fallback pieces, and is constructed
 * identically by the tree builders (which realize the topology) and by the
 * search algorithm (which realizes the word-boundary and scoring semantics).
 * Both are given the same `Core::Configuration`, so both see the same policy.
 */
class UnknownWordFallback : public Core::Component {
public:
    enum Mode {
        // No fallback at all, even if the lexicon defines fallback lemmas.
        Disabled,
        // The fallback sub-tree alone defines the semantics: a piece lemma with a
        // syntactic token produces a word-LM event, one without does not. There is
        // no known-word exclusion. This reproduces the behavior of earlier setups.
        Legacy,
        // Word boundaries and word-LM events are resolved by the search, and a
        // piece sequence which is an exact known pronunciation is never scored as
        // an unknown word.
        KnownExcluding,
    };

    enum Tokenization {
        // A piece marked as "continuation" keeps the word open, a piece marked as
        // "final" ends it (e.g. BPE with a trailing "@@" on continuations).
        ContinuationMarked,
        // A piece marked as "word-start" closes a pending word and opens a new one,
        // every other piece extends the pending word (e.g. SentencePiece "_").
        WordStartMarked,
    };

    /**
     * Role of a lemma within the fallback sub-tree.
     */
    enum PieceRole {
        NotFallback,
        // ContinuationMarked
        Continuation,
        Final,
        // WordStartMarked
        WordStart,
        WordInternal,
    };

    static const Core::Choice          choiceMode;
    static const Core::ParameterChoice paramMode;
    static const Core::Choice          choiceTokenization;
    static const Core::ParameterChoice paramTokenization;
    static const Core::ParameterFloat  paramUnknownWordPenalty;

    UnknownWordFallback(Core::Configuration const& config, Bliss::Lexicon const& lexicon);

    Mode mode() const {
        return mode_;
    }

    Tokenization tokenization() const {
        return tokenization_;
    }

    bool enabled() const {
        return mode_ != Disabled;
    }

    bool excludesKnownWords() const {
        return mode_ == KnownExcluding;
    }

    /**
     * Additive cost `beta` charged once per completed unknown word, outside of all
     * LM scales and in RASR's internal (negative natural log) score convention.
     * Positive discourages the fallback, negative rewards it.
     */
    Score unknownWordPenalty() const {
        return unknownWordPenalty_;
    }

    PieceRole roleOf(Bliss::Lemma const* lemma) const;

    bool isFallbackLemma(Bliss::Lemma const* lemma) const {
        return roleOf(lemma) != NotFallback;
    }

    /**
     * Lemmas the ordinary part of the search tree must not contain: the fallback
     * pieces themselves plus the conventional singleton "unknown" replacement lemma,
     * which the orthographic parser and training pipelines still need but which must
     * not become an ordinary in-vocabulary word.
     */
    bool isExcludedFromOrdinaryTree(Bliss::Lemma const* lemma) const {
        // Independent of the configured mode: as soon as a lexicon describes a
        // fallback inventory, its pieces are fallback pieces. With the fallback
        // disabled they are simply absent from the tree, which is the closed
        // vocabulary baseline the fallback modes are compared against.
        return not allFallbackLemmas_.empty() and
               (lemma == unknownLemma_ or allFallbackLemmas_.find(lemma) != allFallbackLemmas_.end());
    }

    /**
     * True if this fallback piece carries no lexical content: it marks a boundary
     * (and, as a word-start piece, closes a pending word) but does not itself begin
     * or extend a word. Declared by additionally listing the piece lemma in the
     * lexicon's `special="nonword"` group; this is how a standalone word-start
     * marker is distinguished from a word-start piece with lexical content.
     */
    bool isSeparator(Bliss::Lemma const* lemma) const {
        return separatorLemmas_.find(lemma) != separatorLemmas_.end();
    }

    /**
     * True if a piece with this role terminates the pending word *after* being
     * emitted. Only meaningful for ContinuationMarked.
     */
    bool closesWordAfter(PieceRole role) const {
        return tokenization_ == ContinuationMarked && role == Final;
    }

    /**
     * True if a piece with this role terminates a pending word *before* being
     * emitted, i.e. it belongs to the following word. Only meaningful for
     * WordStartMarked.
     */
    bool closesWordBefore(PieceRole role) const {
        return tokenization_ == WordStartMarked && role == WordStart;
    }

    /**
     * Lemmas whose pronunciations may open a fallback word at an ordinary root,
     * in a deterministic order.
     */
    std::vector<Bliss::Lemma const*> const& openingLemmas() const {
        return openingLemmas_;
    }

    /**
     * Lemmas whose pronunciations may occur inside a pending fallback word,
     * in a deterministic order.
     */
    std::vector<Bliss::Lemma const*> const& pendingLemmas() const {
        return pendingLemmas_;
    }

    /**
     * The syntactic token each word LM is advanced with for a genuine unknown
     * word. Null if the fallback is disabled.
     */
    Bliss::SyntacticToken const* unknownSyntacticToken() const {
        return unknownSyntacticToken_;
    }

    /**
     * Short description of the active policy, used for logging and as part of the
     * search-tree image identity.
     */
    std::string describe() const;

    /**
     * Value folded into the persistent search-tree image so that an image built
     * under a different policy is not silently reused.
     */
    u32 topologyKey() const;

private:
    using LemmaRoleMap = std::unordered_map<Bliss::Lemma const*, PieceRole>;
    using LemmaSet     = std::unordered_set<Bliss::Lemma const*>;

    // Collect `lexicon.specialLemmas(name)` in a deterministic (lemma id) order.
    std::vector<Bliss::Lemma const*> collectGroup(Bliss::Lexicon const& lexicon, std::string const& name) const;

    void resolveContinuationMarked(Bliss::Lexicon const& lexicon);
    void resolveWordStartMarked(Bliss::Lexicon const& lexicon);
    void resolveUnknownSyntacticToken(Bliss::Lexicon const& lexicon);

    Mode         mode_;
    Tokenization tokenization_;
    Score        unknownWordPenalty_;

    std::vector<Bliss::Lemma const*> openingLemmas_;
    std::vector<Bliss::Lemma const*> pendingLemmas_;
    LemmaRoleMap                     roles_;
    // Membership in any fallback group, independent of the configured mode.
    LemmaSet allFallbackLemmas_;
    // Fallback pieces which carry no lexical content.
    LemmaSet separatorLemmas_;

    Bliss::Lemma const*          unknownLemma_;
    Bliss::SyntacticToken const* unknownSyntacticToken_;
};

}  // namespace Search

#endif  // SEARCH_UNKNOWN_WORD_FALLBACK_HH
