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

#include "UnknownWordFallback.hh"

#include <algorithm>
#include <sstream>

namespace {

// Resolved from the lexicon: "legacy" if the lexicon defines fallback lemmas,
// "disabled" otherwise. Keeps setups which predate this parameter reproducible.
constexpr int ModeAuto = 3;

constexpr char const* continuationGroup = "unknown-continuation";
constexpr char const* finalGroup        = "unknown-final";
constexpr char const* wordStartGroup    = "unknown-word-start";
constexpr char const* wordInternalGroup = "unknown-word-internal";

}  // namespace

namespace Search {

const Core::Choice UnknownWordFallback::choiceMode(
        "disabled", UnknownWordFallback::Disabled,
        "legacy", UnknownWordFallback::Legacy,
        "known-excluding", UnknownWordFallback::KnownExcluding,
        "auto", ModeAuto,
        Core::Choice::endMark());

const Core::ParameterChoice UnknownWordFallback::paramMode(
        "unknown-word-fallback",
        &UnknownWordFallback::choiceMode,
        "Open-vocabulary subword fallback policy. \"legacy\" lets the fallback lemmata's syntactic tokens drive the "
        "word-LM events without any known-word exclusion. \"known-excluding\" resolves word boundaries and word-LM "
        "events in the search and never scores an exact known pronunciation as an unknown word. \"auto\" selects "
        "\"legacy\" if the lexicon defines fallback lemmata and \"disabled\" otherwise.",
        ModeAuto);

const Core::Choice UnknownWordFallback::choiceTokenization(
        "continuation-marked", UnknownWordFallback::ContinuationMarked,
        "word-start-marked", UnknownWordFallback::WordStartMarked,
        Core::Choice::endMark());

const Core::ParameterChoice UnknownWordFallback::paramTokenization(
        "unknown-word-tokenization",
        &UnknownWordFallback::choiceTokenization,
        "Word-boundary convention of the subword inventory. \"continuation-marked\" expects the special lemma groups "
        "\"unknown-continuation\"/\"unknown-final\" (e.g. BPE with a trailing \"@@\"). \"word-start-marked\" expects "
        "\"unknown-word-start\"/\"unknown-word-internal\" (e.g. SentencePiece with a leading word-start marker).",
        UnknownWordFallback::ContinuationMarked);

const Core::ParameterFloat UnknownWordFallback::paramUnknownWordPenalty(
        "unknown-word-penalty",
        "Additive cost charged once per completed unknown word, outside of all LM scales, in RASR's internal "
        "(negative natural logarithm) score convention. Positive values discourage the fallback, negative values "
        "reward it. Has no effect in \"legacy\" mode.",
        0.0);

UnknownWordFallback::UnknownWordFallback(Core::Configuration const& config, Bliss::Lexicon const& lexicon)
        : Core::Component(config),
          mode_(Disabled),
          tokenization_(static_cast<Tokenization>(paramTokenization(config))),
          unknownWordPenalty_(paramUnknownWordPenalty(config)),
          openingLemmas_(),
          pendingLemmas_(),
          roles_(),
          unknownLemma_(lexicon.specialLemma("unknown")),
          unknownSyntacticToken_(nullptr) {
    for (auto const* group : {continuationGroup, finalGroup, wordStartGroup, wordInternalGroup}) {
        for (auto const* lemma : collectGroup(lexicon, group)) {
            allFallbackLemmas_.insert(lemma);
        }
    }

    bool const hasContinuationGroups = not lexicon.specialLemmas(continuationGroup).empty();
    bool const hasWordStartGroups    = not lexicon.specialLemmas(wordStartGroup).empty();

    int const configuredMode = paramMode(config);
    if (configuredMode == ModeAuto) {
        mode_ = (hasContinuationGroups or hasWordStartGroups) ? Legacy : Disabled;
    }
    else {
        mode_ = static_cast<Mode>(configuredMode);
    }

    if (mode_ == Disabled) {
        return;
    }

    switch (tokenization_) {
        case ContinuationMarked: resolveContinuationMarked(lexicon); break;
        case WordStartMarked: resolveWordStartMarked(lexicon); break;
    }

    if (mode_ == Disabled) {  // A resolve step may have turned the fallback off again
        return;
    }

    resolveUnknownSyntacticToken(lexicon);

    // A fallback piece which is also a "nonword" carries no lexical content: a
    // standalone word-start marker, for instance, separates words without being one.
    for (auto const* lemma : lexicon.specialLemmas("nonword")) {
        if (roles_.find(lemma) != roles_.end()) {
            separatorLemmas_.insert(lemma);
        }
    }

    log() << "Open-vocabulary subword fallback: " << describe();
}

std::vector<Bliss::Lemma const*> UnknownWordFallback::collectGroup(Bliss::Lexicon const& lexicon, std::string const& name) const {
    auto const&                      group = lexicon.specialLemmas(name);
    std::vector<Bliss::Lemma const*> result(group.begin(), group.end());
    // `specialLemmas` returns an unordered set; sort so that the resulting search
    // tree (and therefore its cached image) does not depend on hash iteration order.
    std::sort(result.begin(), result.end(), [](Bliss::Lemma const* a, Bliss::Lemma const* b) {
        return a->id() < b->id();
    });
    return result;
}

void UnknownWordFallback::resolveContinuationMarked(Bliss::Lexicon const& lexicon) {
    std::vector<Bliss::Lemma const*> continuationLemmas = collectGroup(lexicon, continuationGroup);
    std::vector<Bliss::Lemma const*> finalLemmas        = collectGroup(lexicon, finalGroup);

    if (not collectGroup(lexicon, wordStartGroup).empty() or not collectGroup(lexicon, wordInternalGroup).empty()) {
        criticalError("Lexicon defines word-start-marked fallback lemmata, but unknown-word-tokenization is \"continuation-marked\".");
    }

    if (continuationLemmas.empty()) {
        if (not finalLemmas.empty()) {
            criticalError("Special lemma \"%s\" requires at least one \"%s\" lemma.", finalGroup, continuationGroup);
        }
        mode_ = Disabled;
        return;
    }

    if (finalLemmas.empty()) {
        // Legacy form: a singleton "unknown" lemma carries all final-piece pronunciations.
        if (unknownLemma_ == nullptr) {
            criticalError("Special lemma \"%s\" requires a special lemma named \"unknown\" or \"%s\".", continuationGroup, finalGroup);
            mode_ = Disabled;
            return;
        }
        finalLemmas.push_back(unknownLemma_);
    }

    for (auto const* lemma : continuationLemmas) {
        if (lemma->syntacticTokenSequence().size() != 0) {
            criticalError("Special lemma \"%s\" (\"%s\") must have an empty syntactic token sequence.",
                          continuationGroup, lemma->name().str());
        }
        if (lemma->nPronunciations() == 0) {
            criticalError("Special lemma \"%s\" (\"%s\") must have at least one pronunciation.",
                          continuationGroup, lemma->name().str());
        }
        roles_[lemma] = Continuation;
    }

    for (auto const* lemma : finalLemmas) {
        size_t const numSyntacticTokens = lemma->syntacticTokenSequence().size();
        if (mode_ == Legacy) {
            if (numSyntacticTokens != 1) {
                criticalError("Special lemma \"%s\"/\"unknown\" (\"%s\") must have exactly one syntactic token.",
                              finalGroup, lemma->name().str());
            }
        }
        else if (numSyntacticTokens > 1) {
            criticalError("Special lemma \"%s\" (\"%s\") must have at most one syntactic token.",
                          finalGroup, lemma->name().str());
        }
        if (lemma->nPronunciations() == 0) {
            criticalError("Special lemma \"%s\"/\"unknown\" (\"%s\") must have at least one pronunciation.",
                          finalGroup, lemma->name().str());
        }
        roles_[lemma] = Final;
    }

    // Both piece kinds may open a fallback word and both may occur inside one.
    openingLemmas_ = continuationLemmas;
    openingLemmas_.insert(openingLemmas_.end(), finalLemmas.begin(), finalLemmas.end());
    pendingLemmas_ = openingLemmas_;
}

void UnknownWordFallback::resolveWordStartMarked(Bliss::Lexicon const& lexicon) {
    std::vector<Bliss::Lemma const*> wordStartLemmas    = collectGroup(lexicon, wordStartGroup);
    std::vector<Bliss::Lemma const*> wordInternalLemmas = collectGroup(lexicon, wordInternalGroup);

    if (not collectGroup(lexicon, continuationGroup).empty()) {
        criticalError("Lexicon defines continuation-marked fallback lemmata, but unknown-word-tokenization is \"word-start-marked\".");
    }

    if (wordStartLemmas.empty() and wordInternalLemmas.empty()) {
        mode_ = Disabled;
        return;
    }
    if (wordStartLemmas.empty()) {
        criticalError("Special lemma \"%s\" requires at least one \"%s\" lemma; otherwise no fallback word can begin.",
                      wordInternalGroup, wordStartGroup);
        mode_ = Disabled;
        return;
    }

    if (mode_ != KnownExcluding) {
        // In legacy mode the word-LM event is taken from the piece lemma itself, which
        // cannot express "this piece closes the *previous* word". Failing here avoids
        // silently decoding a word-start inventory with continuation-marked semantics.
        criticalError("unknown-word-tokenization \"word-start-marked\" requires unknown-word-fallback \"known-excluding\".");
        mode_ = Disabled;
        return;
    }

    auto checkPiece = [&](Bliss::Lemma const* lemma, char const* group, PieceRole role) {
        if (lemma->syntacticTokenSequence().size() != 0) {
            // The search closes the pending word when it sees the *next* word-start
            // piece, so a syntactic token on the piece itself would be applied to the
            // wrong word. Reject instead of ignoring it.
            criticalError("Special lemma \"%s\" (\"%s\") must have an empty syntactic token sequence; "
                          "the word-LM event of a word-start-marked fallback word is applied by the search.",
                          group, lemma->name().str());
        }
        if (lemma->nPronunciations() == 0) {
            criticalError("Special lemma \"%s\" (\"%s\") must have at least one pronunciation.", group, lemma->name().str());
        }
        roles_[lemma] = role;
    };

    for (auto const* lemma : wordStartLemmas) {
        checkPiece(lemma, wordStartGroup, WordStart);
    }
    for (auto const* lemma : wordInternalLemmas) {
        checkPiece(lemma, wordInternalGroup, WordInternal);
    }

    // Only a word-start piece may open a fallback word at an ordinary root. A first
    // piece without the word-start marker is reachable because the search seeds an
    // additional hypothesis in the pending-word root at the start of a segment.
    openingLemmas_ = wordStartLemmas;

    // Inside a pending word, a word-internal piece extends it and a word-start piece
    // closes it and opens the next one.
    pendingLemmas_ = wordStartLemmas;
    pendingLemmas_.insert(pendingLemmas_.end(), wordInternalLemmas.begin(), wordInternalLemmas.end());
}

void UnknownWordFallback::resolveUnknownSyntacticToken(Bliss::Lexicon const& lexicon) {
    // Prefer the conventional singleton "unknown" lemma, so that the unknown token
    // does not depend on which fallback piece happened to end the word.
    if (unknownLemma_ != nullptr and unknownLemma_->syntacticTokenSequence().size() == 1) {
        unknownSyntacticToken_ = unknownLemma_->syntacticTokenSequence().front();
    }

    // Otherwise take it from the fallback lemmata which carry one, and require them to agree.
    for (auto const& [lemma, role] : roles_) {
        auto const& sts = lemma->syntacticTokenSequence();
        if (sts.size() != 1) {
            continue;
        }
        Bliss::SyntacticToken const* token = sts.front();
        if (unknownSyntacticToken_ == nullptr) {
            unknownSyntacticToken_ = token;
        }
        else if (unknownSyntacticToken_ != token) {
            criticalError("Fallback lemmata disagree about the unknown syntactic token: \"%s\" vs. \"%s\".",
                          unknownSyntacticToken_->symbol().str(), token->symbol().str());
        }
    }

    if (unknownSyntacticToken_ == nullptr) {
        criticalError("The open-vocabulary fallback needs an unknown syntactic token. Define a special lemma "
                      "\"unknown\" with exactly one syntactic token (the word LM's unknown symbol).");
        mode_ = Disabled;
        return;
    }

    log() << "Unknown syntactic token: \"" << unknownSyntacticToken_->symbol().str()
          << "\" (id " << unknownSyntacticToken_->id() << ")";

    (void)lexicon;
}

UnknownWordFallback::PieceRole UnknownWordFallback::roleOf(Bliss::Lemma const* lemma) const {
    auto it = roles_.find(lemma);
    return it == roles_.end() ? NotFallback : it->second;
}

std::string UnknownWordFallback::describe() const {
    std::stringstream ss;
    switch (mode_) {
        case Disabled: return "disabled";
        case Legacy: ss << "legacy"; break;
        case KnownExcluding: ss << "known-excluding"; break;
    }
    ss << ", " << (tokenization_ == ContinuationMarked ? "continuation-marked" : "word-start-marked");
    ss << ", " << openingLemmas_.size() << " opening and " << pendingLemmas_.size() << " pending piece lemmata";
    if (not separatorLemmas_.empty()) {
        ss << " (" << separatorLemmas_.size() << " of them separators without lexical content)";
    }
    if (mode_ == KnownExcluding) {
        ss << ", unknown-word-penalty " << unknownWordPenalty_;
    }
    return ss.str();
}

u32 UnknownWordFallback::topologyKey() const {
    if (mode_ == Disabled) {
        // Zero is the value a tree built without any fallback carries, including one
        // built by a tree builder that does not know about the fallback at all.
        return 0u;
    }

    // Runtime-only parameters such as the unknown-word penalty are deliberately not
    // part of this key: they change the search result but not the tree topology.
    size_t key = static_cast<size_t>(mode_) * 7u + static_cast<size_t>(tokenization_) * 13u + 1u;
    for (auto const* lemma : openingLemmas_) {
        key = Core::combineHashes(key, static_cast<size_t>(lemma->id()) + 1u);
    }
    for (auto const* lemma : pendingLemmas_) {
        key = Core::combineHashes(key, static_cast<size_t>(lemma->id()) + 2u);
    }
    // Never collide with the "no fallback" value.
    return std::max<u32>(1u, static_cast<u32>(key));
}

}  // namespace Search
