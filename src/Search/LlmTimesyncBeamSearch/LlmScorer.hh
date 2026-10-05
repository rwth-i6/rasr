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

#ifndef LLM_SCORER_HH
#define LLM_SCORER_HH

#include <functional>
#include <string>
#include <vector>

#include <Core/Choice.hh>
#include <Core/Component.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>
#include <Search/Types.hh>

namespace Search {

/*
 * Token index in the vocabulary of an external (large) language model.
 * Unrelated to the label indices of the acoustic model and to the syntactic tokens of the lexicon.
 */
typedef s32                           LlmToken;
typedef std::vector<LlmToken>         LlmTokenSequence;
typedef std::vector<LlmTokenSequence> LlmTokenSequenceVariants;

/*
 * Abstract interface to a token-level language model with its own tokenizer, e.g. an LLM such as Qwen.
 *
 * The search hands finished words to `tokenize` as plain text and scores the resulting tokens with
 * `scoreContinuations`. A text may be tokenized into several variants, e.g. of different casing. The search
 * scores all of them and greedily continues with the cheapest one.
 *
 * The language model itself is stateless from the point of view of the search: a history is always passed as
 * the full token sequence it consists of, so an implementation is free to cache e.g. key/value states by token
 * prefix. Caching of tokenizations and of scores is done by the search, so the same request is never sent twice
 * within a segment.
 */
class LlmScorer : public virtual Core::Component,
                  public Core::ReferenceCounted {
public:
    LlmScorer(Core::Configuration const& config)
            : Core::Component(config) {}
    virtual ~LlmScorer() = default;

    // Prepare for a new segment, e.g. drop key/value caches of the previous one.
    virtual void reset() = 0;

    // Tokens every history starts with, e.g. a begin-of-sequence token and/or a prompt.
    virtual LlmTokenSequence initialTokens() = 0;

    // Tokens scored once at the end of a segment, e.g. an end-of-sequence token. May be empty.
    virtual LlmTokenSequence sentenceEndTokens() = 0;

    /*
     * Tokenize each of the given texts independently into one or more variants, e.g. the text as is, lowercased and
     * capitalized. Each text needs at least one variant and every variant at least one token.
     */
    virtual std::vector<LlmTokenSequenceVariants> tokenize(std::vector<std::string> const& texts) = 0;

    /*
     * For each request `i`, return the cost (negative natural log-probability) of every token of
     * `continuations[i]` given `prefixes[i]` and all preceding tokens of `continuations[i]`.
     * The result for request `i` must have the same length as `continuations[i]`.
     */
    virtual std::vector<std::vector<Score>> scoreContinuations(std::vector<LlmTokenSequence> const& prefixes,
                                                               std::vector<LlmTokenSequence> const& continuations) = 0;
};

/*
 * Factory to register types of LlmScorers by name and create them from a config.
 * Allows registering implementations from other places, in particular from Python via librasr.
 */
class LlmScorerFactory {
private:
    // Needs to be declared before `paramLlmScorerType` because the latter refers to it
    Core::Choice choices_;

public:
    typedef std::function<Core::Ref<LlmScorer>(Core::Configuration const&)> CreationFunction;

    Core::ParameterChoice paramLlmScorerType;

    LlmScorerFactory();

    void registerLlmScorer(const char* name, CreationFunction creationFunction);

    // Create an instance of the type given by `paramLlmScorerType`.
    Core::Ref<LlmScorer> createLlmScorer(Core::Configuration const& config) const;

private:
    std::vector<CreationFunction> registry_;
};

}  // namespace Search

#endif  // LLM_SCORER_HH
