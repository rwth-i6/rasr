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

// Token of an external LM with its own vocabulary, e.g. an LLM
typedef s32                   LlmToken;
typedef std::vector<LlmToken> LlmTokenSequence;

// Handle of a token sequence starting with the initial tokens; equal handles mean equal sequences within a segment
typedef u32 LlmHistory;

struct LlmScoringRequest {
    LlmHistory              history;         // History to continue
    LlmTokenSequence        prefix;          // Token sequence of `history`
    LlmTokenSequence        tokens;          // Tokens to score after `history`
    std::vector<LlmHistory> tokenHistories;  // History after each of `tokens`
};

/*
 * Token-level LM with its own tokenizer. Histories are identified by handles, so an implementation can keep
 * states such as key/value caches per handle and fall back to the prefix for handles it does not know.
 */
class LlmScorer : public virtual Core::Component,
                  public Core::ReferenceCounted {
public:
    LlmScorer(Core::Configuration const& config)
            : Core::Component(config) {}
    virtual ~LlmScorer() = default;

    // Start a new segment; handles of the previous one are not used again
    virtual void reset() = 0;

    // Tokens every history starts with, e.g. a begin-of-sequence token or a prompt; must not be empty
    virtual LlmTokenSequence initialTokens() = 0;

    // Tokens scored at the end of a segment; may be empty
    virtual LlmTokenSequence sentenceEndTokens() = 0;

    // Spellings to score for each word, e.g. in different casing; the cheapest one is used
    virtual std::vector<std::vector<std::string>> spellingVariants(std::vector<std::string> const& words);

    virtual std::vector<LlmTokenSequence> tokenize(std::vector<std::string> const& texts) = 0;

    // Cost (negative natural log-probability) of every token of every request
    virtual std::vector<std::vector<Score>> score(std::vector<LlmScoringRequest> const& requests) = 0;

    // Only `activeHistories` and their extensions are requested from now on
    virtual void cleanup(std::vector<LlmHistory> const& activeHistories) {}
};

// Registry of `LlmScorer` types, e.g. implemented in Python and registered via librasr
class LlmScorerFactory {
private:
    // Declared before `paramLlmScorerType`, which refers to it
    Core::Choice choices_;

public:
    typedef std::function<Core::Ref<LlmScorer>(Core::Configuration const&)> CreationFunction;

    Core::ParameterChoice paramLlmScorerType;

    LlmScorerFactory();

    void                 registerLlmScorer(const char* name, CreationFunction creationFunction);
    Core::Ref<LlmScorer> createLlmScorer(Core::Configuration const& config) const;

private:
    std::vector<CreationFunction> registry_;
};

}  // namespace Search

#endif  // LLM_SCORER_HH
