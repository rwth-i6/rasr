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
#ifndef _LM_FSA_LM_HH
#define _LM_FSA_LM_HH

#include <Bliss/Lexicon.hh>
#include "LanguageModel.hh"

namespace Lm {

class FsaLm : public LanguageModel {
public:
    using HistoryDescriptor = Fsa::State;

    static const Core::ParameterString paramFilename;
    static const Core::ParameterBool   paramGarbageLoopMode;
    static const Core::ParameterBool   paramAcceptPartialRepeat;

    FsaLm(const Core::Configuration& c, Bliss::LexiconRef);
    virtual ~FsaLm();

    virtual void                   load();
    void                           setFsa(Fsa::ConstAutomatonRef);
    virtual Fsa::ConstAutomatonRef getFsa() const {
        return fsa_;
    }

    virtual History startHistory() const;
    virtual History extendedHistory(const History&, Token w) const;
    virtual Score   score(const History&, Token w) const;
    virtual Score   sentenceEndScore(const History&) const;

    // efficient sparse lookahead: reachable next states' (token, score)
    // always use infinityScore for backOffScore
    bool isSparse(const History& h) const {
        return true;
    }

    virtual HistorySuccessors getHistorySuccessors(const History& h) const;

    Score getBackOffScore(const History& h) const {
        return infinityScore();
    }

    /**
     * Score for impossible events.
     * Returning Core::Type<Score>::max easily causes search
     * algorithms to fail.  The reason is that multiplication or
     * addition to the maximum value results in 'inf' and a second
     * aritmetic operation on 'inf' yields the value 'nan'.
     * Unfortunatelly the comparison operator will not work
     * normally on 'nan'.
     */
    Score infinityScore() const {
        return infinityScore_;
    }

protected:
    Score infinityScore_;
    // Note: accept & forward has higher priority than start over for the same input
    bool garbageLoopMode_;
    bool acceptPartialRepeat_;

    virtual Fsa::ConstStateRef initialState() const;
    virtual Fsa::ConstStateRef nextState(Fsa::ConstStateRef sp, Token w) const;
    virtual Score              stateScore(Fsa::ConstStateRef sp, Token w) const;
    virtual Score              stateSentenceEndScore(Fsa::ConstStateRef sp) const;
    virtual HistorySuccessors  getStateSuccessors(Fsa::ConstStateRef sp) const;

private:
    class HistoryManager;
    typedef FsaLm Self;

    static Fsa::ConstStateRef invalidHistory;

    Fsa::ConstAlphabetRef  syntacticTokens_;
    Fsa::ConstAutomatonRef fsa_;
};

}  // namespace Lm

#endif  //_LM_FSA_LM_HH
