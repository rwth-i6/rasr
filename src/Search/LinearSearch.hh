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
// $Id: LinearSearch.hh 2006-10-09 18:14:21Z nolden $

#ifndef _SPEECH_LINEAR_SEARCH_HH
#define _SPEECH_LINEAR_SEARCH_HH

#include <Core/Component.hh>
#include <Core/ReferenceCounting.hh>
#include <Core/Parameter.hh>
#include <Signal/SlidingWindow.hh>
#include <Lm/LanguageModel.hh>
#include <Speech/ModelCombination.hh>
#include "Search.hh"
#include "LanguageModelLookahead.hh"

namespace Search {
    class WordPronunciationState; ///The pronunciation including its current state.
    typedef Core::Ref<WordPronunciationState> WordPronunciationStatePointer;

    struct LinearSearchHistoryData {
                LinearSearchHistoryData();
                ~LinearSearchHistoryData();
                std::list<WordPronunciationStatePointer> states;
                bool success;
    };

    class LinearSearch : public Search::SearchAlgorithm {
    public:

                ////////////////////////////////////////////////////////////////////////////////
                // embedded class pronunciation
                ////////////////////////////////////////////////////////////////////////////////
                class Pronunciation {
                public:
                        struct MixtureItem {
                                Mm::MixtureIndex mixture;
                                const Am::StateTransitionModel* stateTransitionModel;
                        };
                        typedef std::vector<MixtureItem> MixtureVector;

                        const Bliss::LemmaPronunciation* const lemma() {
                                return lemma_;
                        }

                        Pronunciation( const Bliss::LemmaPronunciation *const lemma, const Am::AcousticModel* acousticModel );

                        size_t nMixtures() {
                                return mixtures_.size();
                        }

                        MixtureVector& mixtures() {
                                return mixtures_;
                        }

                        inline bool isRegularWord() const {
                                return isRegularWord_;
                        }
                private:
                        bool isRegularWordPrivate() const;

                        bool isRegularWord_;
                        const Bliss::LemmaPronunciation* const lemma_;
                        MixtureVector mixtures_;
                };
                // End nested class

                typedef std::vector<WordPronunciationStatePointer> WordPronunciationStateVector;

                LinearSearch( const Core::Configuration&);

                virtual ~LinearSearch();

                virtual bool setModelCombination(const Speech::ModelCombination &modelCombination);

                virtual void setGrammar( Fsa::ConstAutomatonRef );

                virtual void restart();
                virtual void feed( const Mm::FeatureScorer::Scorer& );
                virtual void getPartialSentence( Traceback &result );
                virtual void getCurrentBestSentence( Traceback &result ) const;
                virtual Core::Ref<const LatticeAdaptor> getCurrentWordLattice() const;
                virtual void resetStatistics();
                virtual void logStatistics() const;
        private:
                friend struct LinearSearchHistoryData;
                friend class WordPronunciationState;
                struct Book {
                        Book();
                        ~Book();
                        Score score; ///This does not include lmScore
                        Score lmScore;
                        bool hadRegularWord;
                        WordPronunciationStatePointer word;
                        Book* bkp;
                        TimeframeIndex time;
                };

                bool isRegularWord( const Pronunciation* pron );

                void bookKeeping( Book& book, bool irregular = false );

                Bliss::LexiconRef lexicon_;
                const Bliss::Lemma *silence_;
                Core::Ref<const Am::AcousticModel> acousticModel_;
                Core::Ref<const Lm::ScaledLanguageModel> lm_;
                const LanguageModelLookahead *lmLookahead_;
                Score pronunciationScale_; // CHANGED from f32
                bool singleWordRecognition_;
                static const Core::ParameterBool paramSingleWordRecognition_;

                template<class IteratorType>
                LinearSearchHistoryData addPronunciations( std::pair<IteratorType, IteratorType> pronunciations );
                bool removePronunciations( const LinearSearchHistoryData& data );
                ///For now this always returns true, because only unigrams are completely supported
                bool isUnigram() const;

                typedef std::deque<Book> BookVector;
                WordPronunciationStateVector  state_;
                TimeframeIndex time_;
                BookVector book_;
                BookVector irregularBook_; //For single-word recognition it is necessary to track the probability of the whole input being non-regular words(silence)
    };
}

#endif
