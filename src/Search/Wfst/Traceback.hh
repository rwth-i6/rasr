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
#ifndef _SEARCH_WFST_TRACEBACK_HH
#define _SEARCH_WFST_TRACEBACK_HH

#include <Search/Search.hh>
#include <Search/Types.hh>
#include <Search/Wfst/ExpandingFsaSearch.hh>
#include <OpenFst/Types.hh>

namespace Search { namespace Wfst {

class WordEndDetector;

class BestPath
{
protected:
    typedef OpenFst::Label Label;
    typedef SearchAlgorithm::Traceback Traceback;
public:
    typedef SearchAlgorithm::ScoreVector ScoreVector;
    struct Item {
        Label word;
        TimeframeIndex time;
        ScoreVector score;
        Item(Label w, TimeframeIndex t, Score s) :
            word(w), time(t), score(s, 0) {}
        Item(Label w, TimeframeIndex t, Score a, Score l) :
            word(w), time(t), score(a, l) {}
        Item(Label w, TimeframeIndex t, const ScoreVector &s) :
            word(w), time(t), score(s) {}
    };
    struct CompareTime
    {
        bool operator()(const Item &a, const Item &b) const {
            return a.time < b.time;
        }
    };

protected:
    typedef std::vector<Item> Path;
public:
    typedef Path::iterator Iterator;
    typedef Path::const_iterator ConstIterator;

    void append(Label word, TimeframeIndex time, Score score) {
        path_.push_back(Item(word, time, score));
    }
    void append(const Item &item) {
        path_.push_back(item);
    }

    void clear() { path_.clear(); }
    bool empty() const { return path_.empty(); }
    ConstIterator begin() const { return path_.begin(); }
    ConstIterator end() const { return path_.end(); }
    Iterator begin() { return path_.begin(); }
    Iterator end() { return path_.end(); }

    void getTraceback(Bliss::LexiconRef lexicon, OutputType outputType,
                      const OpenFst::LabelMap *olabelMap, Traceback *result) const;

protected:
    std::vector<Item> path_;
};

} // namespace Wfst
} // namespace Search

#endif  // _SEARCH_WFST_TRACEBACK_HH
