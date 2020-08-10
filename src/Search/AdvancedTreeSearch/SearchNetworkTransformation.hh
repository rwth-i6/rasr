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
#ifndef ADVANCEDTREESEARCH_SEARCHNETWORKTRANSFORMATION_HH
#define ADVANCEDTREESEARCH_SEARCHNETWORKTRANSFORMATION_HH

/**
 * This file contains a collection of helper classes useful for transformations of the search network
 * */
#include <Search/StateTree.hh>
#include "TreeStructure.hh"

namespace AdvancedTreeSearch {
struct StateWithSuccessors {
private:
    Search::StateTree::StateDesc desc_;
    std::set<Search::StateId>    successors_;
    u32                          hash_;

    void buildHash() {
        hash_             = Search::StateTree::StateDesc::Hash()(desc_);
        u32 numSuccessors = successors_.size();
        hash_ += (numSuccessors >> 3) + (numSuccessors << 11);
        for (std::set<Search::StateId>::iterator it = successors_.begin(); it != successors_.end(); ++it)
            hash_ += ((*it) >> 11) + ((*it) << 5);
    }

public:
    StateWithSuccessors()
            : hash_(0) {
    }

    StateWithSuccessors(const Search::StateTree::StateDesc& _desc, const std::set<Search::StateId>& _successors)
            : desc_(_desc), successors_(_successors) {
        buildHash();
    }

    bool operator==(const StateWithSuccessors& rhs) const {
        return hash_ == rhs.hash_ && desc_ == rhs.desc_ && successors_ == rhs.successors_;
    }

    struct Hash {
        size_t operator()(const StateWithSuccessors& s) {
            return s.hash_;
        }
    };
};

typedef std::unordered_map<StateWithSuccessors, Search::StateId, StateWithSuccessors::Hash> SuffixStructure;
}  // namespace AdvancedTreeSearch

#endif  // ADVANCEDTREESEARCH_SEARCHNETWORKTRANSFORMATION_HH
