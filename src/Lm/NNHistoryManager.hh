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
#ifndef _LM_NN_HISTORY_MANAGER_HH
#define _LM_NN_HISTORY_MANAGER_HH

#include <functional>

#include <Bliss/Symbol.hh>
#include <Core/MurmurHash.hh>
#include "HistoryManager.hh"

namespace Lm {

typedef std::vector<Bliss::Token::Id> TokenIdSequence;

inline size_t token_id_sequence_hash(TokenIdSequence const& ts) {
    return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(ts.data()), ts.size() * sizeof(TokenIdSequence::value_type), 0x78b174eb);
}

inline size_t token_id_sequence_ptr_eq(TokenIdSequence const* lhs, TokenIdSequence const* rhs) {
    if (lhs == rhs) {
        return true;
    }
    if (lhs->size() != rhs->size()) {
        return false;
    }
    for (size_t i = 0; i < lhs->size(); i++) {
        if ((*lhs)[i] != (*rhs)[i]) {
            return false;
        }
    }
    return true;
}

struct TokenIdSequencePtrHash {
    size_t operator()(TokenIdSequence const* token_seq) const {
        return token_id_sequence_hash(*token_seq);
    }
};

struct TokenIdSequencePtrEq {
    bool operator()(TokenIdSequence const* lhs, TokenIdSequence const* rhs) const {
        return token_id_sequence_ptr_eq(lhs, rhs);
    }
};

struct NNCacheBase {
    virtual ~NNCacheBase() = default;

    size_t                                 ref_count;
    std::unique_ptr<const TokenIdSequence> history;
};

class NNHistoryManager : public HistoryManager {
public:
    typedef std::function<void(HistoryHandle)>                                                                     OnReleaseHandler;
    typedef std::unordered_map<TokenIdSequence const*, NNCacheBase*, TokenIdSequencePtrHash, TokenIdSequencePtrEq> NNCacheMap;
    typedef std::function<void(HistoryHandle)>                                                                     VisitorFun;

    NNHistoryManager();
    virtual ~NNHistoryManager();

    template<typename NNCache>
    HistoryHandle     get(TokenIdSequence const& hist);
    void              setOnReleaseHandler(OnReleaseHandler const& handler);
    NNCacheMap const& getNNCacheMap() const;
    void              visit(VisitorFun f) const;

    // implement HistoryManager interface
    virtual HistoryHandle acquire(HistoryHandle handle);
    virtual void          release(HistoryHandle handle);
    virtual HistoryHash   hashKey(HistoryHandle handle) const;
    virtual bool          isEquivalent(HistoryHandle lhs, HistoryHandle rhs) const;
    virtual std::string   format(HistoryHandle handle) const;

private:
    NNCacheMap nn_caches_;

    bool             has_on_release_handler_;
    OnReleaseHandler on_release_handler_;
};

// inline implementations

inline NNHistoryManager::NNHistoryManager()
        : HistoryManager(), has_on_release_handler_(false), on_release_handler_() {
}

inline NNHistoryManager::~NNHistoryManager() {
    for (auto& keyval : nn_caches_) {
        delete keyval.second;
    }
    nn_caches_.clear();
}

template<typename NNCache>
inline HistoryHandle NNHistoryManager::get(TokenIdSequence const& hist) {
    auto iter = nn_caches_.find(&hist);
    if (iter == nn_caches_.end()) {
        NNCacheBase* c = new NNCache();
        c->ref_count   = 0ul;
        c->history     = std::unique_ptr<TokenIdSequence>(new TokenIdSequence(hist));
        auto r         = nn_caches_.insert(std::make_pair(c->history.get(), c));
        iter           = r.first;
    }
    return iter->second;
}

inline void NNHistoryManager::setOnReleaseHandler(OnReleaseHandler const& handler) {
    has_on_release_handler_ = true;
    on_release_handler_     = handler;
}

inline NNHistoryManager::NNCacheMap const& NNHistoryManager::getNNCacheMap() const {
    return nn_caches_;
}

}  // namespace Lm

#endif /* _LM_NN_HISTORY_MANAGER_HH */
