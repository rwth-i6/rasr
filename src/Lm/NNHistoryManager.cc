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
#include "NNHistoryManager.hh"

namespace Lm {

HistoryHandle NNHistoryManager::acquire(HistoryHandle handle) {
    NNCacheBase* c = const_cast<NNCacheBase*>(reinterpret_cast<NNCacheBase const*>(handle));
    c->ref_count++;
    return handle;
}

void NNHistoryManager::release(HistoryHandle handle) {
    NNCacheBase* c = const_cast<NNCacheBase*>(reinterpret_cast<NNCacheBase const*>(handle));
    require_gt(c->ref_count, 0);
    c->ref_count--;
    if (c->ref_count == 0) {
        if (has_on_release_handler_) {
            on_release_handler_(handle);
        }
        auto iter = nn_caches_.find(c->history.get());
        NNCacheBase* to_delete = iter->second;
        nn_caches_.erase(iter);
        delete to_delete; // might cause cascade of more deletions, thus put it at the end after we do not need iterator anymore
    }
}

HistoryHash NNHistoryManager::hashKey(HistoryHandle handle) const {
    NNCacheBase const* c = reinterpret_cast<NNCacheBase const*>(handle);
    return token_id_sequence_hash(*(c->history));
}

bool NNHistoryManager::isEquivalent(HistoryHandle lhs, HistoryHandle rhs) const {
    return lhs == rhs;
}

std::string NNHistoryManager::format(HistoryHandle handle) const {
    NNCacheBase const* c = reinterpret_cast<NNCacheBase const*>(handle);
    std::stringstream ss;
    ss << "NNHistory{ ";
    for (auto token_id : *(c->history)) {
        ss << token_id << " ";
    }
    ss << "}";
    return ss.str();
}

} // namespace Lm

