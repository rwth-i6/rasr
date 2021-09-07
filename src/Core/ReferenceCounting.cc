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
#include "ReferenceCounting.hh"
#include "Hash.hh"
#include "Utility.hh"

using namespace Core;

namespace Core {

ReferenceCounted::~ReferenceCounted() {}

void ReferenceCounted::acquireWeakReference(WeakRefBase* reference) const {
    // require(*reference == this);
    if (!isSentinel(this)) {
        if (not weak_refs_) {
            weak_refs_.reset(new WeakRefSet());
        }
        weak_refs_->insert(reference);
    }
}

void ReferenceCounted::releaseWeakReference(WeakRefBase* reference) const {
    // require(*reference == this);
    if (!isSentinel(this)) {
        verify_(weak_refs_);
        weak_refs_->erase(reference);
    }
}

void ReferenceCounted::free() const {
    require_(!referenceCount_);
    verify_(!isSentinel(this));
    if (weak_refs_) {
        for (auto iter = weak_refs_->begin(); iter != weak_refs_->end(); ++iter) {
            (*iter)->invalidate();
        }
    }
    delete this;
}

}  // namespace Core
