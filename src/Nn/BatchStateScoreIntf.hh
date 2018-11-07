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
#ifndef _NN_BATCHSTATESCOREINTF_HH
#define _NN_BATCHSTATESCOREINTF_HH

#include <Core/Types.hh>

namespace Nn {

/* This is somewhat like a FeatureScorer, but a better interface for our NN code.
 */
template<typename FloatT>
struct BatchStateScoreIntf {
    virtual ~BatchStateScoreIntf() {}

    virtual u32 getBatchLen() = 0;
    virtual FloatT getStateScore(u32 timeIdx, u32 emissionIdx) = 0; // -log space
};

}

#endif // BATCHSTATESCOREINTF_HH
