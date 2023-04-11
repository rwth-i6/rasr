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

#ifndef LABEL_SCORER_HH
#define LABEL_SCORER_HH

#include <Core/Component.hh>
#include <Core/ReferenceCounting.hh>

namespace Nn {


// base class of models for label scoring (basic supports except scoring)
class LabelScorer : public virtual Core::Component,
                    public Core::ReferenceCounted {
};

}  // namespace Nn

#endif
