/** Copyright 2024 RWTH Aachen University. All rights reserved.
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
#ifndef SEARCH_V2_HH
#define SEARCH_V2_HH

namespace Search {

enum TransitionType {
    LABEL_TO_LABEL,
    LABEL_LOOP,
    LABEL_TO_BLANK,
    BLANK_TO_LABEL,
    BLANK_LOOP,
};

}  // namespace Search

#endif  // SEARCH_V2_HH
