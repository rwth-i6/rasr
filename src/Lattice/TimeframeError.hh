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
#ifndef _LATTICE_TIMEFRAME_ERROR_HH
#define _LATTICE_TIMEFRAME_ERROR_HH

#include "Lattice.hh"
#include "Types.hh"

namespace Lattice {

/**
 *  Calculate the summed timeframe errors.
 *  @param total contains the total scores.
 *  @param normalization: 1 = normalization on word level
 *                        0 = normalization on timeframe level
 */
ConstWordLatticeRef getSumTimeframeErrors(
        ConstWordLatticeRef total,
        const ShortPauses&  shortPauses,
        bool                useLemmata,
        f32                 normalization);

/**
 *  Calculate the maximum timeframe errors.
 */
ConstWordLatticeRef getMaximumTimeframeErrors(
        ConstWordLatticeRef total,
        const ShortPauses&  shortPauses,
        bool                useLemmata);

/**
 *  Calcluate the word timeframe accuracy.
 */
ConstWordLatticeRef getWordTimeframeAccuracy(
        ConstWordLatticeRef lattice,
        ConstWordLatticeRef correct,
        const ShortPauses&  shortPauses,
        bool                useLemmata,
        f32                 normalization);

}  // namespace Lattice

#endif  // _LATTICE_TIMEFRAME_ERROR_HH
