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
#ifndef _LATTICE_SMOOTHED_ACCURACY_HH
#define _LATTICE_SMOOTHED_ACCURACY_HH

#include "Accuracy.hh"
#include "SmoothingFunction.hh"

namespace Lattice {

/**
 *  Calculate the smoothed frame state accuracies (cf. state-based Hamming distance).
 *  @param correct represents the reference hypotheses (aka numerator lattice).
 *  The arc weights are assumed to be set to the word posteriors.
 *  @param smoothingFunction: implementation of smoothing function f.
 *  @return: word lattice with the same topology as @param lattice
 *  but with arc weights set to \sum_{t}f'(E[\chi_{spk,t}])\chi_{spk,t} where
 *  E[\chi_{spk,t}] are the accumulated posteriors of @param correct
 *  at timeframe t.
 *  Remark: The accuracies are accumulated and stored as word arc weight.
 */
ConstWordLatticeRef getSmoothedFrameStateAccuracy(ConstWordLatticeRef                                  lattice,
                                                  ConstWordLatticeRef                                  correct,
                                                  Core::Ref<Speech::PhonemeSequenceAlignmentGenerator> alignmentGenerator,
                                                  SmoothingFunction&                                   smoothingFunction);

}  // namespace Lattice

#endif  // _LATTICE_SMOOTHED_ACCURACY_HH
