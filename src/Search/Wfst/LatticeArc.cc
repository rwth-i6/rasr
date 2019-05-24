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
#include <Search/Wfst/Lattice.hh>
#include <fst/register.h>
#include <fst/script/fstscript.h>
#include <fst/script/register.h>

using namespace fst;
using namespace fst::script;
using namespace Search::Wfst;

REGISTER_FST(VectorFst, LatticeArc);
REGISTER_FST_CLASSES(LatticeArc);
REGISTER_FST_OPERATIONS(LatticeArc);
