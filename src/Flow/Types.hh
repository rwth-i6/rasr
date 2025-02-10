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
#ifndef _FLOW_TYPES_HH
#define _FLOW_TYPES_HH

/*
 * flow network types:
 */

#include <Core/Types.hh>

namespace Flow {

typedef s32      PortId;
static const s32 IllegalPortId = -1;

/**
 * Time interval or point in time measures in seconds.
 */
typedef f64 Time;

const Time timeTolerance    = (Time)1e7;
const s32  timeToleranceUlp = 100000;

// possible output types of the node
// the lower 8 bit store the size of one sample
enum class SampleType : unsigned {
    	SampleTypeS8  = 0x0101,
    	SampleTypeU8  = 0x0201,
    	SampleTypeS16 = 0x0302,
    	SampleTypeU16 = 0x0402,
    	SampleTypeF32 = 0x0504
};

}  // namespace Flow

#endif  // _FLOW_TYPES_HH
