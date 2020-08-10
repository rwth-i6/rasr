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
#ifndef THEANOCOMMUNICATOR_HH_
#define THEANOCOMMUNICATOR_HH_

#include <memory>

#include <Core/Component.hh>
#include <Math/Matrix.hh>

#include "CorpusDescription.hh"

namespace Bliss {

/*
 * single global instance
*/
class TheanoCommunicator : public Core::Component {
private:
    static const Core::ParameterInt paramSharedMemKey;
    TheanoCommunicator(const Core::Configuration& c);

    //careful: the offset is in bytes and not related to T
    template<typename T>
    T&     shMem(u32 offset);
    u32&   shMemStatus();
    float& shMemData(u32 idx);  //here idx is in floats, not in bytes!
    u32&   shMemRows();
    u32&   shMemCols();
    float& shMemLoss();
    void   waitForStatus(u32 status);

    static std::unique_ptr<TheanoCommunicator> communicator_;
    std::string                                currentSegmentName_;  //used to cache posteriors for 1 segment
    Math::Matrix<f32>                          posteriors_;

    int   shId_;
    void* shMem_;

public:
    ~TheanoCommunicator();
    static TheanoCommunicator& communicator();
    static void                create(const Core::Configuration& c);
    bool                       waitForErrorSignalRequest(/*out*/ std::string& segmentName);
    const Math::Matrix<f32>&   getPosteriorsForSegment(const Bliss::SpeechSegment* segment);
    void                       writeErrorSignalForSegment(const Bliss::SpeechSegment* segment, float loss, const Math::Matrix<f32>& m_);
};

}  // namespace Bliss

#endif /*THEANOCOMMUNICATOR_HH_*/
