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
#ifndef _SIGNAL_FRAME_INTERPOLATION_HH
#define _SIGNAL_FRAME_INTERPOLATION_HH

#include <Flow/Synchronization.hh>

namespace Signal {

extern Core::ParameterInt paramFrameInterpolationOrder;

/** FrameInterpolationNode: creates one elements for each elements read from "target" stream
 *  by interpolation at target start-times.
 *
 *  For more details @see Flow::SynchronizationNode.
 */
template<class Algorithm>
class FrameInterpolationNode : public Flow::SynchronizationNode<Algorithm> {
private:
    typedef Flow::SynchronizationNode<Algorithm> Precursor;

public:
    FrameInterpolationNode(const Core::Configuration& c);

    virtual ~FrameInterpolationNode() {}

    bool setParameter(const std::string& name, const std::string& value);
};

template<class Algorithm>
FrameInterpolationNode<Algorithm>::FrameInterpolationNode(const Core::Configuration& c)
        : Core::Component(c), Precursor(c) {
    this->setOrder(paramFrameInterpolationOrder(c));
}

template<class Algorithm>
bool FrameInterpolationNode<Algorithm>::setParameter(const std::string& name, const std::string& value) {
    if (paramFrameInterpolationOrder.match(name))
        this->setOrder(paramFrameInterpolationOrder(value));
    else
        return Precursor::setParameter(name, value);

    return true;
}

}  // namespace Signal

#endif  // _SIGNAL_FRAME_INTERPOLATION_HH
