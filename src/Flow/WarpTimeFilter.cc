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
#include "WarpTimeFilter.hh"

using namespace Flow;

const Core::ParameterFloat WarpTimeFilterNode::paramStartTime(
        "start-time", "segment start time.", 0);

WarpTimeFilterNode::WarpTimeFilterNode(const Core::Configuration& c)
        : Core::Component(c),
          Node(c),
          currentTime_(Core::Type<Flow::Time>::max) {
    addInputs(1);
    addOutputs(1);
}

bool WarpTimeFilterNode::setParameter(const std::string& name, const std::string& value) {
    if (paramStartTime.match(name)) {
        if (!warping_.empty()) {
            warning() << "time warping list was nonempty while setting " << name << ", warping-map is discarded!";
            warping_.clear();
        }
        currentTime_ = paramStartTime(value);
        return true;
    }
    else
        return Flow::AbstractNode::setParameter(name, value);
}

bool WarpTimeFilterNode::configure() {
    return putOutputAttributes(0, getInputAttributes(0));
}

bool WarpTimeFilterNode::work(PortId p) {
    DataPtr<Timestamp> in;
    while (getData(0, in)) {
        in = DataPtr<Timestamp>(dynamic_cast<Timestamp*>(in->clone()));
        verify(currentTime_ != Core::Type<Flow::Time>::max);
        verify(in->startTime() >= currentTime_);

        Time offset = currentTime_ - in->startTime();

        if (warping_.empty() || offset != warping_.back().first - warping_.back().second)
            warping_.push_back(std::make_pair(in->startTime() + offset, in->startTime()));

        in->setStartTime(in->startTime() + offset);
        in->setEndTime(in->endTime() + offset);

        currentTime_ = in->endTime();
        return putData(0, in.get());
    }

    if (in.get() == Flow::Data::ood()) {
        return putOod(p);
    }
    else if (in.get() == Flow::Data::eos() && warping_.size()) {
        Core::Component::Message msg = log();
        msg << "warping map:";
        for (std::vector<std::pair<Time, Time>>::iterator it = warping_.begin(); it != warping_.end(); ++it)
            msg << " " << it->first << ":" << it->second;
        // Write the warping map
        warping_.clear();
        currentTime_ = Core::Type<Flow::Time>::max;
    }

    return putData(0, in.get());
}
