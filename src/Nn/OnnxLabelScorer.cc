/* Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 * Licensed under the RWTH ASR License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * 	http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OnnxLabelScorer.hh"
// #include "Prior.hh"

using namespace Nn;

OnnxModelBase::OnnxModelBase(const Core::Configuration& config) :
	Core::Component(config),
	Precursor(config) {}

OnnxFfnnTransducer::OnnxFfnnTransducer(const Core::Configuration& config) :
	Core::Component(config),
	Precursor(config) {}

LabelHistory OnnxFfnnTransducer::startHistory() {

	return labelHistoryManager_->history(new LabelHistoryDescriptor());
}

const std::vector<Score>& OnnxFfnnTransducer::getScores(const LabelHistory& h, bool isLoop) {
	
	return inputBuffer_.at(decodeStep_);
}

/*void OnnxFfnnTransducer::cleanUpBeforeExtension(u32 minPos) {
	scoreCache_.clear();
	batchHashQueue_.clear();
	scoreTransitionCache_.clear();

	if (isPositionDependent_) {
		// cache clean up w.r.t min position among all hypotheses (otherwise memory expensive ?)
		for (std::pair<const u32, ScoreCache>& kv : positionScoreCache_)
			if (kv.first < minPos)
				kv.second.clear();
        }
}*/
