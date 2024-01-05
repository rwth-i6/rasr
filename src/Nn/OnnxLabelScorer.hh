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

#ifndef ONNX_LABEL_SCORER_HH
#define ONNX_LABEL_SCORER_HH

#include <Onnx/IOSpecification.hh>
#include <Onnx/Session.hh>
//#include <Onnx/Value.hh>
#include "LabelScorer.hh"

namespace Nn {

struct OnnxLabelHistory : public LabelHistoryBase {
	std::vector<Score> scores;
	// TensorList variables;
	u32 position;
	bool isBlank; // for next feedback

        typedef LabelHistoryBase Precursor;

        OnnxLabelHistory() : Precursor(), position(0), isBlank(false) {}
        /*TFLabelHistory(const TFLabelHistory& ref) :
		Precursor(ref), scores(ref.scores), variables(ref.variables),
		position(ref.position), isBlank(ref.isBlank) {}*/
};


// Encoder-Decoder Label Scorer based on Onnx back-end
// computation logics based on a predefined order of I/O and op collections in graph ????
// prerequisite: model graph compilation that parse the model into these collections ????
class OnnxModelBase : public LabelScorer {

	typedef LabelScorer Precursor;

	public:
        // config params for graph computation
        static const Core::ParameterBool paramTransformOuputLog;
        static const Core::ParameterBool paramTransformOuputNegate;
        static const Core::ParameterInt paramMaxBatchSize;

		// overwrite descriptor in derived class for specific history
		typedef OnnxLabelHistory LabelHistoryDescriptor;

    public:
		OnnxModelBase(const Core::Configuration& config);
		virtual ~OnnxModelBase();

        virtual void reset();
        //virtual void cleanUpBeforeExtension(u32 minPos) { cacheHashQueue_.clear(); }

		// history handling
		virtual LabelHistory startHistory() = 0;
		virtual void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop) = 0;

		// get scores for the next output position
		virtual const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop) = 0;

    protected:
        void init();

        bool debug_;

	protected:
        // onnx related members
        Onnx::Session                                         session_;
        static const std::vector<Onnx::IOSpecification>       ioSpec_;  // currently fixed to "features", "feature-size" and "output"
        const Onnx::IOMapping                                 mapping_;
        Onnx::IOValidator                                     validator_;

        // session-run related members
        const std::string              features_onnx_name_;
        const std::string              features_size_onnx_name_;
        const std::vector<std::string> output_onnx_names_;


        // binary function including scaling
        std::function<Score(Score, Score)> decoding_output_transform_function_;

    protected:
        //LabelHistoryDescriptor* startHistoryDescriptor_; // only common stuff, no states or scores

        std::deque<size_t> cacheHashQueue_;
        u32 maxBatchSize_;

        //typedef std::unordered_map<size_t, std::vector<Score>> ScoreCache;

};

class OnnxFfnnTransducer : public OnnxModelBase {
	typedef OnnxModelBase Precursor;
	public:
		OnnxFfnnTransducer(const Core::Configuration& config);
		~OnnxFfnnTransducer() {};

		// history handling
		LabelHistory startHistory();
		void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop) {};

		// get scores for the next output position
		const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop);

	/*	void cleanUpBeforeExtension(u32 minPos);
	private:
		// context (and position) dependent cache: central handling of scores instead of each history
		ScoreCache ScoreCache_;
		std::unordered_set<size_t> batchHashQueue_;

		// HMM topology differs w.r.t. loopUpdateHistory_, if true then
		// - alignment sequence dependency (otherwise output/segment label sequence)
		// - loop scoring based on previous frame labels (otherwise segment labels)
		ScoreCache scoreTransitionCache_;

		// for segmental decoding {position: {context: scores}}
		std::unordered_map<u32, ScoreCache> positionScoreCache_;*/
};

} // namespace

#endif
