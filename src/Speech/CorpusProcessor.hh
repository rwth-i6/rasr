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
#ifndef _SPEECH_CORPUS_PROCESSOR_HH
#define _SPEECH_CORPUS_PROCESSOR_HH

#include <Bliss/CorpusDescription.hh>
#include <Core/Statistics.hh>
#include "CorpusVisitor.hh"

namespace Speech {

class CorpusVisitor;

/**
 * CorpusProcessor base class for algorithms driven by a CorpusVisitor
 *
 * Output (XML format):
 * - CPU time and real time factor (channel: real-time-factor)
 */
class CorpusProcessor : public virtual Core::Component {
protected:
    Core::XmlChannel channelTimer_;
    Core::Timer      timer_;
    void             reportRealTime(Flow::Time);

public:
    CorpusProcessor(const Core::Configuration& c);
    virtual ~CorpusProcessor();

    /** Override this function to sign on to services of the corpus visitor.
     *  Note: call the signOn function of your predecessor.
     */
    virtual void signOn(CorpusVisitor& corpusVisitor);

    virtual void enterCorpus(Bliss::Corpus* corpus);
    virtual void leaveCorpus(Bliss::Corpus* corpus);
    virtual void enterRecording(Bliss::Recording* recoding);
    virtual void leaveRecording(Bliss::Recording* recoding);
    virtual void enterSegment(Bliss::Segment* segment);
    virtual void processSegment(Bliss::Segment* segment);
    virtual void leaveSegment(Bliss::Segment* segment);
    virtual void enterSpeechSegment(Bliss::SpeechSegment*);
    virtual void processSpeechSegment(Bliss::SpeechSegment* segment);
    virtual void leaveSpeechSegment(Bliss::SpeechSegment*);
};

}  // namespace Speech

#endif  // _SPEECH_CORPUS_PROCESSOR_HH
