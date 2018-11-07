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
#ifndef _BLISS_THEANO_SEGMENT_ORDERING_VISITOR_HH
#define _BLISS_THEANO_SEGMENT_ORDERING_VISITOR_HH

#include "CorpusDescription.hh"
#include "SegmentOrdering.hh"

namespace Bliss {

/**
 * Changes the order of processed segments according to the order given by Theano
 * note that each segment can also be visited multiple times
 */
class TheanoSegmentOrderingVisitor : public Bliss::SegmentOrderingVisitor
{
public:
        virtual ~TheanoSegmentOrderingVisitor();
        virtual SegmentOrderingVisitor* copy();

        virtual void leaveCorpus(Bliss::Corpus *corpus);

        virtual void setAutoShuffle(bool enabled);
        virtual void setSegmentList(const std::string &filename);
};


} // namespace Bliss

#endif // _BLISS_THEANO_SEGMENT_ORDERING_VISITOR_HH
