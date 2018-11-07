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
#include <Search/Wfst/BookKeeping.hh>
#include <Search/Wfst/Traceback.hh>
#include <Search/Wfst/WordEnd.hh>
#include <stack>

using namespace Search::Wfst;
using OpenFst::Epsilon;

void FirstBestTraceRecorder::clear()
{
    data_.clear();
    next_ = 0;
}

FirstBestTraceRecorder::TraceRef FirstBestTraceRecorder::addTrace(
    TraceRef sibling, TraceRef predecessor, Label output,
    const StateSequence *input, TimeframeIndex time, Score score, Score arcScore, bool wordEnd)
{
    if (next_ >= data_.size()) {
        enlarge();
    }
    TraceRef ref = next_;
    Trace& trace = data_[ref];
    verify_(!trace.active);
    next_ = trace.predecessor;
    verify_(next_ == data_.size() || !data_[next_].active);
    trace.predecessor = predecessor;
    trace.sibling = InvalidTraceRef;
    trace.output = output;
    trace.input = input;
    trace.time = time;
    trace.score = score;
    trace.wordEnd = wordEnd;
    trace.used = true;
    return ref;
}

void FirstBestTraceRecorder::updateTrace(TraceRef sibling, TraceRef predecessor, Score score)
{
    Trace &trace = data_[sibling];
    if (score < trace.score) {
        trace.score = score;
        trace.predecessor = predecessor;
    }
}

void FirstBestTraceRecorder::updateTime(TraceRef t, TimeframeIndex time)
{
    data_[t].time = time;
}

void FirstBestTraceRecorder::purgeBegin()
{
    for (TraceArray::iterator i = data_.begin(); i != data_.end(); ++i)
        i->active = false;
}

void FirstBestTraceRecorder::purgeNotify(TraceRef trace)
{
    if (createLattice_)
        purgeNotifyDfs(trace);
    else
        purgeNotifyLinear(trace);
}

void FirstBestTraceRecorder::purgeNotifyLinear(TraceRef trace)
{
    while (trace != InvalidTraceRef) {
        verify_(data_[trace].used);
        if (data_[trace].active)
            break; // avoid going path twice
        data_[trace].active = true;
        trace = data_[trace].predecessor;
    }
}

void FirstBestTraceRecorder::purgeNotifyDfs(TraceRef trace)
{
    std::stack<TraceRef> stack;
    stack.push(trace);
    while (!stack.empty()) {
        TraceRef r = stack.top(); stack.pop();
        Trace &t = data_[r];
        if (!t.active) {
            t.active = true;
            if (t.predecessor != InvalidTraceRef)
                stack.push(t.predecessor);
            if (t.sibling != InvalidTraceRef)
                stack.push(t.sibling);
        }
    }
}

void FirstBestTraceRecorder::purgeEnd()
{
    for (TraceRef i = 0; i < data_.size(); ++i) {
        Trace &item = data_[i];
        if (item.used && !item.active) {
            item.predecessor = next_;
            item.used = false;
            next_ = i;
        }
    }
}

void FirstBestTraceRecorder::enlarge()
{
    u32 oldSize = data_.size();
    u32 newSize = oldSize + incrementSize;
    data_.resize(newSize);
    for (u32 i = oldSize; i < newSize;) {
        Trace &item = data_[i];
        item.predecessor = ++i;
    }
}

bool FirstBestTraceRecorder::hasWordEndTime(const WordEndDetector& wordEnds, TraceRef end)
{
    u32 nOutput = 0, nTime = 0;
    TraceRef tr = end;
    while (tr != TraceRecorder::InvalidTraceRef) {
        const Trace &trace = data_[tr];
        if (wordEnds.isWordEnd(trace))
            ++nTime;
        if (trace.output != OpenFst::Epsilon)
            ++nOutput;
        tr = trace.predecessor;
    }
    return (nTime >= nOutput);
}

void FirstBestTraceRecorder::createBestPath(
        const WordEndDetector &wordEnds, bool ignoreLast, TraceRef end, BestPath *path)
{
    path->clear();
    // insert trace back item for sentence end
    const Trace &last = data_[end];
    path->append(Epsilon, last.time, last.score);
    // std::cout << "best: " << book[trace].score << " " << book[trace].score << std::endl;
    std::deque<Label> outputs;
    std::deque<BestPath::Item> timeAndScore;
    bool ignoreOutput = ignoreLast;
    for (TraceRef trace = end; trace != InvalidTraceRef; trace = data_[trace].predecessor) {
        const Trace &currentTrace = data_[trace];
        if (wordEnds.isNonWord(currentTrace.input)) {
            path->append(Epsilon, currentTrace.time, currentTrace.score);
        } else if (wordEnds.isWordEnd(currentTrace))
            timeAndScore.push_back(BestPath::Item(Epsilon, currentTrace.time, currentTrace.score));
        if (currentTrace.output != Epsilon) {
            if (!ignoreOutput)
                outputs.push_back(currentTrace.output);
            else
                ignoreOutput = false;
        }
        if (!outputs.empty() && !timeAndScore.empty()) {
            BestPath::Item item = timeAndScore.front();
            item.word = outputs.front();
            path->append(item);
            outputs.pop_front();
            timeAndScore.pop_front();
        }
    }
    std::reverse(path->begin(), path->end());
    std::sort(path->begin(), path->end(), BestPath::CompareTime());
}

Search::Wfst::Lattice* FirstBestTraceRecorder::createLattice(TraceRef end)
{
    defect();
    return 0;
}
