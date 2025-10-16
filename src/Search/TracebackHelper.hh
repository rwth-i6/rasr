#include "Traceback.hh"

namespace Search {

inline Bliss::LemmaPronunciation* epsilonLemmaPronunciation() {
    return reinterpret_cast<Bliss::LemmaPronunciation*>(1);
}

inline void traceback(Core::Ref<LatticeTrace> end, Traceback& result, Core::Ref<LatticeTrace> boundary) {
    result.clear();
    for (; end && end != boundary; end = end->predecessor) {
        result.push_back(*end);
    }
    std::reverse(result.begin(), result.end());
}

class RootTraceSearcher {
public:
    RootTraceSearcher(std::vector<Core::Ref<LatticeTrace>> traces)
            : rootTrace_(0) {
        for (std::vector<Core::Ref<LatticeTrace>>::const_iterator it = traces.begin(); it != traces.end(); ++it) {
            addTrace(it->get(), 0, true);
        }

        for (std::map<LatticeTrace*, TraceDesc>::iterator it = traces_.begin(); it != traces_.end(); ++it) {
            if (it->second.length == 1) {
                // This is "the" root trace
                //require(not rootTrace_->predecessor);
                rootTrace_ = it->first;
            }
        }

        TraceDesc     desc       = traces_[rootTrace_];
        LatticeTrace* prev_trace = rootTrace_;
        while (desc.followers.size() == 1 && !desc.has_active_hyps) {  // can not be sure if current root trace still have active state
            LatticeTrace* follower = desc.followers.front();
            if (traces_[follower].has_active_hyps) {
                break;
            }
            prev_trace = rootTrace_;
            rootTrace_ = follower;
            desc       = traces_[rootTrace_];
        }

        for (LatticeTrace* follower : desc.followers) {
            // when follower is an epsilon transition, the trace is not yet stable
            if (follower->pronunciation == epsilonLemmaPronunciation()) {
                rootTrace_ = prev_trace;
                break;
            }
        }
    }

    LatticeTrace* rootTrace() const {
        return rootTrace_;
    }

private:
    int addTrace(LatticeTrace* trace, LatticeTrace* follower, bool has_active_hyps = false) {
        std::map<LatticeTrace*, TraceDesc>::iterator it = traces_.find(trace);

        if (it != traces_.end()) {
            // Already there, just add follower
            TraceDesc& desc((*it).second);
            desc.has_active_hyps |= has_active_hyps;
            if (follower) {
                desc.followers.push_back(follower);
            }
            return desc.length;
        }
        else {
            // Add the predecessors, compute the length, and add the new trace + follower
            int length = 1;
            if (trace->predecessor) {
                length += addTrace(trace->predecessor.get(), trace, false);
            }
            TraceDesc desc;
            desc.length          = length;
            desc.has_active_hyps = has_active_hyps;
            if (follower) {
                desc.followers.push_back(follower);
            }
            traces_.insert(std::make_pair(trace, desc));
            return length;
        }
    }

    struct TraceDesc {
        int                        length;
        std::vector<LatticeTrace*> followers;
        bool                       has_active_hyps;
    };

    std::map<LatticeTrace*, TraceDesc> traces_;
    LatticeTrace*                      rootTrace_;
};

}  // namespace Search
