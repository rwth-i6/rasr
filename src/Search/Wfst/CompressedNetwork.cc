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
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <Fsa/Stack.hh>
#include <Search/Wfst/CompressedNetwork.hh>

using namespace Search::Wfst;
using Search::Score;

enum AutomatonType { AutomatonTypeFsa, AutomatonTypeFst };

template<class Automaton>
class CompressedNetwork::Builder
{
public:
    typedef u32 StateIndex;
private:
    States &states_;
    Arcs &arcs_;
    EpsilonArcs &epsilonArcs_;
    CompressedNetwork *network_;
    const Automaton &fsa_;
    std::vector<StateIndex> stateIndex_;
    static const StateIndex InvalidStateIndex;
    bool removeEpsArcs_;
    size_t curStates_, curArcs_, curEpsArcs_;

public:
    Builder(CompressedNetwork *network, const Automaton &fsa, bool removeEpsArcs) :
        states_(network->states_),
        arcs_(network->arcs_),
        epsilonArcs_(network->epsilonArcs_),
        network_(network),
        fsa_(fsa),
        removeEpsArcs_(removeEpsArcs) {}


private:
    State& getState(typename Automaton::StateId stateId)
    {
        StateIndex idx = getStateIndex(stateId);
        return states_[idx];
    }

    StateIndex getStateIndex(typename Automaton::StateId stateId)
    {
        StateIndex idx = stateIndex_[stateId];
        if (idx == InvalidStateIndex) {
            idx = stateIndex_[stateId] = curStates_;
            new(states_ + curStates_) State(fsa_.isFinal(stateId), fsa_.finalWeightValue(stateId));
            ++curStates_;
        }
        return idx;
    }

    void createArcs(typename Automaton::StateId stateId, Fsa::Stack<typename Automaton::StateId> &statesToExplore)
    {
        /**! @todo check: do not expand any epsilon arc */
        Fsa::Stack<EpsilonArc> arcsToExplore;

        StateIndex stateIndex = getStateIndex(stateId);
        states_[stateIndex].begin = curArcs_;
        states_[stateIndex].epsilonArcsBegin = curEpsArcs_;

        for (typename Automaton::ArcIterator a = fsa_.arcs(stateId); !a.done(); a.next()) {
            const typename Automaton::Arc &arc = a.value();
            verify(fsa_.arcInput(arc) <= Core::Type<InternalLabel>::max);
            verify(fsa_.arcOutput(arc) <= Core::Type<InternalLabel>::max);
            if (fsa_.arcInput(arc) == OpenFst::Epsilon) {
                if (removeEpsArcs_ && (fsa_.arcOutput(arc) == OpenFst::Epsilon && !fsa_.isFinal(fsa_.arcTarget(arc)))) {
                    arcsToExplore.push(EpsilonArc(fsa_.arcTarget(arc), fsa_.arcOutput(arc), fsa_.arcWeightValue(arc)));
                } else {
                    new(epsilonArcs_ + curEpsArcs_) EpsilonArc(getStateIndex(fsa_.arcTarget(arc)), fsa_.arcOutput(arc), fsa_.arcWeightValue(arc));
                    ++curEpsArcs_;
                    statesToExplore.push(fsa_.arcTarget(arc));
                }
            } else {
                new(arcs_ + curArcs_) Arc(getStateIndex(fsa_.arcTarget(arc)), fsa_.arcInput(arc), fsa_.arcOutput(arc), fsa_.arcWeightValue(arc));
                ++curArcs_;
                statesToExplore.push(fsa_.arcTarget(arc));
            }
        }
        while (!arcsToExplore.empty()) {
            EpsilonArc epsArc = arcsToExplore.pop();
            for (typename Automaton::ArcIterator a = fsa_.arcs(epsArc.nextstate); !a.done(); a.next()) {
                const typename Automaton::Arc &arc = a.value();
                Score weight = epsArc.weight + Score(fsa_.arcWeightValue(arc));
                if (fsa_.arcInput(arc) == OpenFst::Epsilon) {
                    if (fsa_.arcOutput(arc) == OpenFst::Epsilon && !fsa_.isFinal(fsa_.arcTarget(arc)))
                        arcsToExplore.push(EpsilonArc(fsa_.arcTarget(arc), OpenFst::Epsilon, weight));
                    else {
                        // verify(epsArc.output == Epsilon);
                        new(epsilonArcs_ + curEpsArcs_) EpsilonArc(getStateIndex(fsa_.arcTarget(arc)), fsa_.arcOutput(arc), weight);
                        ++curEpsArcs_;
                        statesToExplore.push(fsa_.arcTarget(arc));
                    }
                } else {
                    typename Automaton::Label output = epsArc.olabel;
                    if (fsa_.arcOutput(arc) != OpenFst::Epsilon) {
                        // verify(epsArc.output == Epsilon);
                        output = fsa_.arcOutput(arc);
                    }
                    // const StateSequence *hmm = &stateSequences_[fsa_.arcInput(arc)];
                    new(arcs_ + curArcs_) Arc(getStateIndex(fsa_.arcTarget(arc)), fsa_.arcInput(arc), output, weight);
                    ++curArcs_;
                    statesToExplore.push(fsa_.arcTarget(arc));
                }
            }
        }
        u32 nArcs = curArcs_ - states_[stateIndex].begin;
        u32 nEpsArcs = curEpsArcs_ - states_[stateIndex].epsilonArcsBegin;
        verify(nArcs <= Core::Type<ArcCount>::max);
        verify(nEpsArcs <= Core::Type<EpsArcCount>::max);
        states_[stateIndex].nArcs = nArcs;
        states_[stateIndex].nEpsilonArcs = nEpsArcs;
    }

public:
    void createNetwork()
    {
        stateIndex_.resize(fsa_.nStates(), InvalidStateIndex);
        network_->nStates_ = fsa_.nStates();
        network_->nArcs_ = fsa_.nArcs();
        network_->nEpsilonArcs_ = fsa_.nEpsilonArcs();
        states_ = static_cast<State*>(malloc(network_->nStates_ * sizeof(State)));
        arcs_ = static_cast<Arc*>(malloc(network_->nArcs_ * sizeof(Arc)));
        epsilonArcs_ = static_cast<EpsilonArc*>(malloc(network_->nEpsilonArcs_ * sizeof(EpsilonArc)));
        curStates_ = curArcs_ = curEpsArcs_ = 0;
        typename Automaton::StateId initial = fsa_.initialStateId();
        // verify(initial != Fsa::InvalidStateId);
        Fsa::Stack<typename Automaton::StateId> statesToExplore;
        statesToExplore.push(initial);
        while (!statesToExplore.empty()) {
            typename Automaton::StateId s = statesToExplore.pop();
            State &state = getState(s);
            if (state.begin != InvalidArcIndex) {
                // state already expanded
                continue;
            }
            createArcs(s, statesToExplore);
        }
        verify(curStates_ == network_->nStates());
        verify(curArcs_ == network_->nArcs());
        verify(curEpsArcs_ = network_->nEpsilonArcs());
        network_->initialStateIndex_ = stateIndex_[initial];
    }
};

template<class A>
const typename CompressedNetwork::Builder<A>::StateIndex
    CompressedNetwork::Builder<A>::InvalidStateIndex = Core::Type<StateIndex>::max;

/******************************************************************************/

struct CompressedNetwork::ImageHeader
{
    typedef u64 Offset; /* file size may exceed 4G */

    char magic[8];
    u32  version;
    u32  initialStateIndex;
    u32  nStates;
    u32  nArcs;
    u32  nEpsilonArcs;
    Offset  statesOffset;
    Offset  arcsOffset;
    Offset epsArcsOffset;
    Offset  end;

    static const char *magicWord;
    static const u32 formatVersion;
};

const char *CompressedNetwork::ImageHeader::magicWord = "RWTH_NWF";
const u32   CompressedNetwork::ImageHeader::formatVersion = 4;

/******************************************************************************/

const Core::Choice CompressedNetwork::choiceAutomatonType(
        "fsa", AutomatonTypeFsa,
        "fst", AutomatonTypeFst,
        Core::Choice::endMark());
const Core::ParameterChoice CompressedNetwork::paramAutomatonType_(
        "automaton-type", &choiceAutomatonType, "type of network", AutomatonTypeFst);
const Core::ParameterString CompressedNetwork::paramNetworkFile_(
        "network-file", "search network to load", "");

const CompressedNetwork::ArcIndex CompressedNetwork::InvalidArcIndex = Core::Type<CompressedNetwork::ArcIndex>::max;
const Score CompressedNetwork::NonFinalWeight = static_cast<Score>(0xffffffff);

CompressedNetwork::CompressedNetwork(const Core::Configuration &c, bool loadNetwork) :
    Core::Component(c), states_(0), arcs_(0), epsilonArcs_(0), mmap_(0),
    mmapSize_(0), loadNetwork_(loadNetwork)
{
}

CompressedNetwork::~CompressedNetwork()
{
    if (mmap_) {
        munmap(mmap_, mmapSize_);
    } else {
        ::free(states_);
        ::free(arcs_);
        ::free(epsilonArcs_);
    }
}
bool CompressedNetwork::init() {
    std::string networkFile = paramNetworkFile_(config);
    if (loadNetwork_) {
        if (!read(networkFile)) {
            error("cannot load network file %s", networkFile.c_str());
            return false;
        }
    }
    return true;
}


bool CompressedNetwork::build(Fsa::ConstAutomatonRef f, bool removeEpsArcs)
{
    if (!f->hasProperty(Fsa::PropertySortedByInput)) {
        warning("input automaton to FSA search is not sorted by input.");
        return false;
    }
    const Fsa::StaticAutomaton *ptr = dynamic_cast<const Fsa::StaticAutomaton*>(f.get());
    ensure(ptr);
    Builder<FsaAutomatonAdapter> builder(this, ptr, removeEpsArcs);
    builder.createNetwork();
    return true;
}

bool CompressedNetwork::build(const OpenFst::VectorFst *f, bool removeEpsArcs)
{
    if (!f->Properties(FstLib::kILabelSorted, false)) {
        warning("input automaton to FSA search is not sorted by input.");
        return false;
    }
    Builder<FstAutomatonAdapter> builder(this, f, removeEpsArcs);
    builder.createNetwork();
    return true;
}

bool CompressedNetwork::write(const std::string &file) const
{
    int fd = open(file.c_str(),
                  O_CREAT|O_WRONLY|O_TRUNC,
                  S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    if (fd == -1) {
        warning("failed to open file '%s' for writing", file.c_str());
        return false;
    }
    int r = writeData(fd);
    if (r != 0) {
        warning("failed to write network '%s'", file.c_str());
    }
    close(fd);
    log("wrote %d states, %d arcs, %d epsilon arcs", nStates(), nArcs(), nEpsilonArcs());
    return (r == 0);
}

namespace {
template<class T> bool writeArray(int fd, u64 &offset, T *data, size_t nElements)
{
    off_t pos, pad;
    ssize_t nBytes;
    if ((pos = lseek(fd, 0, SEEK_CUR)) == (off_t) -1) return false;
    pad = (pos + 7) % 8;
    if ((pos = lseek(fd, pad, SEEK_CUR)) == (off_t) -1) return false;
    offset = pos;
    nBytes = sizeof(T) * nElements;
    if (write(fd, data, nBytes) != nBytes) return false;
    return true;
}
}

u32 CompressedNetwork::writeData(int fd) const
{
    off_t pos;
    ssize_t nBytes;
    ImageHeader header;

    // write header
    memcpy(header.magic, ImageHeader::magicWord, 8);
    header.version = ImageHeader::formatVersion;
    header.initialStateIndex = initialStateIndex_;
    header.nStates = nStates();
    header.nArcs = nArcs();
    header.nEpsilonArcs = nEpsilonArcs();
    header.statesOffset = 0;
    header.arcsOffset = 0;
    header.epsArcsOffset = 0;
    header.end = 0;
    nBytes = sizeof(ImageHeader);
    if (::write(fd, &header, nBytes) != nBytes) return 1;

    // write arrays
    if (!writeArray(fd, header.statesOffset, states_, nStates())) return 2;
    if (!writeArray(fd, header.arcsOffset, arcs_, nArcs())) return 3;
    if (!writeArray(fd, header.epsArcsOffset, epsilonArcs_, nArcs())) return 4;

    // determine file size
    if ((pos = lseek(fd, 0, SEEK_CUR)) == (off_t) -1) return 5;
    header.end = pos;

    // write header with offsets
    if ((pos = lseek(fd, 0, SEEK_SET)) == (off_t) -1) return 6;
    nBytes = sizeof(ImageHeader);
    if (::write(fd, &header, nBytes) != nBytes) return 7;
    return 0;
}

bool CompressedNetwork::read(const std::string &file)
{
    int fd = open(file.c_str(), O_RDONLY);
    if (fd == -1) {
        error("cannot open '%s' for reading", file.c_str());
        return false;
    }
    bool r = readData(fd);
    if (!r) {
        error("cannot read network from '%s'", file.c_str());
    } else {
        log("memory mapped '%s'", file.c_str());
    }
    return r;
}

bool CompressedNetwork::readData(int fd)
{
    ssize_t nBytes;
    ImageHeader header;
    nBytes = sizeof(ImageHeader);
    if (::read(fd, &header, nBytes) != nBytes) {
        warning("cannot read header");
        return false;
    }
    if (memcmp(header.magic, ImageHeader::magicWord, 8)) {
        warning("bad magic word in file header");
        return false;
    }
    if (header.version != ImageHeader::formatVersion) {
        warning("file format is in version %d, expected %d", header.version, ImageHeader::formatVersion);
        return false;
    }
    nStates_ = header.nStates;
    nArcs_ = header.nArcs;
    nEpsilonArcs_ = header.nEpsilonArcs;
    initialStateIndex_ = header.initialStateIndex;

    mmap_ = (char*) mmap(0, mmapSize_ = header.end, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_ == MAP_FAILED) {
        error("mmap failed");
        return false;
    }
    states_ = reinterpret_cast<State*>(mmap_ + header.statesOffset);
    arcs_ = reinterpret_cast<Arc*>(mmap_ + header.arcsOffset);
    epsilonArcs_ = reinterpret_cast<EpsilonArc*>(mmap_ + header.epsArcsOffset);
    return true;
}
