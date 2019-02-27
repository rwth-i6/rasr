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
#include <sched.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#include "TheanoCommunicator.hh"

using namespace Bliss;
using namespace std;

namespace {
std::string getSegmentName(const Bliss::SpeechSegment* segment) {
    return segment->recording()->parent()->name() + "/" + segment->recording()->name() + "/" + segment->name();
}

}  // namespace

const Core::ParameterInt TheanoCommunicator::paramSharedMemKey(
        "shared-mem-key", "(random) number which is used as key for shared memory", -1);

std::unique_ptr<TheanoCommunicator> TheanoCommunicator::communicator_;

TheanoCommunicator& TheanoCommunicator::communicator() {
    hope(communicator_);
    return *communicator_;
}

enum TheanoStatus {
    IDLE                = 0,
    REQUEST_ERRORSIGNAL = 1,
    ERRORSIGNAL_WRITTEN = 2,
    TERMINATED          = 4
};

namespace {
/**
 * layout of shared memory:
 * 4 bytes TheanoStatus flag
 * 4 bytes float nRows
 * 4 bytes float nCols
 * CTL_SEG_SIZE-12 bytes segmentName (0 terminated)
 * rest: data
 **/
const u32 SHARED_MEM_SIZE    = 5 * 1024 * 1024;  //5M should be enough
const u32 CTL_SEG_SIZE       = 512;
const u32 STATUS_BEGIN       = 0;
const u32 ROWS_BEGIN         = 4;
const u32 COLS_BEGIN         = 8;
const u32 LOSS_BEGIN         = 12;
const u32 SEGMENT_NAME_BEGIN = 16;
const u32 DATA_BEGIN         = CTL_SEG_SIZE;
const u32 MAX_SEG_NAME_LEN   = DATA_BEGIN - SEGMENT_NAME_BEGIN;
}  // namespace

template<typename T>
//careful: the offset is in bytes and not related to T
T& TheanoCommunicator::shMem(u32 offset) {
    return *reinterpret_cast<T*>(static_cast<char*>(shMem_) + offset);
}

u32& TheanoCommunicator::shMemStatus() {
    return shMem<u32>(STATUS_BEGIN);
}

float& TheanoCommunicator::shMemData(u32 idx) {
    return shMem<float>(DATA_BEGIN + 4 * idx);
}

u32& TheanoCommunicator::shMemRows() {
    return shMem<u32>(ROWS_BEGIN);
}

u32& TheanoCommunicator::shMemCols() {
    return shMem<u32>(COLS_BEGIN);
}

float& TheanoCommunicator::shMemLoss() {
    return shMem<float>(LOSS_BEGIN);
}

void TheanoCommunicator::waitForStatus(u32 status) {
    while (!(shMemStatus() & status)) {
        sched_yield();
    }
}

TheanoCommunicator::TheanoCommunicator(const Core::Configuration& c)
        : Core::Component(c) {
    key_t shMemKey = paramSharedMemKey(c);
    if (shMemKey == -1) {
        criticalError("sharedMemKey not specified");
    }
    log() << "allocating " << SHARED_MEM_SIZE << " bytes of shared memory...";
    shId_ = shmget(shMemKey, SHARED_MEM_SIZE, IPC_CREAT | 0660);
    if (shId_ < 0) {
        criticalError("failed to allocated shared memory (shmget failed)");
    }
    shMem_ = shmat(shId_, 0, 0);
    if (shMem_ == (void*)-1) {
        criticalError("failed to allocate shared memory (shmat failed)");
    }
    //shmctl(shId_, IPC_RMID, 0); //mark for deletion, will only be deleted after detaching
    //note: seems not to work like this..., so do it in destructor instead
    shMemStatus() = IDLE;
    log() << "shared memory allocation was successfull";
}

TheanoCommunicator::~TheanoCommunicator() {
    shmctl(shId_, IPC_RMID, 0);
    shmdt(shMem_);
}

void TheanoCommunicator::create(const Core::Configuration& c) {
    if (!communicator_) {
        communicator_.reset(new TheanoCommunicator(c));
    }
}

bool TheanoCommunicator::waitForErrorSignalRequest(/*out*/ std::string& segmentName) {
    waitForStatus(REQUEST_ERRORSIGNAL | TERMINATED);
    if (shMemStatus() == TERMINATED) {
        segmentName = "invalid";
        return false;
    }

    segmentName = &shMem<char>(SEGMENT_NAME_BEGIN);
    log() << "error signal for segment " << segmentName << " was requested by theano";
    return true;
}

const Math::Matrix<f32>& TheanoCommunicator::getPosteriorsForSegment(const Bliss::SpeechSegment* segment) {
    std::string segmentName = getSegmentName(segment);
    if (segmentName == currentSegmentName_) {
        /* use caching only a single time per segment
        assumption: each segment is requested 2 times in a row,
        once for rescoring and once for accumulation
        after this the same segment might be requested again
        but then we want to get new posteriors from theano */
        currentSegmentName_ = "";
        return posteriors_;
    }

    timeval start;
    TIMER_START(start);

    //shared memory transmission
    std::string shMemSegmentName = &shMem<char>(SEGMENT_NAME_BEGIN);
    if (segmentName != shMemSegmentName) {
        criticalError("segment names do not match: ") << segmentName << " " << shMemSegmentName;
    }
    int rows = shMemRows();
    int cols = shMemCols();
    posteriors_.resize(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            posteriors_[i][j] = shMemData(i * cols + j);
        }
    }
    shMemStatus() = IDLE;

    currentSegmentName_ = segmentName;
    timeval end;
    f64     diff = 0;
    TIMER_STOP(start, end, diff);
    log() << "time getting posteriors: " << diff;
    return posteriors_;
}

void TheanoCommunicator::writeErrorSignalForSegment(const Bliss::SpeechSegment* segment, float loss, const Math::Matrix<f32>& errSig) {
    std::string segmentName = getSegmentName(segment);
    log() << "writing error signal for segment " << segmentName;
    if (shMemStatus() != IDLE) {
        criticalError("unexpected shMemStatus");
    }
    shMemRows() = static_cast<float>(errSig.nRows());
    shMemCols() = static_cast<float>(errSig.nColumns());
    shMemLoss() = loss;

    for (u32 i = 0; i < errSig.nRows(); ++i) {
        for (u32 j = 0; j < errSig.nColumns(); ++j) {
            shMemData(i * errSig.nColumns() + j) = errSig[i][j];
        }
    }

    shMemStatus() = ERRORSIGNAL_WRITTEN;
    log() << "done writing error signal";
}
