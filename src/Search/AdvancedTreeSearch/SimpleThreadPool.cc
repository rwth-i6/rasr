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
#include "SimpleThreadPool.hh"
#include <Core/Debug.hh>
#include <Core/Thread.hh>

class SimpleThreadPool::Thread : public Core::Thread {
public:
    Thread();

    void stopThread();

    void startJob(Job* job);

    virtual void run();

    Job*            job_;
    bool            jobReady_, waiting_;
    Core::Condition wait_;
};

SimpleThreadPool::Thread::Thread()
        : job_(0),
          jobReady_(false),
          waiting_(false) {
}

void SimpleThreadPool::Thread::stopThread() {
    startJob(0);
}

void SimpleThreadPool::Thread::startJob(SimpleThreadPool::Job* job) {
    verify(job_ == 0);
    job_      = job;
    jobReady_ = false;
    while (!waiting_)
        wait_.timedWait(1);
    while (waiting_)
        wait_.broadcast();
}

void SimpleThreadPool::Thread::run() {
    while (true) {
        waiting_ = true;
        wait_.wait();
        waiting_ = false;
        if (!job_) {
            jobReady_ = true;
            return;
        }
        job_->run();
        jobReady_ = true;
        wait_.broadcast();
    }
}

SimpleThreadPool::SimpleThreadPool(u32 nThreads) {
    for (u32 t = 0; t < nThreads; ++t) {
        idleThreads_.push_back(new Thread());
        idleThreads_.back()->start();
    }
}

SimpleThreadPool::~SimpleThreadPool() {
    wait();
    verify(busyThreads_.empty());

    for (u32 t = 0; t < busyThreads_.size(); ++t) {
        idleThreads_[t]->stopThread();
        idleThreads_[t]->wait();
        delete idleThreads_[t];
    }
}

void SimpleThreadPool::start(Job* job, bool enforceSync) {
    manage();

    if (enforceSync || idleThreads_.empty()) {
        job->run();
        delete job;
    }
    else {
        idleThreads_.back()->startJob(job);
        busyThreads_.push_back(idleThreads_.back());
        idleThreads_.pop_back();
    }
}

void SimpleThreadPool::wait(bool one) {
    if (manage() && one)
        return;
    while (!busyThreads_.empty()) {
        busyThreads_.front()->wait_.timedWait(1);
        if (manage() && one)
            return;
    }
    verify(busyThreads_.empty());
}

u32 SimpleThreadPool::manage() {
    u32 deleted = 0;
    for (u32 t = 0; t < busyThreads_.size();) {
        if (busyThreads_[t]->jobReady_) {
            delete busyThreads_[t]->job_;
            busyThreads_[t]->job_ = 0;
            idleThreads_.push_back(busyThreads_[t]);
            busyThreads_.erase(busyThreads_.begin() + t);
            ++deleted;
        }
        else {
            ++t;
        }
    }
    return deleted;
}

bool SimpleThreadPool::idle() const {
    return busyThreads_.empty();
}
