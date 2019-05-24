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
#ifndef SIMPLE_THREAD_POOL_HH
#define SIMPLE_THREAD_POOL_HH

#include <Core/Types.hh>

/**
 * The API of this thread-pool is not thread-safe, it must only be used from within the foreground.
 * */

class SimpleThreadPool {
public:
    /// Override this class to implement your jobs
    struct Job {
        /// This function is executed in the background thread
        virtual void run() = 0;
        /// The destructor is always called in the foreground, you can use it to process the results
        virtual ~Job() {
        }
    };

    SimpleThreadPool(u32 nThreads);

    ~SimpleThreadPool();

    /// Starts a job.
    /// If there is no idle thread, or if enforceSync is true, the job is run synchronously.
    /// Ownership of the job goes to the ThreadJobPool.
    void start(Job* job, bool enforceSync = false);

    /// If @param one is true, returns after at least one job has finished,
    /// otherwise waits until _all_ jobs are finished.
    /// Always returns immediately if there are no running jobs.
    void wait(bool one = false);

    /// Check for finished jobs and eventually call their destructors
    /// This happens automatically from within wait()
    /// Returns the number of jobs that were finished.
    u32 manage();

    /// Returns true if all threads are idle
    bool idle() const;

private:
    class Thread;
    std::vector<Thread*> idleThreads_;
    std::vector<Thread*> busyThreads_;
};

#endif
