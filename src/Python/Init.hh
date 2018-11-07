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
#ifndef _PYTHON_PYTHONINIT_HH
#define _PYTHON_PYTHONINIT_HH

namespace Python {

/*
 * Init CPython and relevant systems.
 * Note that this is *not* fully thread-safe.
 *
 * Keep an instance of this class and call init() whereever you want to use Python.
 * The first init() call will init CPython, and the last uninit() call will finalize CPython.
 * Note that uninit() is also automatically called in the destructor.
 *
 * Note that the default state is that no thread is holding the GIL.
 * This is also the state after the first init() call.
 * Use ScopedGIL wherever you want to call Python code.
 */
class Initializer {
    static unsigned int initCounter_;
    bool                isInitialized_;

public:
    Initializer()
            : isInitialized_(false) {}
    ~Initializer() {
        uninit();
    }
    Initializer(const Initializer& other)
            : isInitialized_(false) {
        if (other.isInitialized_)
            init();
    }
    Initializer& operator=(const Initializer&) = delete;

    // init() can be called safely when CPython is already initialized, with or without the GIL.
    // Will incr initCounter_.
    // If we init CPython, we will return with the GIL released.
    void init();

    // Decr initCounter_, and, if zero, uninit CPython.
    void uninit();

    // Will be registerd via std::atexit() on the first init() call,
    // but it's also save to call this multiple times at exit
    // if you can assure that there wont be any Python access afterwards.
    static void AtExitUninitHandler();
};

}  // namespace Python

#endif  // PYTHONINIT_HH
