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
#ifndef NN_PYTHONCONTROL_HH
#define NN_PYTHONCONTROL_HH

#include <Core/Component.hh>
#include <Python/Init.hh>
#include <Python/Numpy.hh>
#include <memory>
#include <string>
#include <Python.h>

namespace Nn {

class PythonControl : virtual public Core::Component {
    typedef Core::Component Precursor;

public:
    PythonControl(const Core::Configuration& config, const std::string& sprintUnit, bool isOptional);
    virtual ~PythonControl();
    struct Internal;

    // All these are save to be called in any state, they don't need the Python GIL (but it's also ok if you hold it).
    bool isEnabled() const {
        return pyObject_;
    }
    void run_iterate_corpus();
    void run_control_loop();
    void run_custom(const char* method, const char* kwArgsFormat, ...) const;
    void exit();

    // These expect to be run only with the Python GIL held.
    PyObject*                 run_custom_with_result(const char* method, const char* kwArgsFormat, ...) const;
    Core::Component::Message  pythonCriticalError(const char* msg = 0, ...) const;
    Python::CriticalErrorFunc getPythonCriticalErrorFunc() const;

protected:
    std::string         sprintUnit_;
    Python::Initializer pythonInitializer_;
    PyObject*           pyObject_;

    std::shared_ptr<Internal> internal_;
};

}  // namespace Nn

#endif  // PYTHONCONTROL_HH
