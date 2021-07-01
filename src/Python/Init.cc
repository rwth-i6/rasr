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
#include "Init.hh"
#include <Core/Debug.hh>
#include <cstdlib>
#include "Numpy.hh"
#include "Utilities.hh"

namespace Python {

void Initializer::AtExitUninitHandler() {
    // Could already be finalized elsewhere.
    if (!Py_IsInitialized())
        return;

    // It is important to call Py_Finalize() at the very end.
    // This is because we want to able to do CPython calls at any time.
    // Note that we anyway always want this function to be called
    // at least once somewhere because it can trigger some important
    // Python cleanup code. E.g. Theano profiling by default works this way.

    // http://stackoverflow.com/questions/13586451/when-is-a-function-registered-with-atexit-called
    // There could be global objects which destructors are going to be called after this.
    // However, all lifetime of any Initializer objects should have ended much before that.
    // Thus, if initCounter_ > 0, those are objects which are likely not going to be
    // deleted anymore.
    if (initCounter_ > 0)
        Core::printWarning("There are %u left-over Python::Initializer instances. We uninit Python now.", initCounter_);

    Python::ScopedGIL gil;
    Py_Finalize();
}

unsigned int Initializer::initCounter_ = 0;

static const Core::ParameterString paramPythonHome(
        "python-home",
        "if set, is used for Py_SetPythonHome. an alternative would be to set the PYTHONHOME env variable",
        "");
static std::string _pythonHome;

static const Core::ParameterString paramPythonProgramName(
        "python-program-name",
        "if set, is used for Py_SetProgramName. accessiable via sys.executable",
        "");
static std::string _pythonProgramName;

void Initializer::init() {
    if (isInitialized_)
        return;
    isInitialized_ = true;
    if (initCounter_++ > 0)
        return;
    // We are here if this is the first Initializer.

    // CPython could be initialized by external code.
    // We hope that everything is fine then.
    // (This case usually doesn't happen. It could happen via other external libs or so.)
    if (Py_IsInitialized()) {
        Core::printWarning("Python is already initialized before first Python::Initializer instance.");
        return;
    }

    if (Core::Application::us()) {
        _pythonHome        = paramPythonHome(Core::Application::us()->getConfiguration());
        _pythonProgramName = paramPythonProgramName(Core::Application::us()->getConfiguration());

        if (!_pythonHome.empty())
            Py_SetPythonHome(Py_DecodeLocale(_pythonHome.c_str(), nullptr));
        if (!_pythonProgramName.empty())
            Py_SetProgramName(Py_DecodeLocale(_pythonProgramName.c_str(), nullptr));
    }
    else
        Core::printWarning("Python::Initializer: no Application instance found, cannot load parameters");

    // Init CPython if not yet initialized. Safe to be called multiple times.
    Py_InitializeEx(0 /* don't install signal handlers */);

    // Start the CPython interpreter's thread-awareness, if not yet done.
    // Safe to be called multiple times.
    PyEval_InitThreads();

    // Note that we expect that we have the CPython GIL acquired
    // at this moment. If we initialized CPython above, this is the case.

    // Allow other Python threads to run in the meanwhile.
    // Note that this means that we explicitely will need to aquire the CPython GIL
    // before any further CPython API call. We do this via Python::ScopedGIL.
    PyEval_SaveThread();

    // See comment in AtExitUninitHandler().
    std::atexit(AtExitUninitHandler);
    if (Core::Application::us())
        Core::Application::us()->atexit(AtExitUninitHandler);

    // Acquire the GIL to do some further initing.
    ScopedGIL gil;

    initNumpy();
}

void Initializer::uninit() {
    if (!isInitialized_)
        return;
    isInitialized_ = false;
    verify_gt(initCounter_, 0);
    if (--initCounter_ > 0)
        return;

    // CPython could be finalized by external code.
    // (This case usually should not happen.)
    if (!Py_IsInitialized()) {
        Core::printWarning("Python is already uninitialized before last Python::Initializer instance uninits.");
        return;
    }

    Python::ScopedGIL gil;
    // Py_Finalize is done via std::atexit. See constructor code+comment.
    // However, we also do it here because I get strange crashes in some cases in Theanos CUDA exit.
    Py_Finalize();
}

}  // namespace Python
