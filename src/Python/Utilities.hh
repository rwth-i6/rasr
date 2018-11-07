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
#ifndef _PYTHON_UTILITIES_HH
#define _PYTHON_UTILITIES_HH

#include <functional>
#include <list>
#include <string>

#include <Python.h>

#include <Core/Assertions.hh>
#include <Core/Component.hh>
#include <Core/Utility.hh>

namespace Python {

/*
 * Ensures that we have the GIL aquired in this scope.
 * This can also be used if we already have the GIL.
 *
 * (Almost) all CPython API calls expect that we have the CPython GIL aquired.
 * This must only be constructed when CPython is initialized.
 *
 * Note that the default state in our project is that no thread is holding the GIL.
 * This is also the state after the first Initializer::init() call.
 *
 * When you have aquired the GIL, you can use ScopedAllowThreads
 * in embedded scopes to release it temporarily.
 */
struct ScopedGIL {
    PyGILState_STATE gstate;

    ScopedGIL() {
        verify(Py_IsInitialized());
        gstate = PyGILState_Ensure();
    }

    ~ScopedGIL() {
        // Very special case: If we already finalized CPython,
        // we must not call PyGILState_Release anymore.
        if (unlikely(!Py_IsInitialized()))
            return;
        PyGILState_Release(gstate);
    }

    ScopedGIL(const ScopedGIL&) = delete;
};

/*
 * Like a Py_BEGIN_ALLOW_THREADS/Py_END_ALLOW_THREADS scope.
 * Releases the GIL in this scope.
 * This expects that you have the GIL when this is constructed.
 */
struct ScopedAllowThreads {
    PyThreadState* savedState;

    ScopedAllowThreads()
            : savedState(PyEval_SaveThread()) {}
    ~ScopedAllowThreads() {
        PyEval_RestoreThread(savedState);
    }
    ScopedAllowThreads(const ScopedAllowThreads&) = delete;
};

/*
 * Simple wrapper around PyObject.
 * Does automatic reference counting.
 */
struct ObjRef {
    PyObject* obj;

    // These will automatically take the GIL if needed.
    ObjRef()
            : obj(nullptr) {}
    ~ObjRef() {
        if (obj) {
            ScopedGIL gil;
            clear();
        }
    }
    ObjRef(const ObjRef& other)
            : obj(nullptr) {
        *this = other;
    }
    ObjRef& operator=(const ObjRef& other) {
        if (*this || other) {
            ScopedGIL gil;
            copyRef(other.obj);
        }
        return *this;
    }
    operator bool() const {
        return obj != nullptr;
    }
    operator PyObject*() const {
        return obj;
    }
    PyObject* operator->() const {
        return obj;
    }
    PyObject* release() {
        PyObject* res = obj;
        obj           = nullptr;
        return res;
    }

    // These must be called with the GIL.
    void clear() {
        Py_CLEAR(obj);
    }
    ObjRef& takeOver(PyObject* o) {
        clear();
        obj = o;
        return *this;
    }
    ObjRef& copyRef(PyObject* o) {
        if (obj == o)
            return *this;
        clear();
        obj = o;
        if (obj)
            Py_INCREF(obj);
        return *this;
    }
};

typedef std::function<Core::Component::Message()> CriticalErrorFunc;

// The following expect to have the Python GIL.

PyObject* PyCallKw(PyObject* obj, const char* method, const char* kwArgsFormat, ...);
bool      PyCallKw_IgnRet_HandleError(CriticalErrorFunc errorFunc, PyObject* obj, const char* method, const char* kwArgsFormat, ...);
void      addSysPath(const std::string& path);
bool      pyObjToStr(PyObject* obj, std::string& str);
template<typename T>
PyObject* newObject(const T& source);
template<typename T>
PyObject* newList(const T& list);
PyObject* incremented(PyObject* obj);
template<typename T>
bool dict_SetItemString(PyObject* dict, const char* key, const T& value);

bool                     handlePythonError();
Core::Component::Message criticalError(const char* msg, ...);
std::string              formatPretty(PyObject* obj);
void                     dumpModulesEnv();

}  // namespace Python

#endif  // UTILITIES_HH
