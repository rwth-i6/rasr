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
#include "Utilities.hh"

#include <Core/Application.hh>

namespace Python {

// kwArgsFormat: https://docs.python.org/2/c-api/arg.html
PyObject* PyCallKw(PyObject* obj, const char* method, const char* kwArgsFormat, ...) {
    PyObject* meth   = nullptr;
    PyObject* args   = nullptr;
    PyObject* kwArgs = nullptr;
    PyObject* res    = nullptr;

    meth = PyObject_GetAttrString(obj, method);
    if (!meth)
        goto final;

    args = PyTuple_New(0);
    if (!args)
        goto final;

    va_list vargs;
    va_start(vargs, kwArgsFormat);
    kwArgs = Py_VaBuildValue(kwArgsFormat, vargs);
    va_end(vargs);
    if (!kwArgs)
        goto final;

    res = PyObject_Call(meth, args, kwArgs);

final:
    Py_XDECREF(meth);
    Py_XDECREF(args);
    Py_XDECREF(kwArgs);
    return res;
}

bool PyCallKw_IgnRet_HandleError(CriticalErrorFunc errorFunc, PyObject* obj, const char* method, const char* kwArgsFormat, ...) {
    PyObject* meth   = nullptr;
    PyObject* args   = nullptr;
    PyObject* kwArgs = nullptr;
    PyObject* res    = nullptr;

    meth = PyObject_GetAttrString(obj, method);
    if (!meth)
        goto final;

    args = PyTuple_New(0);
    if (!args)
        goto final;

    va_list vargs;
    va_start(vargs, kwArgsFormat);
    kwArgs = Py_VaBuildValue(kwArgsFormat, vargs);
    va_end(vargs);
    if (!kwArgs)
        goto final;

    res = PyObject_Call(meth, args, kwArgs);

final:
    Py_XDECREF(meth);
    Py_XDECREF(args);
    Py_XDECREF(kwArgs);
    if (res) {
        Py_DECREF(res);
        return true;
    }

    errorFunc().form("exception while calling '%s'", method);
    handlePythonError();
    return false;
}

void addSysPath(const std::string& path) {
    // add path to sys.path
    PyObject* pySysPath = PySys_GetObject((char*)"path");  // borrowed ref
    if (!pySysPath) {
        criticalError("Python: no sys.path. cannot add '%s'", path.c_str());
        return;
    }
    PyObject* res = PyObject_CallMethod(pySysPath, (char*)"insert", (char*)"is", 0, path.c_str());
    if (!res) {
        criticalError("Python: sys.path.append failed. cannot add '%s'", path.c_str());
        return;
    }
    Py_CLEAR(res);
}

bool pyObjToStr(PyObject* obj, std::string& str) {
    if (!obj)
        return false;
    if (PyUnicode_CheckExact(obj)) {
        PyObject* strObj = PyUnicode_AsUTF8String(obj);
        if (!strObj)
            return false;
        bool res = false;
        if (PyUnicode_Check(strObj)) {
            res = pyObjToStr(strObj, str);
        }
        str = PyBytes_AsString(strObj);
        res = true;
        Py_DECREF(strObj);
        return res;
    }
    else {
        PyObject* unicodeObj = obj;
        if (!unicodeObj)
            return false;
        bool res = false;
        if (PyUnicode_Check(unicodeObj))
            res = pyObjToStr(unicodeObj, str);
        Py_DECREF(unicodeObj);
        return res;
    }
    return false;
}

template<>
PyObject* newObject(const std::string& source) {
    return PyUnicode_FromStringAndSize(source.data(), source.size());
}

template<>
PyObject* newObject(const size_t& source) {
    return PyLong_FromSize_t(source);
}

template<>
PyObject* newObject(const f32& source) {
    return PyFloat_FromDouble(source);
}

template<>
PyObject* newObject(const f64& source) {
    return PyFloat_FromDouble(source);
}

template<typename T>
PyObject* newList(const T& source) {
    PyObject* list = PyList_New(source.size());
    if (!list)
        return nullptr;
    size_t i = 0;
    for (const auto& item : source) {
        PyObject* obj = newObject(item);
        if (!obj) {
            Py_CLEAR(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, i, obj);
        ++i;
    }
    return list;
}

PyObject* incremented(PyObject* obj) {
    Py_INCREF(obj);
    return obj;
}

template<typename T>
bool dict_SetItemString(PyObject* dict, const char* key, const T& value) {
    PyObject* valueObj = newObject(value);
    if (!valueObj)
        return false;
    int res = PyDict_SetItemString(dict, key, valueObj);
    Py_DECREF(valueObj);
    return res == 0;
}

template PyObject* newList<std::vector<std::string>>(const std::vector<std::string>& list);
template PyObject* newList<std::list<std::string>>(const std::list<std::string>& list);
template bool      dict_SetItemString<std::string>(PyObject*, const char*, const std::string&);
template bool      dict_SetItemString<size_t>(PyObject*, const char*, const size_t&);
template bool      dict_SetItemString<f32>(PyObject*, const char*, const f32&);
template bool      dict_SetItemString<f64>(PyObject*, const char*, const f64&);

// If there was an exception in the current thread:
//  - Runs the standard global exception handler.
//  - Cleans the error state.
//  - Returns true.
// Otherwise: Returns false.
// Thus, this can be called safely multiple times.
bool handlePythonError() {
    if (PyErr_Occurred()) {
        PyErr_Print();
        return true;
    }
    return false;
}

Core::Component::Message criticalError(const char* msg, ...) {
    handlePythonError();

    va_list ap;
    va_start(ap, msg);
    Core::Component::Message msgHelper = Core::Application::us()->vCriticalError(msg, ap);
    va_end(ap);
    return msgHelper;
}

static std::string formatPrettyFallback(PyObject* obj) {
    std::string raw;
    if (!pyObjToStr(obj, raw))
        return "(pformat error) (pyObjToStr error)";
    else
        return "(pformat error) " + raw;
}

std::string formatPretty(PyObject* obj) {
    if (!obj)
        return "nullptr";
    ObjRef pyMod;
    pyMod.takeOver(PyImport_ImportModule("pprint"));
    if (!pyMod) {
        handlePythonError();
        return formatPrettyFallback(obj);
    }
    ObjRef pyRes;
    pyRes.takeOver(PyCallKw(pyMod, "pformat", "{s:O}", "object", obj));
    if (!pyRes) {
        handlePythonError();
        return formatPrettyFallback(obj);
    }
    std::string s;
    if (!pyObjToStr(pyRes, s))
        return formatPrettyFallback(obj);
    return s;
}

void dumpModulesEnv() {
    PyObject* mods = PyImport_GetModuleDict();  // borrowed ref
    fprintf(stderr, "sys.modules = %s\n", formatPretty(mods).c_str());
    PyObject* pySysPath = PySys_GetObject((char*)"path");  // borrowed ref
    fprintf(stderr, "sys.path = %s\n", formatPretty(pySysPath).c_str());
}

}  // namespace Python
