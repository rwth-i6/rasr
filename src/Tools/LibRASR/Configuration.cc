/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "Configuration.hh"

#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <Core/Configuration.hh>
#include <Core/StringUtilities.hh>
#include <Core/Types.hh>

namespace {

// Names in python are relative to the selection of a config, e.g. "search-algorithm.max-beam-size" in a config with
// selection "lib-rasr" refers to "lib-rasr.search-algorithm.max-beam-size".
// The checks mirror the preconditions of Core::Configuration, which would otherwise abort the python process.

std::string absoluteName(Core::Configuration const& config, std::string const& name) {
    return config.getSelection().empty() ? name : config.getSelection() + "." + name;
}

// Path of a parameter or selection, e.g. "search-algorithm.max-beam-size"
void checkPath(std::string const& path) {
    if (not Core::Configuration::isWellFormedParameterName(path)) {
        throw py::value_error("Invalid name '" + path + "': expected names separated by '.' without '*', e.g. \"search-algorithm.max-beam-size\"");
    }
}

// Pattern of the parameters to set, e.g. "*.log.channel"
void checkPattern(std::string const& pattern) {
    if (not Core::Configuration::isWellFormedResourceName(pattern)) {
        throw py::value_error("Invalid name '" + pattern + "': expected names separated by '.', where '*' matches any names, e.g. \"*.log.channel\"");
    }
}

// Core::strconv silently wraps integers outside of the 32-bit range of RASR's int parameters
bool parseInt(std::string const& value, s32& result) {
    if (not Core::strconv(value, result)) {
        return false;
    }
    char*     end  = nullptr;
    long long full = std::strtoll(value.c_str(), &end, 10);
    // Without digits, the value is one of the words for infinity, which Core maps to the limits
    return end == value.c_str() or full == result;
}

// Like ParameterIntVector, i.e. elements separated by `delimiter` and ranges such as "1:4"
bool parseIntList(std::string const& value, std::vector<s32>& result, std::string const& delimiter) {
    if (not Core::str2vector(value, result, delimiter, ":", Core::Type<s32>::min, Core::Type<s32>::max, Core::Type<u32>::min, Core::Type<u32>::max, s32(1))) {
        return false;
    }
    // As for single ints, the numbers of all elements and range borders are checked for the 32-bit range
    for (std::string element : Core::split(value, delimiter)) {
        Core::stripWhitespace(element);
        size_t range = element.find(':', 1);
        s32    number;
        if (not element.empty() and (not parseInt(element.substr(0, range), number) or (range != std::string::npos and not parseInt(element.substr(range + 1), number)))) {
            return false;
        }
    }
    return true;
}

// Like ParameterFloatVector, i.e. elements separated by `delimiter` and ranges such as "0:2" with step 1
bool parseFloatList(std::string const& value, std::vector<f64>& result, std::string const& delimiter) {
    return Core::str2vector(value, result, delimiter, ":", Core::Type<f64>::min, Core::Type<f64>::max, Core::Type<u32>::min, Core::Type<u32>::max, f64(1.0));
}

void checkDelimiter(std::string const& delimiter) {
    if (delimiter.empty()) {
        throw py::value_error("The delimiter must not be empty");
    }
}

// Converts a python value to RASR's string format, e.g. True to "true"
std::string toString(py::handle value) {
    if (py::isinstance<py::bool_>(value)) {
        return value.cast<bool>() ? "true" : "false";
    }
    if (py::isinstance<py::str>(value)) {
        return value.cast<std::string>();
    }
    if (PyIndex_Check(value.ptr())) {  // int, but also e.g. numpy integers
        return py::str(py::int_(py::reinterpret_borrow<py::object>(value)));
    }
    if (PyObject_HasAttrString(value.ptr(), "__fspath__")) {
        return py::module_::import("os").attr("fspath")(value).cast<std::string>();
    }
    if (PyObject_HasAttrString(value.ptr(), "__float__")) {  // float, but also e.g. numpy floats
        return py::repr(py::float_(py::reinterpret_borrow<py::object>(value)));
    }
    throw py::type_error("Unsupported value of type " + py::type::of(value).attr("__name__").cast<std::string>() + ", expected str, bool, int, float, a path or a list of these");
}

Core::Configuration subConfiguration(Core::Configuration const& config, std::string const& path) {
    checkPath(path);
    Core::Configuration result(config);
    result.setSelection(absoluteName(config, path));
    return result;
}

void setValue(Core::Configuration& config, std::string const& name, py::object const& value, std::string const& delimiter) {
    checkPattern(name);
    std::string string;
    if (py::isinstance<py::list>(value) or py::isinstance<py::tuple>(value)) {
        checkDelimiter(delimiter);
        bool first = true;
        for (py::handle element : value) {
            std::string elementString = toString(element);
            // Otherwise the element would be split into several elements when reading the list
            if (elementString.find(delimiter) != std::string::npos) {
                throw py::value_error("List element '" + elementString + "' contains the delimiter '" + delimiter + "'");
            }
            string += (first ? "" : delimiter) + elementString;
            first = false;
        }
    }
    else {
        string = toString(value);
    }
    config.set(absoluteName(config, name), string);
}

std::optional<std::string> getValue(Core::Configuration const& config, std::string const& name) {
    checkPath(name);
    // Core::Configuration::get only accepts single names, so the parent of the parameter is selected first
    size_t              separator = name.rfind('.');
    Core::Configuration parent    = separator == std::string::npos ? config : subConfiguration(config, name.substr(0, separator));
    std::string         value;
    if (parent.get(separator == std::string::npos ? name : name.substr(separator + 1), value)) {
        return value;
    }
    return std::nullopt;
}

// Reads the value of `name` and converts it with `parse`, returns `defaultValue` if it is not set
template<typename T, typename Parse>
std::optional<T> getParsedValue(Core::Configuration const& config, std::string const& name, std::optional<T> const& defaultValue, char const* typeName, Parse parse) {
    std::optional<std::string> value = getValue(config, name);
    if (not value) {
        return defaultValue;
    }
    T result;
    if (not parse(*value, result)) {
        throw py::value_error("Value '" + *value + "' of '" + name + "' is not a valid " + typeName);
    }
    return result;
}

template<typename T, typename Parse>
void defineGetter(py::class_<Core::Configuration>& pyConfig, char const* method, char const* typeName, Parse parse, char const* doc) {
    pyConfig.def(
            method,
            [typeName, parse](Core::Configuration const& self, std::string const& name, std::optional<T> const& defaultValue) {
                return getParsedValue<T>(self, name, defaultValue, typeName, parse);
            },
            py::arg("name"), py::arg("default") = py::none(), doc);
}

template<typename T, typename Parse>
void defineListGetter(py::class_<Core::Configuration>& pyConfig, char const* method, char const* typeName, Parse parse, char const* doc) {
    pyConfig.def(
            method,
            [typeName, parse](Core::Configuration const& self, std::string const& name, std::optional<std::vector<T>> const& defaultValue, std::string const& delimiter) {
                checkDelimiter(delimiter);
                return getParsedValue<std::vector<T>>(self, name, defaultValue, typeName, [&](std::string const& value, std::vector<T>& result) { return parse(value, result, delimiter); });
            },
            py::arg("name"), py::arg("default") = py::none(), py::arg("delimiter") = " ", doc);
}

}  // namespace

void bindConfiguration(py::module_& module) {
    py::class_<Core::Configuration> pyConfig(module, "Configuration");
    pyConfig.def(py::init([]() {
                Core::Configuration config;
                config.setSelection("lib-rasr");
                return config;
            }))
            .def(py::init<Core::Configuration const&>(), py::arg("config"))
            .def(py::init(&subConfiguration), py::arg("config"), py::arg("selection"))
            .def("set", &setValue, py::arg("name"), py::arg("value") = "true", py::arg("delimiter") = " ",
                 "Set a value relative to the selection, e.g. config.set(\"search-algorithm.max-beam-size\", 12). '*' matches any names.\n"
                 "str, bool, int, float and path values are stored as strings in RASR's format, e.g. True as \"true\". "
                 "Lists and tuples of these are joined with `delimiter`, their elements must not contain it.")
            .def("__getitem__", &getValue, py::arg("name"), "Alias for get_str(name).")
            .def("enable_logging", &Core::Configuration::enableLogging)
            .def("set_from_file", static_cast<bool (Core::Configuration::*)(std::string const&)>(&Core::Configuration::setFromFile), py::arg("filename"))
            .def("get_selection", &Core::Configuration::getSelection)
            .def("get_name", &Core::Configuration::getName)
            .def(
                    "set_selection", [](Core::Configuration& self, std::string const& selection) {
                        checkPath(selection);
                        self.setSelection(selection);
                    },
                    py::arg("selection"))
            .def("resolve", static_cast<std::string (Core::Configuration::*)(std::string const& value) const>(&Core::Configuration::resolve), py::arg("value"));

    // The getters read a value relative to the selection and parse it like the corresponding parameter of RASR components
    defineGetter<std::string>(
            pyConfig, "get_str", "str", [](std::string const& value, std::string& result) { result = value; return true; },
            "Read a value, or `default` if it is not set.");
    defineGetter<bool>(
            pyConfig, "get_bool", "bool", [](std::string const& value, bool& result) { return Core::strconv(value, result); },
            "Read a bool like ParameterBool, or `default` if it is not set. \"true\", \"yes\", \"on\" and \"1\" are True, \"false\", \"no\", \"off\" and \"0\" are False.");
    defineGetter<s32>(
            pyConfig, "get_int", "int", &parseInt,
            "Read a 32-bit int like ParameterInt, or `default` if it is not set. \"infinite\" and \"-infinite\" are the limits.");
    defineGetter<f64>(
            pyConfig, "get_float", "float", [](std::string const& value, f64& result) { return Core::strconv(value, result); },
            "Read a float like ParameterFloat, or `default` if it is not set. \"infinite\" and \"-infinite\" are the limits.");
    defineListGetter<std::string>(
            pyConfig, "get_str_list", "str list", [](std::string const& value, std::vector<std::string>& result, std::string const& delimiter) { return Core::str2vector(value, result, delimiter); },
            "Read a list of strings separated by `delimiter` like ParameterStringVector, or `default` if it is not set.");
    defineListGetter<s32>(
            pyConfig, "get_int_list", "int list", &parseIntList,
            "Read a list of 32-bit ints separated by `delimiter` like ParameterIntVector, or `default` if it is not set. Ranges such as \"1:4\" are allowed.");
    defineListGetter<f64>(
            pyConfig, "get_float_list", "float list", &parseFloatList,
            "Read a list of floats separated by `delimiter` like ParameterFloatVector, or `default` if it is not set. Ranges such as \"0:2\" with step 1 are allowed.");
}
