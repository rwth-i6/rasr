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
#ifndef _TENSORFLOW_TENSOR_MAP_HH
#define _TENSORFLOW_TENSOR_MAP_HH

#include <Core/Component.hh>

namespace Tensorflow {

// base class that contains information for both inputs and outputs
class TensorInfo : public Core::Configurable {
public:
    typedef Core::Configurable Precursor;

    static const Core::ParameterString paramParamName;
    static const Core::ParameterString paramTensorName;
    static const Core::ParameterString paramSeqLengthTensorName;

    TensorInfo(Core::Configuration const& config);
    ~TensorInfo() = default;

    std::string const& param_name() const;
    std::string const& tensor_name() const;
    std::string const& seq_length_tensor_name() const;

private:
    std::string param_name_;
    std::string tensor_name_;
    std::string seq_length_tensor_name_;
};

// input specific information (none yet)
class TensorInputInfo : public TensorInfo {
public:
    typedef TensorInfo Precursor;

    TensorInputInfo(Core::Configuration const& config);
    ~TensorInputInfo() = default;
};

// output specific information (none yet)
class TensorOutputInfo : public TensorInfo {
public:
    typedef TensorInfo Precursor;

    TensorOutputInfo(Core::Configuration const& config);
    ~TensorOutputInfo() = default;
};

template<typename Info>
class TensorMap : public Core::Component {
public:
    typedef Core::Component Precursor;

    TensorMap(Core::Configuration const& config);
    ~TensorMap() = default;

    bool        has_info(std::string const& name) const;
    Info const& get_info(std::string const& name) const;

private:
    std::unordered_map<std::string, Info> tensor_infos_;
};

typedef TensorMap<TensorInputInfo>  TensorInputMap;
typedef TensorMap<TensorOutputInfo> TensorOutputMap;

// -------------------- inline implementations --------------------

// ---------- TensorInfo ----------

inline TensorInfo::TensorInfo(Core::Configuration const& config)
        : Precursor(config), param_name_(paramParamName(config)), tensor_name_(paramTensorName(config)), seq_length_tensor_name_(paramSeqLengthTensorName(config)) {
}

inline std::string const& TensorInfo::param_name() const {
    return param_name_;
}

inline std::string const& TensorInfo::tensor_name() const {
    return tensor_name_;
}

inline std::string const& TensorInfo::seq_length_tensor_name() const {
    return seq_length_tensor_name_;
}

// ---------- TensorInputInfo ----------

inline TensorInputInfo::TensorInputInfo(Core::Configuration const& config)
        : Precursor(config) {
}

// ---------- TensorOutputInfo ----------

inline TensorOutputInfo::TensorOutputInfo(Core::Configuration const& config)
        : Precursor(config) {
}

// ---------- TensorMap ----------

template<typename Info>
inline bool TensorMap<Info>::has_info(std::string const& name) const {
    return tensor_infos_.find(name) != tensor_infos_.end();
}

template<typename Info>
inline Info const& TensorMap<Info>::get_info(std::string const& name) const {
    auto iter = tensor_infos_.find(name);
    if (iter == tensor_infos_.end()) {
        criticalError("Could not find information for input/output: ") << name;
    }
    return iter->second;
}

}  // namespace Tensorflow

#endif  // _TENSORFLOW_FORWARD_NODE_HH
