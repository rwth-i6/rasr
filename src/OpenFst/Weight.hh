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
#ifndef _OPENFST_WEIGHT_HH
#define _OPENFST_WEIGHT_HH

namespace OpenFst {

/**
 * convert weights using casting operators of the
 * involved weight classes
 */
template<class WeightFrom, class WeightTo>
struct ImplicitWeightConverter {
    WeightTo operator()(const WeightFrom& w) const {
        return static_cast<WeightTo>(w);
    }
};

template<class T, class WeightTo>
struct ImplicitWeightConverter<FstLib::FloatWeightTpl<T>, WeightTo> {
    WeightTo operator()(const FstLib::FloatWeightTpl<T>& w) const {
        return static_cast<WeightTo>(w.Value());
    }
};

template<class T, class WeightTo>
struct ImplicitWeightConverter<FstLib::TropicalWeightTpl<T>, WeightTo> {
    WeightTo operator()(const FstLib::TropicalWeightTpl<T>& w) const {
        return static_cast<WeightTo>(w.Value());
    }
};

template<class T, class WeightTo>
struct ImplicitWeightConverter<FstLib::LogWeightTpl<T>, WeightTo> {
    WeightTo operator()(const FstLib::LogWeightTpl<T>& w) const {
        return static_cast<WeightTo>(w.Value());
    }
};

}  // namespace OpenFst

#endif /* _OPENFST_WEIGHT_HH */
