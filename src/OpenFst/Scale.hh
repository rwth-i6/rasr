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
#ifndef _OPENFST_SCALE_HH
#define _OPENFST_SCALE_HH

#include <Core/Types.hh>
#include <fst/arc-map.h>
#include <fst/mutable-fst.h>
#include "Types.hh"

#include <Core/Types.hh>
#include "Types.hh"

namespace OpenFst {

/**
 * Multiply all (float) weights by a scaling factor.
 */
template<class Arc>
class ScaleMapper {
public:
    ScaleMapper(f32 scale)
            : scale_(scale) {}

    Arc operator()(const Arc& arc) const {
        if (arc.weight == Arc::Weight::Zero() || arc.weight == Arc::Weight::One())
            return arc;
        Arc newArc    = arc;
        newArc.weight = arc.weight.Value() * scale_;
        return newArc;
    }

    FstLib::MapFinalAction FinalAction() const {
        return FstLib::MAP_NO_SUPERFINAL;
    }

    FstLib::MapSymbolsAction InputSymbolsAction() const {
        return FstLib::MAP_COPY_SYMBOLS;
    }

    FstLib::MapSymbolsAction OutputSymbolsAction() const {
        return FstLib::MAP_COPY_SYMBOLS;
    }

    u64 Properties(u64 props) const {
        return props;
    }

private:
    f32 scale_;
};

template<class Arc>
void scaleWeights(FstLib::MutableFst<Arc>* fst, f32 scale) {
    FstLib::ArcMap(fst, ScaleMapper<Arc>(scale));
}

}  // namespace OpenFst

#endif
