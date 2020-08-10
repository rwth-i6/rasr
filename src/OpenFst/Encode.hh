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
#ifndef _OPENFST_ENCODE_HH
#define _OPENFST_ENCODE_HH

#include <fst/encode.h>
#include "Types.hh"

#include "Types.hh"

namespace OpenFst {

/**
 * Behaves like EncodeMapper, except for epsilon arcs with trivial weight,
 * which are always mapped to label 0.
 *
 * Due to the lack of virtual functions in EncodeMapper, the additional
 * logic is added using the Decorator Pattern.
 */
template<class A>
class EpsilonEncodeMapper {
    typedef FstLib::EncodeMapper<A> Mapper;
    typedef FstLib::EncodeType      EncodeType;
    typedef typename A::Weight      Weight;

public:
    EpsilonEncodeMapper(uint32 flags, EncodeType type)
            : mapper_(flags, type) {}

    EpsilonEncodeMapper(const EpsilonEncodeMapper& mapper)
            : mapper_(mapper.mapper_) {}

    EpsilonEncodeMapper(const Mapper& mapper)
            : mapper_(mapper) {}

    EpsilonEncodeMapper(const EpsilonEncodeMapper& mapper, EncodeType type)
            : mapper_(mapper.mapper_, type) {}

    EpsilonEncodeMapper(const Mapper& mapper, EncodeType type)
            : mapper_(mapper, type) {}

    ~EpsilonEncodeMapper() {}

    A operator()(const A& arc) {
        A result = mapper_(arc);
        if (mapper_.Type() == FstLib::ENCODE && arc.nextstate != FstLib::kNoStateId &&
            arc.ilabel == 0 &&
            (!(mapper_.Flags() & FstLib::kEncodeWeights) || arc.weight == Weight::One()) &&
            (!(mapper_.Flags() & FstLib::kEncodeLabels) || arc.olabel == 0)) {
            result.ilabel = 0;
        }
        return result;
    }

    FstLib::MapFinalAction FinalAction() const {
        return mapper_.FinalAction();
    }

    FstLib::MapSymbolsAction InputSymbolsAction() const {
        return mapper_.InputSymbolsAction();
    }

    FstLib::MapSymbolsAction OutputSymbolsAction() const {
        return mapper_.OutputSymbolsAction();
    }

    uint64 Properties(uint64 props) {
        return mapper_.Properties(props);
    }

    const uint32 flags() const {
        return mapper_.flags();
    }
    const EncodeType type() const {
        return mapper_.type();
    }
    const FstLib::internal::EncodeTable<A>& table() const {
        return mapper_.table();
    }

    bool Write(std::ostream& strm, const string& source) {
        return mapper_.Write(strm, source);
    }

    bool Write(const string& filename) {
        return mapper_.Write(filename);
    }

    FstLib::SymbolTable* InputSymbols() const {
        return mapper_.InputSymbols();
    }

    FstLib::SymbolTable* OutputSymbols() const {
        return mapper_.OutputSymbols();
    }

    void SetInputSymbols(const FstLib::SymbolTable* syms) {
        mapper_.SetInputSymbols(syms);
    }

    void SetOutputSymbols(const FstLib::SymbolTable* syms) {
        mapper_.SetOutputSymbols(syms);
    }

private:
    void   operator=(const EpsilonEncodeMapper&);  // Disallow.
    Mapper mapper_;
};

}  // namespace OpenFst

#endif /* ENCODE_HH_ */
