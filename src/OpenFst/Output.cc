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
#include "Output.hh"
#include <iostream>

namespace OpenFst
{

bool writeFsa(Fsa::ConstAutomatonRef f, const std::string &file)
{
    std::ofstream out(file.c_str());
    if (!out)
        return false;

    VectorFst *fst = convertFromFsa(f);
    bool r = fst->Write(out, FstLib::FstWriteOptions(file, true, true, true));
    delete fst;
    return r;
}

bool writeOpenFst(const Fsa::Resources &resources, Fsa::ConstAutomatonRef f,
                  std::ostream &o, Fsa::StoredComponents what, bool progress) {
    if (o) {
        VectorFst *fst = convertFromFsa(f);
        bool r = fst->Write(o, FstLib::FstWriteOptions("", true, true, true));
        delete fst;
        return r;
    }
    return true;
}


VectorFst* convertFromFsa(Fsa::ConstAutomatonRef f)
{
    return convertFromFsa<Fsa::Automaton, OpenFst::VectorFst>(f);
}


bool write(const VectorFst &fst, const std::string &filename) {
    std::ofstream o(filename.c_str());
    if (!o) return false;
    return fst.Write(o, FstLib::FstWriteOptions(filename, true, true, true));
}

}
