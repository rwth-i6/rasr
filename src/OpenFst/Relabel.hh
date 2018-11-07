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
#ifndef _OPENFST_RELABEL_HH
#define _OPENFST_RELABEL_HH

#include <utility>
#include <vector>
#include <fst/relabel.h>
#include <OpenFst/Types.hh>


namespace OpenFst {

typedef std::pair<Label, Label> LabelPair;

typedef std::vector<LabelPair> LabelMapping;


inline void Relabel(FstLib::MutableFst<Arc> *fst,
             const std::map<Label, Label> &ilabels,
             const std::map<Label, Label> &olabels)
{
    LabelMapping ipairs, opairs;
    for (std::map<Label,Label>::const_iterator l = ilabels.begin(); l != ilabels.end(); ++l)
        ipairs.push_back(LabelPair(l->first, l->second));
    for (std::map<Label,Label>::const_iterator l = olabels.begin(); l != olabels.end(); ++l)
        ipairs.push_back(LabelPair(l->first, l->second));
    FstLib::Relabel(fst, ipairs, opairs);
}


}

#endif // _OPENFST_RELABEL_HH
