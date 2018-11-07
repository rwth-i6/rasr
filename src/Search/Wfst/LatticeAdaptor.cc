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
#include <Search/Wfst/LatticeAdaptor.hh>
#include <Search/Wfst/Lattice.hh>
#include <Search/Wfst/Types.hh>
#include <Search/LatticeHandler.hh>
#include <Lattice/Lattice.hh>
#include <OpenFst/FstMapper.hh>
#include <Fsa/Output.hh>

using namespace Search::Wfst;

WfstLatticeAdaptor::~WfstLatticeAdaptor()
{
    delete l_;
}

bool WfstLatticeAdaptor::write(const std::string &id, Search::LatticeHandler *handler) const
{
    return handler->write(id, *this);
}

WfstLatticeAdaptor::WordLatticeRef  WfstLatticeAdaptor::wordLattice(const Search::LatticeHandler *handler) const
{
    return handler->convert(*this);
}
