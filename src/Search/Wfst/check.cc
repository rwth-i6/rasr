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
#include <Core/Application.hh>
#include <Core/Debug.hh>
#ifndef CMAKE_DISABLE_MODULE_HH
#include <Modules.hh>
#endif
#include <OpenFst/Count.hh>
#include <OpenFst/Types.hh>
#include <Search/Wfst/DynamicLmFst.hh>
#include <Search/Wfst/Lattice.hh>
#include <Search/Wfst/Network.hh>
#include <fst/arcsort.h>
#include <fst/compose.h>
#include <fst/determinize.h>

class TestApplication : public virtual Core::Application {
public:
    /**
     * Standard constructor for setting title.
     */
    TestApplication()
            : Core::Application() {
        setTitle("check");
    }

    std::string getUsage() const {
        return "test network\n";
    }

    int main(const std::vector<std::string>& arguments) {
        log("reading ") << arguments[0];
        Search::Wfst::Lattice *l = Search::Wfst::Lattice::Read(arguments[0]), det;
        Fsa::AutomatonCounts   c = OpenFst::count(*l);
        log("states: %d, arcs: %d", c.nStates_, c.nArcs_);
        FstLib::Determinize(*l, &det);
        log("writing ") << arguments[1];
        det.Write(arguments[1]);
        return 0;

        return EXIT_SUCCESS;
    }  //end main

protected:
};

APPLICATION(TestApplication)
