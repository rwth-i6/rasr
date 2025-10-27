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
#include <Fsa/Automaton.hh>

#include "FlfCore/Lattice.hh"

// ===========================================================================
// Application

class MyApplication : public Core::Application {
public:
    std::string getUsage() const {
        return "...";
    }

    int main(const std::vector<std::string>& arguments) {
        Flf::ConstSemiringRef semiring = Flf::Semiring::create(Fsa::SemiringTypeTropical, 2);
        Flf::ByteVector       v;
        {
            std::cout << semiring->describe(semiring->one(), Fsa::HintShowDetails) << std::endl;
            semiring->compress(v, semiring->one());
            Flf::ScoresRef s = semiring->create();
            s->set(0, 12.0);
            s->set(1, 4.0);
            std::cout << semiring->describe(s, Fsa::HintShowDetails) << std::endl;
            semiring->compress(v, s);
            std::cout << semiring->describe(semiring->invalid(), Fsa::HintShowDetails) << std::endl;
            semiring->compress(v, semiring->invalid());
            std::cout << semiring->describe(semiring->zero(), Fsa::HintShowDetails) << std::endl;
            semiring->compress(v, semiring->zero());
        }

        Flf::ByteVector::const_iterator vIt = v.begin();
        std::cout << semiring->describe(semiring->uncompress(vIt), Fsa::HintShowDetails) << std::endl;
        std::cout << semiring->describe(semiring->uncompress(vIt), Fsa::HintShowDetails) << std::endl;
        std::cout << semiring->describe(semiring->uncompress(vIt), Fsa::HintShowDetails) << std::endl;
        std::cout << semiring->describe(semiring->uncompress(vIt), Fsa::HintShowDetails) << std::endl;
        return 0;
    }
};

APPLICATION(MyApplication)
