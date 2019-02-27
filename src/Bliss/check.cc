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
// $Id$

#include <fstream>

#include <Core/Application.hh>
#include <Fsa/Output.hh>

#include "CorpusDescription.hh"
#include "EditDistance.hh"
#include "Lexicon.hh"
#include "Phonology.hh"

class TestApplication : public Core::Application {
public:
    virtual std::string getUsage() const {
        return "short program to test Bliss features\n";
    }

    TestApplication()
            : Core::Application() {
        setTitle("check");
    }

    int main(const std::vector<std::string>& arguments) {
        Bliss::LexiconRef lex = Bliss::Lexicon::create(select("lexicon"));

        {
            Core::XmlChannel ch(config, "dump-lexicon");
            if (ch.isOpen())
                lex->writeXml(ch);
        }

        return 0;
    }
};

APPLICATION(TestApplication)
