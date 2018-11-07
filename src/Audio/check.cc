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

#include <Audio/Module.hh>
#include <Core/Application.hh>
#include <Core/Parameter.hh>
#include <Flow/Module.hh>
#include <Flow/Network.hh>
#include <Flow/Registry.hh>

class TestApplication : public Core::Application {
public:
    virtual std::string getUsage() const {
        return "short program to test audio network\n";
    }

    TestApplication() {
        INIT_MODULE(Flow);
        INIT_MODULE(Audio);
        setTitle("check");
    }

    static const Core::ParameterString p_network;

    int main(const std::vector<std::string>& arguments) {
        Flow::Network net(select("network"), false);
        net.buildFromFile(p_network(config));
        net.go();

        return 0;
    }
};

const Core::ParameterString TestApplication::p_network(
        "network-file", "feature extraction network file");

APPLICATION(TestApplication)
