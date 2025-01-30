/** Copyright 2025 RWTH Aachen University. All rights reserved.
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
#ifndef ENCODER_FACTORY_HH
#define ENCODER_FACTORY_HH

#include <Core/Choice.hh>
#include <Core/Configuration.hh>
#include <Core/Parameter.hh>
#include <Core/ReferenceCounting.hh>
#include <functional>
#include "Encoder.hh"

namespace Nn {

class EncoderFactory : public Core::ReferenceCounted {
private:
    Core::Choice choices_;

public:
    Core::ParameterChoice paramEncoderType;

    EncoderFactory();

    typedef std::function<Core::Ref<Encoder>(Core::Configuration const&)> CreationFunction;

    void               registerEncoder(const char* name, CreationFunction creationFunction);
    Core::Ref<Encoder> createEncoder(Core::Configuration const& config) const;

private:
    typedef std::vector<CreationFunction> Registry;

    Registry registry_;
};

}  // namespace Nn

#endif  // ENCODER_FACTORY_HH
