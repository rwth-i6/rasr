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
#include "MixtureSetReader.hh"
#include <Core/Directory.hh>
#include <Core/FormatSet.hh>
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include "Module.hh"
namespace Mm {
class LogLinearMixtureSet : public Core::ReferenceCounted {};
}  // namespace Mm

using namespace Mm;

Core::ComponentFactory<MixtureSetReader::Reader, const std::string>* MixtureSetReader::reader_ = 0;

MixtureSetReader::MixtureSetReader(const Core::Configuration& c)
        : Core::Component(c) {
    if (!reader_) {
        reader_ = new Core::ComponentFactory<Reader, const std::string>;
        registerReader<FormatReader>(".pms");
        registerReader<FormatReader>(".gz");
    }
}

// ================================================================================

bool MixtureSetReader::FormatReader::read(const std::string& filename, MixtureSetRef& result) const {
    result = MixtureSetRef(new MixtureSet);
    if (Mm::Module::instance().formats().read(filename, *result)) {
        return true;
    }
    else {
        result.reset();
        return false;
    }
}

// ================================================================================

bool MixtureSetReader::MixtureSetEstimatorReader::read(const std::string& filename, MixtureSetRef& result) const {
    Core::Ref<AbstractMixtureSetEstimator> estimator(
            Mm::Module::instance().createMixtureSetEstimator(config));
    if (readMixtureSetEstimator(filename, *estimator))
        result = estimator->estimate();
    return result;
}

bool MixtureSetReader::MixtureSetEstimatorReader::readMixtureSetEstimator(
        const std::string& filename, AbstractMixtureSetEstimator& estimator) const {
    log("Loading mixture set estimator from file \"%s\" ...", filename.c_str());
    Core::BinaryInputStream bis(filename);
    if (!bis) {
        error("Failed to open \"%s\" for reading", filename.c_str());
        return false;
    }
    estimator.read(bis);
    if (!bis) {
        error("Failed to read mixture estimator from \"%s\".", filename.c_str());
        return false;
    }
    return true;
}
