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
#ifndef _TENSORFLOW_VANILLA_GRAPH_LOADER_HH
#define _TENSORFLOW_VANILLA_GRAPH_LOADER_HH

#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/public/session.h>
#include "GraphLoader.hh"

namespace Tensorflow {

class VanillaGraphLoader : public GraphLoader {
public:
    typedef GraphLoader Precursor;

    static Core::ParameterString paramSavedModelDir;

    VanillaGraphLoader(Core::Configuration const& config);
    ~VanillaGraphLoader() = default;

    virtual std::unique_ptr<Graph> load_graph();
    virtual void                   initialize(Session& session);

private:
    std::string saved_model_dir_;
    tf::SavedModelBundle bundle;
};

inline VanillaGraphLoader::VanillaGraphLoader(Core::Configuration const& config)
        : Core::Component(config), VanillaGraphLoader::Precursor(config), saved_model_dir_(paramSavedModelDir(config)) {
}

}  // namespace Tensorflow

#endif  // _TENSORFLOW_VANILLA_GRAPH_LOADER_HH
