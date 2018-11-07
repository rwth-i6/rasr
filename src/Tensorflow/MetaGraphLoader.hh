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
#ifndef _TENSORFLOW_META_GRAPH_LOADER_HH
#define _TENSORFLOW_META_GRAPH_LOADER_HH

#include "GraphLoader.hh"

namespace Tensorflow {

class MetaGraphLoader : public GraphLoader {
public:
    typedef GraphLoader Precursor;

    static Core::ParameterString paramMetaGraphFile;
    static Core::ParameterString paramSavedModelFile;

    MetaGraphLoader(Core::Configuration const& config);
    ~MetaGraphLoader() = default;

    virtual std::unique_ptr<Graph> load_graph();
    virtual void                   initialize(Session& session);

private:
    std::string meta_graph_file_;
    std::string saved_model_file_;

    std::string              restore_op_name_;
    std::string              restore_filename_tensor_name_;
    std::vector<std::string> variable_initializers_;
};

inline MetaGraphLoader::MetaGraphLoader(Core::Configuration const& config)
        : Core::Component(config), MetaGraphLoader::Precursor(config), meta_graph_file_(paramMetaGraphFile(config)), saved_model_file_(paramSavedModelFile(config)) {
}

}  // namespace Tensorflow

#endif /* _TENSORFLOW_META_GRAPH_LOADER_HH */
