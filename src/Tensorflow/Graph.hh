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
#ifndef _TENSORFLOW_GRAPH_HH
#define _TENSORFLOW_GRAPH_HH

#include <tensorflow/core/framework/graph.pb.h>

#include <Core/Types.hh>

namespace Tensorflow {

namespace tf = tensorflow;

using DataType = tf::DataType;

struct Variable {
    std::string      name;
    std::string      initial_value_name;
    std::string      initializer_name;
    std::string      snapshot_name;
    DataType         type;
    std::vector<s64> shape;
};

class Session;
class GraphLoader;

class Graph {
public:
    friend class Session;
    friend class GraphLoader;

    Graph()          = default;
    virtual ~Graph() = default;

    void addLibrary(std::string const& l);
    void addInput(std::string const& i);
    void addUpdateOp(std::string const& uo);
    void addStateVar(std::string const& sv);
    void addVariable(Variable const& v);

    std::vector<std::string> const& libraries() const;
    std::vector<std::string> const& inputs() const;
    std::vector<std::string> const& update_ops() const;
    std::vector<std::string> const& state_vars() const;

    std::unordered_map<std::string, Variable> const& variables() const;

protected:
    std::vector<std::string> libraries_;
    std::vector<std::string> inputs_;
    std::vector<std::string> update_ops_;
    std::vector<std::string> state_vars_;

    std::unordered_map<std::string, Variable> variables_;

    tf::GraphDef graph_def_;

    void setGraphDef(tf::GraphDef const& graph_def);
};

inline void Graph::addLibrary(std::string const& l) {
    libraries_.push_back(l);
}

inline void Graph::addInput(std::string const& i) {
    inputs_.push_back(i);
}

inline void Graph::addUpdateOp(std::string const& uo) {
    update_ops_.push_back(uo);
}

inline void Graph::addStateVar(std::string const& sv) {
    state_vars_.push_back(sv);
}

inline void Graph::addVariable(Variable const& v) {
    variables_[v.name] = v;
}

inline std::vector<std::string> const& Graph::libraries() const {
    return libraries_;
}

inline std::vector<std::string> const& Graph::inputs() const {
    return inputs_;
}

inline std::vector<std::string> const& Graph::update_ops() const {
    return update_ops_;
}

inline std::vector<std::string> const& Graph::state_vars() const {
    return state_vars_;
}

inline std::unordered_map<std::string, Variable> const& Graph::variables() const {
    return variables_;
}

inline void Graph::setGraphDef(tf::GraphDef const& graph_def) {
    graph_def_ = graph_def;
}

}  // namespace Tensorflow

#endif  // _TENSORFLOW_GRAPH_HH
