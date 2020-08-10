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
#ifndef PROJECTIONNODE_HH_
#define PROJECTIONNODE_HH_

#include <Flow/Node.hh>
#include <Flow/Vector.hh>

namespace Signal {

template<class T>
class ProjectionNode : public Flow::SleeveNode {
private:
    static const Core::ParameterString paramProjectionComponents;
    // components_[i] = -1 : discard component
    // components_[i] = j: j-th component of projected vector = i-th component of input vector
    std::vector<s32> components_;
    u32              nComponents_;
    std::string      componentsFilename_;

protected:
    virtual void loadComponents(const std::string& filename);

public:
    static std::string filterName() {
        return std::string("projection-") + Core::Type<T>::name;
    }

public:
    ProjectionNode(const Core::Configuration& c);
    virtual ~ProjectionNode();

    Flow::PortId getInput(const std::string& name) {
        return 0;
    }
    Flow::PortId getOutput(const std::string& name) {
        return 0;
    }
    virtual bool configure();
    virtual bool setParameter(const std::string& name, const std::string& value);
    virtual bool work(Flow::PortId p);
};

template<class T>
const Core::ParameterString ProjectionNode<T>::paramProjectionComponents(
        "components-file", "name of file to load");

template<class T>
ProjectionNode<T>::ProjectionNode(const Core::Configuration& c)
        : Component(c), SleeveNode(c) {
    log() << "Initializing projection";
    addInput(0);
    addOutput(0);
    log() << "loading components";
    loadComponents(paramProjectionComponents(c));
}

template<class T>
ProjectionNode<T>::~ProjectionNode() {}

template<class T>
bool ProjectionNode<T>::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    getInputAttributes(0, *attributes);
    if (!configureDatatype(attributes, Flow::Vector<T>::type()))
        return false;
    return putOutputAttributes(0, attributes);
}

template<class T>
bool ProjectionNode<T>::setParameter(const std::string& name, const std::string& value) {
    if (paramProjectionComponents.match(name)) {
        componentsFilename_ = value;
        loadComponents(componentsFilename_);
    }
    else
        return false;
    return true;
}

template<class T>
bool ProjectionNode<T>::work(Flow::PortId p) {
    Flow::DataPtr<Flow::Vector<T>> in;
    if (!getData(0, in))
        return putData(0, in.get());

    Flow::Vector<T>* out = new Flow::Vector<T>;

    //loop over components
    u32 cmp = 0;
    for (u32 i = 0; i < nComponents_; i++) {
        cmp = components_[i];
        out->push_back((*in)[cmp]);
    }
    out->setTimestamp(*in);
    return putData(0, out);
}

template<typename T>
void ProjectionNode<T>::loadComponents(const std::string& filename) {
    if (filename.empty()) {
        error() << "components filename is empty.";
    }
    else {
        components_.clear();
        nComponents_ = 0;
        s32 cmp      = 0;

        Core::TextInputStream tis(filename);
        if (!tis.good()) {
            error() << "failed to read from components file";
        }
        while (tis >> cmp) {
            if (nComponents_ >= 1) {
                require(components_.back() < cmp);
            }
            components_.push_back(cmp);
            nComponents_++;
        }
    }
}

}  // namespace Signal

#endif /* PROJECTIONNODE_HH_ */
