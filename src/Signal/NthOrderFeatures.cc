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
#include "NthOrderFeatures.hh"
#include <Core/CompressedStream.hh>

using namespace Signal;

bool NthOrderFeatures::SecondsSelection::load(
        const std::string& filename) {
    verify(!filename.empty());
    Core::CompressedInputStream cis(filename);
    std::string                 line;
    u32                         prevI = 0;
    u32                         prevJ = 0;
    nSeconds_                         = 0;
    while (cis) {
        std::getline(cis, line);
        Core::stripWhitespace(line);
        if (!line.empty() and (*line.c_str() != '#')) {
            std::vector<std::string> fields = Core::split(line, " ");
            require(fields.size() == 2);
            u32 currI, currJ;
            Core::strconv(fields[0], currI);
            Core::strconv(fields[1], currJ);
            require(currI <= currJ);
            require((prevI < currI) or ((prevI == currI) and (prevJ <= currJ)));
            selection_[currI][currJ] = nSeconds_;
            prevI                    = currI;
            prevJ                    = currJ;
            ++nSeconds_;
        }
    }
    return true;
}

//==================================================================================================
NthOrderFeatures::NthOrderFeatures()
        : order_(none),
          outputSize_(0) {}

void NthOrderFeatures::setOutputSize(size_t inputSize) {
    outputSize_ = 0;
    if (order_ & zeroth) {
        outputSize_ += 1;
    }
    if (order_ & first) {
        outputSize_ += inputSize;
    }
    if (order_ & diagonalSecond) {
        outputSize_ += inputSize;
    }
    if (order_ & second) {
        outputSize_ += ((inputSize + 1) * inputSize) / 2;
    }
    if (order_ & selectedSecond) {
        outputSize_ += secondsSelection_.size();
    }
    if (order_ & diagonalThird) {
        outputSize_ += inputSize;
    }
    if (order_ & third) {
        outputSize_ += ((inputSize + 2) * (inputSize + 1) * inputSize) / 6;
    }
    if (order_ & asymmetricThird) {
        outputSize_ += inputSize * ((inputSize + 1) * inputSize / 2);
    }
    if (order_ & diagonalFourth) {
        outputSize_ += inputSize;
    }
    if (order_ & diagonalFifth) {
        outputSize_ += inputSize;
    }
    if (order_ & diagonalSixth) {
        outputSize_ += inputSize;
    }
    if (order_ & diagonalSeventh) {
        outputSize_ += inputSize;
    }
    if (order_ & diagonalEighth) {
        outputSize_ += inputSize;
    }
    if (order_ & diagonalNinth) {
        outputSize_ += inputSize;
    }
}

void NthOrderFeatures::apply(const std::vector<f32>& in, std::vector<f32>& out) {
    require(out.size() == 0);
    if (order_ & first) {
        out.resize(out.size() + in.size());
        std::copy(in.rbegin(), in.rend(), out.rbegin());
    }
    if (order_ & diagonalSecond) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(2));
    }
    if (order_ & second) {
        for (size_t i = 0; i < in.size(); ++i) {
            for (size_t j = i; j < in.size(); ++j) {
                out.push_back(in[i] * in[j]);
            }
        }
    }
    if (order_ & selectedSecond) {
        for (size_t i = 0; i < in.size(); ++i) {
            if (secondsSelection_.setI(i)) {
                for (size_t j = i; j < in.size(); ++j) {
                    if (secondsSelection_.setJ(j)) {
                        out.push_back(in[i] * in[j]);
                    }
                }
            }
        }
    }
    if (order_ & diagonalThird) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(3));
    }
    if (order_ & third) {
        for (size_t i = 0; i < in.size(); ++i) {
            for (size_t j = i; j < in.size(); ++j) {
                for (size_t k = j; k < in.size(); ++k) {
                    out.push_back(in[i] * in[j] * in[k]);
                }
            }
        }
    }
    if (order_ & asymmetricThird) {
        for (size_t i = 0; i < in.size(); ++i) {
            for (size_t j = 0; j < in.size(); ++j) {
                for (size_t k = j; k < in.size(); ++k) {
                    out.push_back(in[i] * in[j] * in[k]);
                }
            }
        }
    }
    if (order_ & diagonalFourth) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(4));
    }
    if (order_ & diagonalFifth) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(5));
    }
    if (order_ & diagonalSixth) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(6));
    }
    if (order_ & diagonalSeventh) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(7));
    }
    if (order_ & diagonalEighth) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(8));
    }
    if (order_ & diagonalNinth) {
        out.resize(out.size() + in.size());
        std::transform(in.rbegin(), in.rend(), out.rbegin(), power<f32>(9));
    }
    if (order_ & zeroth) {
        out.push_back(1);
    }
    verify(out.size() == outputSize());
}

//==================================================================================================
const Core::Choice NthOrderFeaturesNode::choiceOrderType(
        "none", NthOrderFeatures::none,
        "zeroth", NthOrderFeatures::zeroth,
        "first", NthOrderFeatures::first,
        "diagonal-second", NthOrderFeatures::diagonalSecond,
        "full-second", NthOrderFeatures::second,
        "second", NthOrderFeatures::second,
        "selected-second", NthOrderFeatures::selectedSecond,
        "diagonal-third", NthOrderFeatures::diagonalThird,
        "third", NthOrderFeatures::third,
        "asymmetric-third", NthOrderFeatures::asymmetricThird,
        "diagonal-fourth", NthOrderFeatures::diagonalFourth,
        "diagonal-fifth", NthOrderFeatures::diagonalFifth,
        "diagonal-sixth", NthOrderFeatures::diagonalSixth,
        "diagonal-seventh", NthOrderFeatures::diagonalSeventh,
        "diagonal-eighth", NthOrderFeatures::diagonalEighth,
        "diagonal-ninth", NthOrderFeatures::diagonalNinth,
        Core::Choice::endMark());

const Core::ParameterString NthOrderFeaturesNode::paramOrderType(
        "order",
        "select nth order features",
        "none");

const Core::ParameterString NthOrderFeaturesNode::paramSecondsSelectionFile(
        "selection-file",
        "file to read second-order features selection from");

NthOrderFeaturesNode::NthOrderFeaturesNode(
        const Core::Configuration& c)
        : Component(c),
          SleeveNode(c),
          nthOrder_(0) {}

NthOrderFeaturesNode::~NthOrderFeaturesNode() {
    delete nthOrder_;
}

bool NthOrderFeaturesNode::setParameter(const std::string& name, const std::string& value) {
    if (paramOrderType.match(name)) {
        int                            order  = (int)NthOrderFeatures::none;
        const std::vector<std::string> fields = Core::split(value, "-and-");
        for (std::vector<std::string>::const_iterator it = fields.begin(); it != fields.end(); ++it) {
            order |= choiceOrderType[*it];
        }
        if (!nthOrder_) {
            nthOrder_ = createNthOrderFeatures();
            nthOrder_->setOrder((NthOrderFeatures::OrderType)order);
            nthOrder_->loadSecondsSelection(paramSecondsSelectionFile(config));
        }
    }
    else {
        return false;
    }
    return true;
}

bool NthOrderFeaturesNode::configure() {
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    getInputAttributes(0, *attributes);
    if (!configureDatatype(attributes, Flow::Vector<f32>::type()))
        return false;
    return putOutputAttributes(0, attributes);
}

bool NthOrderFeaturesNode::work(Flow::PortId p) {
    Flow::DataPtr<Flow::Vector<f32>> in;
    if (!getData(0, in)) {
        return putData(0, in.get());
    }
    verify(nthOrder_);
    nthOrder_->setOutputSize(in->size());
    Flow::Vector<f32>* out = new Flow::Vector<f32>;
    out->setTimestamp(*in);
    nthOrder_->apply(*in, *out);
    return putData(0, out);
}
