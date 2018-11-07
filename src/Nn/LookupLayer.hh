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
#ifndef _NN_LOOKUP_LAYER_HH
#define _NN_LOOKUP_LAYER_HH

// Neural Network Layer implementation

#include "LinearLayer.hh"

namespace Nn {

#include <vector>

/*
 * Lookup layer
 *
 * Expects inputs to be indices into the weight matrix, indices are converted from usual feature type
 * Backpropagation is not supported at them moment
 *
 */
template<typename T>
class LookupLayer : public virtual LinearLayer<T> {
    typedef LinearLayer<T> Precursor;
protected:
    typedef typename Types<T>::NnVector NnVector;
    typedef typename Types<T>::NnMatrix NnMatrix;

    using Precursor::isComputing_;
    using Precursor::hasBias_;
    using Precursor::bias_;
    using Precursor::weights_;
public:
    LookupLayer(const Core::Configuration &config);
    virtual ~LookupLayer();
public:
    // initialization methods
    virtual void setInputDimension(u32 stream, u32 size);
protected:
    virtual void _forward(const std::vector<NnMatrix*>& input, NnMatrix& output, bool reset);
    virtual void setParameters(const Math::Matrix<T>& parameters);
};

} // namespace Nn

#endif // _NN_LOOKUP_LAYER_HH
