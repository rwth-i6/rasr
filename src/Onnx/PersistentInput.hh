/** Copyright 2026 RWTH Aachen University. All rights reserved.
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
#ifndef _ONNX_PERSISTENT_INPUT_HH
#define _ONNX_PERSISTENT_INPUT_HH

#include "Session.hh"
#include "Value.hh"

namespace Onnx {

/*
 * Input tensor that is passed unchanged to many session runs, e.g. the encoder states in every step of an attention decoder.
 * The tensor is stored once and each run gets a non-owning view of it, so it is not copied on the host per run.
 * For sessions on the CUDA execution provider, the tensor is additionally uploaded to the device once (at the first run
 * that needs it) and the view points to device memory, so onnxruntime doesn't copy it to the device in every run either.
 */
class PersistentInput {
public:
    PersistentInput();
    ~PersistentInput();

    PersistentInput(PersistentInput const&)            = delete;
    PersistentInput& operator=(PersistentInput const&) = delete;

    bool empty() const;

    // Replace the tensor by `value`, which must be in host memory
    void set(Value&& value);

    // Remove the tensor, the device memory stays allocated for the next one
    void clear();

    // Non-owning view as input for a run of `session`, valid until the next `set` or `clear`
    Value view(Session const& session);

private:
    Value  host_;
    void*  device_;          // copy of `host_` in device memory, if `uploaded_`
    size_t deviceCapacity_;  // in bytes
    int    deviceId_;        // CUDA device of `device_`
    bool   uploaded_;

    void upload();
};

}  // namespace Onnx

#endif  // _ONNX_PERSISTENT_INPUT_HH
