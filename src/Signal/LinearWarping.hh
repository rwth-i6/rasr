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
#ifndef _SIGNAL_LINEAR_WARPING_HH
#define _SIGNAL_LINEAR_WARPING_HH

#include <Core/Hash.hh>
#include <Flow/DataAdaptor.hh>
#include <Math/AnalyticFunction.hh>
#include "Warping.hh"

namespace Signal {

/** Two piece linear warping function node.
 *  Equal to generic warping node parmetrized for linear warping up the speed of changing warping factor.
 *  A warping function cache is used to change warping function faster.
 *  Thus different warping functions can be applied even for each time frame.
 */
class LinearWarpingNode : public WarpingNode {
    typedef WarpingNode Predecessor;

public:
    struct hash {
        static const f64 strechFactor;
        size_t           operator()(f64 x) const;
    };
    typedef std::unordered_map<f64, Warping*, hash> WarpingCache;
    typedef Flow::Time                              Time;

public:
    static Core::ParameterFloat paramWarpingFactor;
    static Core::ParameterFloat paramWarpingLimit;

private:
    Flow::Float64 warpingFactor_;
    f64           warpingLimit_;
    WarpingCache  warpingCache_;

private:
    void updateWarpingFactor(const Flow::Timestamp& featureTimeStamp);
    void setWarpingFactor(f64 warpingFactor) {
        warpingFactor_() = warpingFactor;
    }
    void setWarpingLimit(f64 warpingLimit);

    /** Searches for warping object in the cache and creates a new one if not found. */
    const Warping& warping();
    Warping*       createWarping();

    /** Removes Warping objects from the cache. */
    void clear();
    void reset();

protected:
    virtual void initWarping() {
        clear();
    }
    virtual void apply(const Flow::Vector<f32>& in, std::vector<f32>& out);

public:
    static std::string filterName() {
        return "signal-linear-warping";
    }
    LinearWarpingNode(const Core::Configuration& c);
    virtual ~LinearWarpingNode() {
        clear();
    }

    virtual bool         setParameter(const std::string& name, const std::string& value);
    virtual bool         configure();
    virtual Flow::PortId getInput(const std::string& name);
};

}  // namespace Signal

#endif  // _SIGNAL_LINEAR_WARPING_HH
