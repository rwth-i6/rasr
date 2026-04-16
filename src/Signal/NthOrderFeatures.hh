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
#ifndef _SIGNAL_NTH_ORDER_FEATURES_HH
#define _SIGNAL_NTH_ORDER_FEATURES_HH

#include <Flow/Node.hh>
#include <Flow/Vector.hh>

namespace Signal {

/** NthOrderFeatures
 * Calculates nth order features.
 */
class NthOrderFeatures {
public:
    enum OrderType {
        none            = 0,
        zeroth          = 1,
        first           = 2,
        diagonalSecond  = 4,
        second          = 8,
        selectedSecond  = 16,
        diagonalThird   = 32,
        third           = 64,
        asymmetricThird = 128,
        diagonalFourth  = 256,
        diagonalFifth   = 512,
        diagonalSixth   = 1024,
        diagonalSeventh = 2048,
        diagonalEighth  = 4096,
        diagonalNinth   = 8192,
    };

protected:
    OrderType order_;
    size_t    outputSize_;
    class SecondsSelection {
    private:
        typedef std::unordered_map<u32, u32>     JValues;
        typedef std::unordered_map<u32, JValues> IValues;
        IValues                                  selection_;
        IValues::const_iterator                  currI_;
        JValues::const_iterator                  currJ_;
        u32                                      nSeconds_;

    public:
        SecondsSelection() {}
        bool load(const std::string&);
        bool setI(u32 i) {
            currI_ = selection_.find(i);
            currJ_ = currI_->second.end();
            return currI_ != selection_.end();
        }
        bool setJ(u32 j) {
            require(currI_ != selection_.end());
            currJ_ = currI_->second.find(j);
            return currJ_ != currI_->second.end();
        }
        u32 index() {
            require(currJ_ != currI_->second.end());
            return currJ_->second;
        }
        u32 size() const {
            return nSeconds_;
        }
    };
    SecondsSelection secondsSelection_;

protected:
    template<class T>
    struct power {
    private:
        T n_;

    public:
        power(T n)
                : n_(n) {}
        T operator()(T x) {
            return pow(x, n_);
        }
    };

public:
    NthOrderFeatures();

    void setOrder(OrderType order) {
        order_ = order;
    }
    void   setOutputSize(size_t size);
    size_t outputSize() const {
        return outputSize_;
    }
    bool loadSecondsSelection(const std::string& filename) {
        return (order_ & selectedSecond) ? secondsSelection_.load(filename) : true;
    }
    void apply(const std::vector<f32>& in, std::vector<f32>& out);
};

/** Nth Order Features Node
 *  Augments the incoming first order features
 *  with zeroth and higher order features.
 */
class NthOrderFeaturesNode : public Flow::SleeveNode {
public:
    static const Core::Choice          choiceOrderType;
    static const Core::ParameterString paramOrderType;
    static const Core::ParameterString paramSecondsSelectionFile;

protected:
    NthOrderFeatures* nthOrder_;

public:
    static std::string filterName() {
        return "signal-nth-order-features";
    }
    NthOrderFeaturesNode(const Core::Configuration& c);
    virtual ~NthOrderFeaturesNode();

    virtual bool              setParameter(const std::string& name, const std::string& value);
    virtual bool              configure();
    virtual bool              work(Flow::PortId p);
    virtual NthOrderFeatures* createNthOrderFeatures() {
        return new NthOrderFeatures;
    }
};
}  // namespace Signal

#endif  // _SIGNAL_NTH_ORDER_FEATURES_HH
