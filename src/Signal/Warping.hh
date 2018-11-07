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
#ifndef _SIGNAL_WARPING_HH
#define _SIGNAL_WARPING_HH

#include "Node.hh"
#include <cmath>
#include <numeric>
#include <Core/Parameter.hh>
#include <Math/AnalyticFunction.hh>
#include <Flow/Vector.hh>

namespace Signal {


    // Warping
    //////////

    class Warping {
    public:
        typedef f32 Data;
        typedef f64 FloatIndex;
        typedef std::vector<FloatIndex> WarpingFunction;
        enum MergeType { AritmeticMean, SelectBegin };
        enum InterpolationType { KeepEnd, InsertZero, LinearInterpolation };
        static const f64 copyTolerance;
    private:
        /** InverseItem is base class for items in inverseWarpingFunction */
        struct InverseItem {
            /** warps an interval of @param in.
             * Typical warping kinds:
             *   -merge: merge an interval of @param in into one value,
             *   -interpolate: create a valuebetween two subsequent elements in @param in
             */
            virtual ~InverseItem() {}
            virtual Data apply(const std::vector<Data> &in) const = 0;
        };

        /** Copies the element at @param index_.
         *  Used instead of a coplex inverse-item if it resulted in copiing one single element.
         *  E.g.: interpolation with position 0 or 1 or merge of an interval [begin..begin+1).
         */
        class CopyInverseItem : public InverseItem {
            u32 index_;
        public:
            CopyInverseItem(u32 index) : index_(index) {}
            virtual ~CopyInverseItem() {}

            virtual Data apply(const std::vector<Data> &in) const {
                require_(index_ < in.size()); return in[index_];
            }
        };

        /** AritmeticMeanInverseItem merges the interval [begin .. end) by calculating its aritmetic mean
         */
        class AritmeticMeanInverseItem : public InverseItem {
            u32 begin_;
            u32 end_;
        public:
            AritmeticMeanInverseItem(u32 begin, u32 end) : begin_(begin), end_(end) {}
            virtual ~AritmeticMeanInverseItem() {}

            virtual Data apply(const std::vector<Data> &in) const {
                verify_(begin_ < end_ && end_ <= in.size());

                return std::accumulate(in.begin() + begin_, in.begin() + end_, (Data)0) /
                    Data(end_ - begin_);
            }
        };

        /** SelectBeginInverseItem merges the interval [begin .. end) by selecting its first element
         */
        struct SelectBeginInverseItem : public CopyInverseItem {
            SelectBeginInverseItem(u32 begin, u32 end) : CopyInverseItem(begin) {}
        };

        /** KeepEndInverseItem creates values of the interval (first .. end) at a relative position
         *  by copying the value at end.
         */
        struct KeepEndInverseItem : public CopyInverseItem {
            KeepEndInverseItem(u32 first, u32 last, f64 relativePosition) : CopyInverseItem(last) {}
        };

        /** InsertZeroInverseItem creates values of the interval (first .. end) at a relative position
         *  by inserting the value zero.
         */
        struct InsertZeroInverseItem : public InverseItem {
            InsertZeroInverseItem(u32 first, u32 last, f64 relativePosition) {}
            virtual ~InsertZeroInverseItem() {}

            virtual Data apply(const std::vector<Data> &in) const {
                verify_(!in.empty()); return 0;
            }
        };

        /** LinearInterpolationInverseItem creates values of the interval (first .. end)
         *  at a relative position by linear interpolation.
         */
        class LinearInterpolationInverseItem : public InverseItem {
            u32 leftIndex_;
            /** relative position in the interval [leftIndex_..leftIndex_+1] */
            Data relativePosition_;
        public:
            LinearInterpolationInverseItem(u32 first, u32 last, f64 relativePosition) {
                require(first < last);
                require(relativePosition > 0 && relativePosition < 1.0);
                f64 position = first + f64(last - first) * relativePosition;
                leftIndex_ = (u32)floor(position);
                relativePosition_ = position - leftIndex_;
            }
            virtual ~LinearInterpolationInverseItem() {}

            virtual Data apply(const std::vector<Data> &in) const {
                verify_(leftIndex_ + 1 < in.size());
                return in[leftIndex_] * (Data(1) - relativePosition_) +
                    in[leftIndex_ + 1] * relativePosition_;
            }
        };
    private:
        /** inverse of the warping function: contains commands (merge or interpolation)
         * how create the value of the warped function at a given warped index
         */
        std::vector<InverseItem*> inverseWarpingFunction_;
        size_t inputSize_;
    private:
        InverseItem* createMerger(MergeType type, u32 begin, u32 end) const;
        InverseItem* createInterpolator(InterpolationType type,
                                        u32 first, u32 last, f64 relativePosition) const;
        void deleteInverseItems();
        static bool equalRelativePosition(f64 x, f64 y);
    public:
        Warping();
        virtual ~Warping() { deleteInverseItems(); }

        /** @param warpingFunction maps indexes of the input vector to indexes in the warped vector.
         *  For more to @param mergeType and to @param InterpolationType @see InverseItem.
         */
        void setWarpingFunction(Math::UnaryAnalyticFunctionRef warpingFunction, size_t inputSize,
                                MergeType mergeType, InterpolationType interpolationType);
        /** @param inverseWarpingFunction contains indexes, i.e. maps warped indices
         *  to indices of the input vector.
         *
         *  For more to @param InterpolationType @see InverseItem.
         */
        void setInverseWarpingFunction(Math::UnaryAnalyticFunctionRef inverseWarpingFunction,
                                       size_t inputSize, InterpolationType interpolationType);

        void apply(const std::vector<Data> &in, std::vector<Data> &out) const;
    };


    // WarpingNode
    //////////////

    class WarpingNode : public virtual Flow::Node {
    public:
        static const Core::Choice choiceMergeType;
        static const Core::ParameterChoice paramMergeType;
        static const Core::Choice choiceInterpolationType;
        static const Core::ParameterChoice paramInterpolationType;
        static const Core::ParameterBool paramInterpolateOverWarpedAxis;
    private:
        bool needInit_;
    protected:
        void init(size_t inputSize);

        size_t inputSize_;

        f64 sampleRate_;
        void setSampleRate(f64 sampleRate);

        Warping::MergeType mergeType_;
        void setMergeType(const Warping::MergeType type);
        Warping::InterpolationType interpolationType_;
        void setInterpolationType(const Warping::InterpolationType type);
        bool interpolateOverWarpedAxis_;
        void setInterpolateOverWarpedAxis(bool InterpolateOverWarpedAxis);
    protected:
        void setNeedInit() { needInit_ = true; };
        void resetNeedInit() { needInit_ = false; };
        /** Override this function to perform own initialization.
         */
        virtual void initWarping() {}

        /** Override this function to perform the specific warping
         */
        virtual void apply(const Flow::Vector<f32> &in, std::vector<f32> &out) = 0;

        /** Performs configuration
         *  Steps:
         *    1) Retrieves attributes of own input port.
         *    2) Sets values of own members.
         *    3) Merges @param successorAttributes into its own attribute.
         *    4) Puts the final attributes into the output node.
         *  @return is result of putting the final attribute.
         */
        bool configure(const Flow::Attributes &successorAttributes);
    public:
        WarpingNode(const Core::Configuration &c);
        virtual ~WarpingNode() {}

        Flow::PortId getInput(const std::string &name) { return 0; }
        Flow::PortId getOutput(const std::string &name) { return 0; }
        virtual bool setParameter(const std::string &name, const std::string &value);
        virtual bool work(Flow::PortId p);
    };
}


#endif // _SIGNAL_WARPING_HH
