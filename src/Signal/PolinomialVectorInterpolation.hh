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
#ifndef _SIGNAL_POLINOMIAL_VECTOR_INTERPOLATION_HH
#define _SIGNAL_POLINOMIAL_VECTOR_INTERPOLATION_HH

#include <Core/Utility.hh>
#include <Flow/Synchronization.hh>
#include <Flow/Vector.hh>
#include <Math/Lapack/Lapack.hh>
#include "SlidingWindow.hh"

namespace Signal {

/** PolinomialVectorInterpolation: creates new vectors at given target start-times
 *  by polinomial interpolation
 *
 *  Error messages:
 *    -there exists a target start-time outside of the input start-time interval
 */

class PolinomialVectorInterpolation {
public:
    typedef Flow::Time          Time;
    typedef Flow::Vector<f32>   Data;
    typedef Flow::DataPtr<Data> DataPointer;

private:
    Math::Lapack::Matrix<f64> A_;
    Math::Lapack::Matrix<f64> B_;
    Math::Lapack::Vector<int> pivotIndices_;

    std::string lastError_;

    SlidingWindow<DataPointer> slidingWindow_;

    bool equationSystemSolved_;

private:
    /** Seeks in the input stream until @param time is found.
     *
     *  At the beginning slidingWindow_ is filled to its maximum size.
     *  After end-of-stream the last elements in the slidingWindow_ are kept until reset() is called.
     *  After seeking slidingWindow_ contains contol points directly left and right from the @param time.
     */
    void seek(Time time);

    /** Checks if @param time does not lie outside of the input control points.
     *
     *  @return is false if
     *    -@param time is earlier than the very first input stream start-time or
     *	later then the very last input stream start-time.
     */
    bool checkInterpolationTime(Time time);

    /** Avoids solving of the linear equation system for @param time values
     *  equal to the start-time of one of the control points.
     *
     *  @return is true if @param time is equal to the start-time of one of the control points.
     *  If a control point is found, it is copied to @param out.
     */
    bool copyControlPoint(Time time, DataPointer& out) const;

    /** updates the size of linear equation system */
    void resize();

    /** calculates the interpolation parameters */
    void calculateParameters();

    /** creates a new element at @param time
     *
     * Start and end time of @param out are both set to @param time!
     */
    bool calculateOutput(Time time, DataPointer& out);

protected:
    /** override nextData to supply control points on demand */
    virtual bool nextData(DataPointer& dataPointer) = 0;

public:
    static std::string name() {
        return "signal-vector-polinomial-interpolation";
    }
    PolinomialVectorInterpolation();
    virtual ~PolinomialVectorInterpolation() {}

    /** @return is the vector created by interpolation at @param time
     *
     *  If @param time is found in the input stream:
     *    start-time and end-time are delivered un-changed
     *  Else:
     *     start-time and end-time of a predicted output are both set to @param time.
     *
     *  If @return false call lastError() to get a explanation.
     */
    bool work(const Flow::Timestamp& time, DataPointer& out);

    /** sets the order of interpolation polinom */
    void setOrder(u32 order);

    std::string lastError() const {
        return lastError_;
    }

    void reset();
};

}  // namespace Signal

#endif  // _SIGNAL_POLINOMIAL_VECTOR_INTERPOLATION_HH
