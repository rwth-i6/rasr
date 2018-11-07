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
#ifndef _SIGNAL_ARXESTIMATOR_HH
#define _SIGNAL_ARXESTIMATOR_HH

#include "Lse.hh"
#include "SegmentEstimator.hh"

namespace Signal
{

    // ArxEstimator
    //////////////////

    class ArxEstimator
    {
    private:
        typedef f32 _float;

        LeastSquaresBuilder<_float> least_squares_builder_;
        QrLeastSquares<_float> least_squares_;

        u8 order_B_;
        u8 order_A_;

        std::vector<_float> F_tilde_;
        std::vector<_float> C_tilde_;
        std::vector<_float> D_tilde_;
        std::vector<_float> CF_tilde_;

        std::vector<_float>* u_;
        std::vector<_float>* y_;
        std::vector<_float>* y0_;

        bool need_init_;

        bool init() { convolution(); return !(need_init_ = false); }
        void freeBuffers();

        bool prepare(const std::vector<_float>* u, const std::vector<_float>* y, const std::vector<_float>* y0);

        void convolution();

    public:
        ArxEstimator() :
            least_squares_builder_(least_squares_),
            order_B_(0), order_A_(0),
            F_tilde_(0), D_tilde_(0), CF_tilde_(0),
            u_(0), y_(0), y0_(0),
            need_init_(false) {};
        virtual ~ArxEstimator() { freeBuffers(); };

        void setOrder_B(u8 order_B) { least_squares_builder_.setOrder_B(order_B_ = order_B); };
        void setOrder_A(u8 order_A) { least_squares_builder_.setOrder_A(order_A_ = order_A); };

        void setNumberOfSamples(u32 nr_sample); //call is F or C or D is given!

        void setC(const std::vector<_float>& C_tilde) { C_tilde_ = C_tilde; need_init_ = true; };
        void setF(const std::vector<_float>& F_tilde) { F_tilde_ = F_tilde; need_init_ = true; };
        void setD(const std::vector<_float>& D_tilde) { D_tilde_ = D_tilde; };

        bool work(const std::vector<_float>* u, const std::vector<_float>* y, const std::vector<_float>* y0 = 0,
                  _float* estimation_error = 0,
                  std::vector<_float>* B_tilde = 0, std::vector<_float>* A_tilde = 0);

        void reset() { freeBuffers(); least_squares_.reset(); need_init_ = true; }
    };
}


#endif // _SIGNAL_ARXESTIMATOR_HH
