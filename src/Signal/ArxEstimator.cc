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
#include "ArxEstimator.hh"

using namespace Signal;

// ArxEstimator
///////////////

void ArxEstimator::convolution() {
    //C_tilde and F_tilde does not contain the first (1) element !

    CF_tilde_.resize(F_tilde_.size() + C_tilde_.size());

    u8 t, tau;

    for (t = 0; t < F_tilde_.size() + C_tilde_.size(); t++)
        CF_tilde_[t] = 0.0;

    for (t = 0; t < F_tilde_.size(); t++) {
        CF_tilde_[t] += F_tilde_[t];
        for (tau = 0; tau < t && tau < C_tilde_.size(); tau++)
            CF_tilde_[t] += C_tilde_[tau] * F_tilde_[t - tau - 1];
    }

    for (t = F_tilde_.size(); t < F_tilde_.size() + C_tilde_.size(); t++) {
        CF_tilde_[t - F_tilde_.size()] += C_tilde_[t - F_tilde_.size()];
        for (tau = t - F_tilde_.size(); tau < t && tau < C_tilde_.size(); tau++)
            CF_tilde_[t] += C_tilde_[tau] * F_tilde_[t - tau - 1];
    }
}

bool ArxEstimator::prepare(const std::vector<_float>* u,
                           const std::vector<_float>* y,
                           const std::vector<_float>* y0) {
    u32 nr_sample = y->size();
    u32 i;

    const std::vector<_float>* u_work = u;
    const std::vector<_float>* y_work = y;

    if (D_tilde_.size() || CF_tilde_.size()) {
        if (u_ == 0 || y_ == 0 || y0_ == 0)
            return false;

        u_work = u_;
        y_work = y_;

        // D / FC * u[t]
        if (u) {
            u_work = u_;

            for (u32 t = 0; t < nr_sample; t++) {
                (*u_)[t] = (*u)[t];

                for (i = 1; i <= D_tilde_.size(); i++)
                    (*u_)[t] += D_tilde_[i - 1] * (t >= i ? (*u)[t - i] : 0.0);

                for (i = 1; i <= CF_tilde_.size(); i++)
                    (*u_)[t] -= CF_tilde_[i - 1] * (t >= i ? (*u_)[t - i] : 0.0);
            }
        }

        // D / C * y0[t]
        if (y0) {
            for (u32 t = 0; t < nr_sample; t++) {
                (*y0_)[t] = 0.0;

                //input for (D/AC)*y0
                for (i = t + 1; i <= F_tilde_.size(); i++)
                    (*y0_)[t] -= F_tilde_[i - 1] * (*y0)[F_tilde_.size() - (i - t)];

                for (i = 1; i <= D_tilde_.size(); i++)
                    (*y0_)[t] += D_tilde_[i - 1] * (t >= i ? (*y0_)[t - i] : 0.0);

                for (i = 1; i <= CF_tilde_.size(); i++)
                    (*y0_)[t] -= CF_tilde_[i - 1] * (t >= i ? (*y0_)[t - i] : 0.0);
            }
        }

        // D / FC * y[t]
        for (u32 t = 0; t < nr_sample; t++) {
            (*y_)[t] = (*y)[t];

            for (i = 1; i <= D_tilde_.size(); i++)
                (*y_)[t] += D_tilde_[i - 1] * (t >= i ? (*y)[t - i] : 0.0);

            for (i = 1; i <= CF_tilde_.size(); i++)
                (*y_)[t] -= CF_tilde_[i - 1] * (t >= i ? (*y_)[t - i] : 0.0);
        }
    }

    return least_squares_builder_.work(u_work, y_work, y0);
}

bool ArxEstimator::work(const std::vector<_float>* u,
                        const std::vector<_float>* y,
                        const std::vector<_float>* y0,
                        _float*                    estimation_error,
                        std::vector<_float>* B_tilde, std::vector<_float>* A_tilde) {
    if (y == 0 || (order_B_ > 0 && u == 0) || (y0 != 0 && y0->size() < F_tilde_.size()))
        return false;

    std::vector<_float> theta;

    if ((need_init_ && !init()) ||
        !prepare(u, y, y0) ||
        !least_squares_.work(estimation_error, B_tilde || A_tilde ? &theta : 0))
        return false;

    u8 i;
    u8 k = 0;
    if (order_B_ && B_tilde) {
        B_tilde->resize(order_B_);
        for (i = 0; i < order_B_; i++)
            (*B_tilde)[i] = theta[k++];
    }
    else
        k += order_B_;

    if (order_A_ && A_tilde) {
        A_tilde->resize(order_A_);
        for (i = 0; i < order_A_; i++)
            (*A_tilde)[i] = theta[k++];
    }

    return true;
}

void ArxEstimator::setNumberOfSamples(u32 nr_sample) {
    if (!u_)
        u_ = new std::vector<_float>;
    u_->resize(nr_sample);

    if (!y_)
        y_ = new std::vector<_float>;
    y_->resize(nr_sample);

    if (!y0_)
        y0_ = new std::vector<_float>;
    y0_->resize(nr_sample);
}

void ArxEstimator::freeBuffers() {
    if (u_) {
        delete u_;
        u_ = 0;
    }

    if (y_) {
        delete y_;
        y_ = 0;
    }

    if (y0_) {
        delete y0_;
        y0_ = 0;
    }
};
