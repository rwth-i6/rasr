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
#ifndef _SIGNAL_LSE_HH
#define _SIGNAL_LSE_HH

#include <Core/Types.hh>
#include <Math/Lapack/Lapack.hh>
#include <vector>
#include <stdio.h>

namespace Signal {
const f64 eps64 = 1.1103e-16 * 2.0;
const f32 eps32 = 5.9605e-8 * 2.0;

// Solves:
// Sum_t{ z[t]*(y[t]-y_hat[t]) } = 0 problem
// Where:
// y[t] is the measurement,
// y_hat = X[t]*theta is the linear predictor,
// X[t] = f(y[t-1],y[t-2],...,u[t-1],u[t-2],...,y_hat[t-1],y_hat[t-2]...) is the input of the predictor
// theta is the parameter vector of the predictor
// z[t] = f(y[t-1],y[t-2],...,u[t-1],u[t-2],...,y_hat[t-1],y_hat[t-2]...) is the correlation function

// Least Squares Problem if : z[y] = y[t]
// Instrumental Variables Problem if : z[y] = g(u[t])

// LeastSquares
////////////////////////

// Solves the min||y - X*theta|| problem

template<class T>
class LeastSquares {
public:
    class iterator {
    private:
        T*  buffer_;
        u8  row_size_;
        u32 column_size_;

    public:
        iterator(T* buffer = 0, u8 row_size = 0, u32 column_size = 0)
                : buffer_(buffer), row_size_(row_size), column_size_(column_size){};

        operator bool() {
            return buffer_ != 0;
        };

        T& operator()(const u32 row, const u8 column = 0) {
            return buffer_[column * column_size_ + row];
        }

        u8 rowSize() {
            return row_size_;
        };
        u32 columnSize() {
            return column_size_;
        };
    };

protected:
    u32 nr_sample_;
    u8  nr_parameter_;

    T* X_;
    T* y_;

    T*  work_;
    u32 lwork_;

    bool need_init_;

    virtual bool init() {
        return !(need_init_ = !reallocateBuffers());
    };
    virtual bool reallocateBuffers();
    virtual bool queryWorkspaceSize(u32& size) = 0;
    virtual void freeBuffers();

public:
    LeastSquares()
            : nr_sample_(0), nr_parameter_(0), X_(0), y_(0), work_(0), lwork_(0), need_init_(true){};
    virtual ~LeastSquares() {
        freeBuffers();
    };

    void setNumberOfSamples(u32 nr_sample) {
        if (nr_sample_ != nr_sample) {
            nr_sample_ = nr_sample;
            need_init_ = true;
        }
    }

    void setNumberOfParameters(u8 nr_parameter) {
        if (nr_parameter_ != nr_parameter) {
            nr_parameter_ = nr_parameter;
            need_init_    = true;
        }
    }

    virtual void reset() {
        freeBuffers();
        need_init_ = true;
    }

    iterator get_y() {
        return (need_init_ && !init()) ? iterator(0, 0, 0) : iterator(y_, 1, nr_sample_);
    };

    iterator get_X() {
        return (need_init_ && !init()) ? iterator(0, 0, 0) : iterator(X_, nr_parameter_, nr_sample_);
    }
};

template<class T>
bool LeastSquares<T>::reallocateBuffers() {
    if (X_)
        delete X_;
    X_ = new T[nr_parameter_ * nr_sample_];

    if (y_)
        delete y_;
    y_ = new T[nr_sample_];

    u32 size;
    if (!queryWorkspaceSize(size))
        return false;

    if (size != lwork_) {
        if (work_)
            delete work_;
        work_  = new T[size];
        lwork_ = size;
    }

    return true;
}

template<class T>
void LeastSquares<T>::freeBuffers() {
    if (X_) {
        delete X_;
        X_ = 0;
    }
    if (y_) {
        delete y_;
        y_ = 0;
    }
    if (work_) {
        delete work_;
        work_ = 0;
    }

    nr_sample_    = 0;
    nr_parameter_ = 0;

    lwork_ = 0;
}

// QrLeastSquares
//////////////////

template<class T>
class QrLeastSquares : public LeastSquares<T> {
public:  // constructor & destructor
    QrLeastSquares(){};
    virtual ~QrLeastSquares(){};

public:  // methods
    virtual bool work(T* estimation_error, std::vector<T>* theta);

protected:  // methods
    virtual bool queryWorkspaceSize(u32& size);
};

template<class T>
bool QrLeastSquares<T>::work(T* estimation_error, std::vector<T>* theta) {
    if (this->need_init_)
        return false;

    char trans = 'N';
    int  m     = this->nr_sample_;
    int  n     = this->nr_parameter_;
    int  nrhs  = 1;
    int  lwork = this->lwork_;
    int  info;

    Math::Lapack::gels(&trans, &m, &n, &nrhs, this->X_, &m, this->y_, &m, this->work_, &lwork, &info);

    if (info != 0)
        return false;

    if (theta) {
        theta->resize(this->nr_parameter_);
        for (u32 i = 0; i < this->nr_parameter_; i++)
            (*theta)[i] = this->y_[i];
    }

    if (estimation_error) {
        *estimation_error = 0.0;

        for (u32 i = this->nr_parameter_; i < this->nr_sample_; i++)
            *estimation_error += this->y_[i] * this->y_[i];
    }

    return true;
}

template<class T>
bool QrLeastSquares<T>::queryWorkspaceSize(u32& size) {
    char trans = 'N';
    int  m     = this->nr_sample_;
    int  n     = this->nr_parameter_;
    int  nrhs  = 1;
    int  lwork = -1;
    T    w;
    int  info;

    Math::Lapack::gels(&trans, &m, &n, &nrhs, this->X_, &m, this->y_, &m, &w, &lwork, &info);

    return ((info == 0) && ((size = (u32)w) != 0.0));
}

// SvdLeastSquares
//////////////////

template<class T>
class SvdLeastSquares : public LeastSquares<T> {
protected:  // attributes
    T* singularValues_;

public:  // constructor & destructor
    SvdLeastSquares()
            : singularValues_(0){};
    virtual ~SvdLeastSquares(){};

public:  // methods
         //singular values <= max(singular value) * tolerance are zerod
         //Effective rank is the number of non-zero singular values
    virtual bool work(const T tolerance, u8* effectiveRank, std::vector<T>* theta);

protected:  // methods
    virtual bool reallocateBuffers();
    virtual void freeBuffers();

    virtual bool queryWorkspaceSize(u32& size);
};

template<class T>
bool SvdLeastSquares<T>::work(const T tolerance, u8* effectiveRank, std::vector<T>* theta) {
    if (this->need_init_)
        return false;

    int m     = this->nr_sample_;
    int n     = this->nr_parameter_;
    int nrhs  = 1;
    T   rcond = tolerance;
    int rank;
    int lwork = this->lwork_;
    int info;

    gelss(&m, &n, &nrhs, this->X_, &m, this->y_, &m, this->singularValues_, &rcond, &rank, this->work_, &lwork, &info);

    if (info != 0)
        return false;

    if (theta) {
        theta->resize(this->nr_parameter_);

        for (u32 i = 0; i < this->nr_parameter_; i++)
            (*theta)[i] = this->y_[i];
    }

    //if (condition_2norm)
    //*condition_2norm = singularValues_[0] / singularValues_[order_B_ + order_A_ - 1];

    if (effectiveRank)
        *effectiveRank = rank;

    printf("Sing. Values (Treshold= %.6f): ", singularValues_[0] * tolerance);
    for (u8 i = 0; i < this->nr_parameter_; i++)
        printf("%.6f ", singularValues_[i]);
    printf("\n");

    return true;
}

template<class T>
bool SvdLeastSquares<T>::queryWorkspaceSize(u32& size) {
    int m    = this->nr_sample_;
    int n    = this->nr_parameter_;
    int nrhs = 1;
    T   rcond;
    int rank;
    int lwork = -1;
    T   w;
    int info;

    gelss(&m, &n, &nrhs, this->X_, &m, this->y_, &m, this->singularValues_, &rcond, &rank, &w, &lwork, &info);

    return ((info == 0) && ((size = (u32)w) != 0.0));
}

template<class T>
bool SvdLeastSquares<T>::reallocateBuffers() {
    if (singularValues_)
        delete singularValues_;

    singularValues_ = new T[this->nr_parameter_];

    return singularValues_ && LeastSquares<T>::reallocateBuffers();
}

template<class T>
void SvdLeastSquares<T>::freeBuffers() {
    if (singularValues_) {
        delete singularValues_;
        singularValues_ = 0;
    }
    LeastSquares<T>::freeBuffers();
}

// LeastSquaresBuilder
//////////////////////

// Builds:
// X matrix
// y vector

template<class T>
class LeastSquaresBuilder {
private:
    LeastSquares<T>& least_squares_;

    u8 order_B_;
    u8 order_A_;

    bool need_init_;

    bool init() {
        least_squares_.setNumberOfParameters(order_B_ + order_A_);
        return !(need_init_ = false);
    }

public:
    LeastSquaresBuilder(LeastSquares<T>& least_squares)
            : least_squares_(least_squares), order_B_(0), order_A_(0), need_init_(true){};

    void setOrder_B(u8 order_B) {
        if (order_B_ != order_B) {
            order_B_   = order_B;
            need_init_ = true;
        }
    }
    void setOrder_A(u8 order_A) {
        if (order_A_ != order_A) {
            order_A_   = order_A;
            need_init_ = true;
        }
    }

    bool work(const std::vector<T>* u, const std::vector<T>* y, const std::vector<T>* y0);
};

template<class T>
bool LeastSquaresBuilder<T>::work(const std::vector<T>* u, const std::vector<T>* y, const std::vector<T>* y0) {
    if (y == 0 || (u && y->size() != u->size()) || ((order_B_ > 0 && u == 0) && (y0 != 0 && y->size() != y0->size())))
        return false;

    if (need_init_ && !init())
        return false;

    u8  start_t   = std::max(order_B_, order_A_);
    u32 nr_sample = y->size();

    least_squares_.setNumberOfSamples(nr_sample - start_t);

    typename LeastSquares<T>::iterator ls_y = least_squares_.get_y();
    typename LeastSquares<T>::iterator ls_X = least_squares_.get_X();

    if (!ls_y || !ls_X)
        return false;

    u32 i, k;
    for (u32 t = start_t; t < nr_sample; t++) {
        ls_y(t - start_t) = (*y)[t] - (y0 ? (*y0)[t] : 0.0);
        //cout << "t= " << t << " " << ls_y(t - start_t) << "\t";

        k = 0;
        for (i = 1; i <= order_B_; i++) {
            ls_X(t - start_t, k) = (*u)[t - i];
            //cout << ls_X(t - start_t, k) << " ";

            k++;
        }

        for (i = 1; i <= order_A_; i++) {
            ls_X(t - start_t, k) = -(*y)[t - i];
            //cout << ls_X(t - start_t, k) << " ";

            k++;
        }

        //cout << endl;
    }

    return true;
}

// CovarianceBuilder
////////////////////

// Builds:
// R = X'X covyariance matrix
// f = X'y vector

template<class T>
class CovarianceBuilder {
private:
    LeastSquares<T>& least_squares_;

    u8 order_B_;
    u8 order_A_;

    typename LeastSquares<T>::iterator X_;
    typename LeastSquares<T>::iterator y_;

    T& Ruu(u8 row, u8 column) {
        return X_(row, column);
    };
    T& Ruy(u8 row, u8 column) {
        return X_(row, column + order_B_);
    };
    T& Ryu(u8 row, u8 column) {
        return X_(row + order_B_, column);
    };
    T& Ryy(u8 row, u8 column) {
        return X_(row + order_B_, column + order_B_);
    };

    T& fuy(u8 row) {
        return y_(row);
    };
    T& fyy(u8 row) {
        return y_(row + order_B_);
    };

    bool need_init_;

    bool init() {
        least_squares_.setNumberOfParameters(order_B_ + order_A_);
        least_squares_.setNumberOfSamples(order_A_ + order_B_);
        need_init_ = !((X_ = least_squares_.get_X()) && (y_ = least_squares_.get_y()));
        return !need_init_;
    }

public:
    CovarianceBuilder(LeastSquares<T>& least_squares)
            : least_squares_(least_squares), order_B_(0), order_A_(0), need_init_(true) {}

    void setOrder_B(u8 order_B) {
        if (order_B_ != order_B) {
            order_B_   = order_B;
            need_init_ = true;
        }
    }
    void setOrder_A(u8 order_A) {
        if (order_A_ != order_A) {
            order_A_   = order_A;
            need_init_ = true;
        }
    }

    bool work(const std::vector<T>* u, const std::vector<T>* y, const std::vector<T>* y0);
};

template<class T>
bool CovarianceBuilder<T>::work(const std::vector<T>* u, const std::vector<T>* y, const std::vector<T>* y0) {
    if (y == 0 || (u != 0 && y->size() != u->size()) || (order_B_ > 0 && u == 0) ||
        (y0 != 0 && y->size() != y0->size()))
        return false;

    if (need_init_ && !init())
        return false;

    u8  start_t   = std::max(order_B_, order_A_);
    u32 nr_sample = y->size();

    u8 i, k;
    for (i = 0; i < order_B_; i++) {
        // Ruu
        for (k = i; k < order_B_; k++) {
            Ruu(i, k) = 0.0;
            for (u32 t = start_t - 1; t < nr_sample - 1; t++)
                Ruu(i, k) += (*u)[t - i] * (*u)[t - k];

            Ruu(k, i) = Ruu(i, k);
        }

        // Ruy = Ryu'
        for (k = 0; k < order_A_; k++) {
            Ruy(i, k) = 0.0;
            for (u32 t = start_t - 1; t < nr_sample - 1; t++)
                Ruy(i, k) += (*u)[t - i] * (-(*y)[t - k]);

            Ryu(k, i) = Ruy(i, k);
        }

        // fuy
        fuy(i) = 0.0;
        for (u32 t = start_t - 1; t < nr_sample - 1; t++)
            fuy(i) += (*u)[t - i] * ((*y)[t + 1] - (y0 ? (*y0)[t + 1] : 0.0));
    }

    for (i = 0; i < order_A_; i++) {
        // Ryy
        for (k = i; k < order_A_; k++) {
            Ryy(i, k) = 0.0;
            for (u32 t = start_t - 1; t < nr_sample - 1; t++)
                Ryy(i, k) += (*y)[t - i] * (-(*y)[t - k]);

            Ryy(k, i) = Ryy(i, k);
        }

        // fyy
        fyy(i) = 0.0;
        for (u32 t = start_t - 1; t < nr_sample - 1; t++)
            fyy(i) += (*y)[t - i] * ((*y)[t + 1] - (y0 ? (*y0)[t + 1] : 0.0));
    }

    return true;
}

// InstrumentalVariablesBuilder
///////////////////////////////

// Builds:
// R = Z'X matrix
// f = Z'y vector

template<class T>
class InstrumentalVariablesBuilder {
private:
    LeastSquares<T>& least_squares_;

    u8 order_B_;
    u8 order_A_;

    typename LeastSquares<T>::iterator X_;
    typename LeastSquares<T>::iterator y_;

    T& Ruu(u8 row, u8 column) {
        return X_(row, column);
    };
    T& Ruy(u8 row, u8 column) {
        return X_(row, column + order_B_);
    };
    T& Rzu(u8 row, u8 column) {
        return X_(row + order_B_, column);
    };
    T& Rzy(u8 row, u8 column) {
        return X_(row + order_B_, column + order_B_);
    };

    T& fuy(u8 row) {
        return y_(row);
    };
    T& fzy(u8 row) {
        return y_(row + order_B_);
    };

    bool need_init_;

    bool init() {
        least_squares_.setNumberOfParameters(order_B_ + order_A_);
        least_squares_.setNumberOfSamples(order_A_ + order_B_);
        need_init_ = !((X_ = least_squares_.get_X()) && (y_ = least_squares_.get_y()));
        return !need_init_;
    }

public:
    InstrumentalVariablesBuilder(LeastSquares<T>& least_squares)
            : least_squares_(least_squares), order_B_(0), order_A_(0), need_init_(true) {}

    void setOrder_B(u8 order_B) {
        if (order_B_ != order_B) {
            order_B_   = order_B;
            need_init_ = true;
        }
    }
    void setOrder_A(u8 order_A) {
        if (order_A_ != order_A) {
            order_A_   = order_A;
            need_init_ = true;
        }
    }

    bool work(const std::vector<T>* u, const std::vector<T>* y, const std::vector<T>* z);
};

template<class T>
bool InstrumentalVariablesBuilder<T>::work(const std::vector<T>* u, const std::vector<T>* y, const std::vector<T>* z) {
    if (y == 0 || (u != 0 && y->size() != u->size()) || (this->order_B > 0 && u == 0) && (z != 0 && y->size() != z->size()))
        return false;

    if (need_init_ && !init())
        return false;

    u8  start_t   = std::max(order_B_, order_A_);
    u32 nr_sample = y->size();

    u8 i, k;
    for (i = 0; i < order_B_; i++) {
        // Ruu
        for (k = i; k < order_B_; k++) {
            Ruu(i, k) = 0.0;
            for (u32 t = start_t - 1; t < nr_sample - 1; t++)
                Ruu(i, k) += (*u)[t - i] * (*u)[t - k];

            Ruu(k, i) = Ruu(i, k);
        }

        // Ruy
        for (k = 0; k < order_A_; k++) {
            Ruy(i, k) = 0.0;
            for (u32 t = start_t - 1; t < nr_sample - 1; t++)
                Ruy(i, k) += (*u)[t - i] * (-(*y)[t - k]);
        }

        // fuy
        fuy(i) = 0.0;
        for (u32 t = start_t - 1; t < nr_sample - 1; t++)
            fuy(i) += (*u)[t - i] * (*y)[t + 1];
    }

    for (i = 0; i < order_A_; i++) {
        // Rzu
        for (k = 0; k < order_B_; k++) {
            Rzu(i, k) = 0.0;
            for (u32 t = start_t - 1; t < nr_sample - 1; t++)
                Rzu(i, k) += (*z)[t - i] * (*u)[t - k];
        }

        // Rzy
        for (k = 0; k < order_A_; k++) {
            Rzy(i, k) = 0.0;
            for (u32 t = start_t - 1; t < nr_sample - 1; t++)
                Rzy(i, k) += (*z)[t - i] * (-(*y)[t - k]);
        }

        // fzy
        fzy(i) = 0.0;
        for (u32 t = start_t - 1; t < nr_sample - 1; t++)
            fzy(i) += (*z)[t - i] * (*y)[t + 1];
    }

    return true;
}
}  // namespace Signal

#endif  // _SIGNAL_LSE_HH
