#include "nrtypes_nr.h"

//#include "nrtypes_lib.h"

namespace Math { namespace Nr {

template <class _Arg, class _Result> struct FunctorBase {
    typedef _Arg ArgumentType;
    typedef _Result ResultType;
    
    virtual  ResultType operator()(const ArgumentType &)const =0;
    virtual ~FunctorBase(){};
};

template <class _Arg, class _Input ,class _Result> struct DerivativesBase {
    typedef _Arg ArgumentType;
    typedef _Input InputType;
    typedef _Result ResultType;
    
    virtual void  operator()(const ArgumentType &, const InputType&, ResultType& )=0;
    virtual ~DerivativesBase(){};
};


template <class _Input ,class _Result> struct GradientBase {
    typedef _Input InputType;
    typedef _Result ResultType;

    virtual void  operator()(const InputType&, ResultType& )const=0;
    virtual ~GradientBase(){};
};

} } // namespace Math::Nr
