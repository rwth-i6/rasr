/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#ifndef _ONNX_MODULE_HH
#define _ONNX_MODULE_HH

#include <Core/Singleton.hh>

namespace Onnx {

class Module_ {
public:
    Module_();
    ~Module_() = default;
};

typedef Core::SingletonHolder<Module_> Module;

}  // namespace Onnx

#endif  // _ONNX_MODULE_HH
