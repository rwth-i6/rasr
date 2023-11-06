#include <Am/Module.hh>
#include <Audio/Module.hh>
#include <Flf/Module.hh>
#include <Flow/Module.hh>
#include <Lm/Module.hh>
#include <Math/Module.hh>
#include <Mm/Module.hh>
#include <Modules.hh>
#include <Signal/Module.hh>
#include <Speech/Module.hh>
#ifdef MODULE_NN
#include <Nn/Module.hh>
#endif
#ifdef MODULE_ONNX
#include <Onnx/Module.hh>
#endif
#ifdef MODULE_TENSORFLOW
#include <Tensorflow/Module.hh>
#endif

extern "C" void initRASR() {
    INIT_MODULE(Flf);
    INIT_MODULE(Am);
    INIT_MODULE(Audio);
    INIT_MODULE(Flow);
    INIT_MODULE(Math);
    INIT_MODULE(Mm);
    INIT_MODULE(Lm);
    INIT_MODULE(Signal);
    INIT_MODULE(Speech);
#ifdef MODULE_NN
    INIT_MODULE(Nn);
#endif
#ifdef MODULE_ONNX
    INIT_MODULE(Onnx);
#endif
#ifdef MODULE_TENSORFLOW
    INIT_MODULE(Tensorflow);
#endif
}
