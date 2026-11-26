/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#include <Core/Application.hh>

#include "IOSpecification.hh"
#include "Session.hh"

const Core::ParameterInt paramNumFrames("num-frames", "number of timeframes", 1);
const Core::ParameterInt paramNumFeatures("num-features", "number of features", 1);

class TestApplication : public Core::Application {
public:
    std::string getUsage() const {
        return "short program to test Onnx integration\n";
    }

    int main(const std::vector<std::string>& arguments) {
        size_t num_frames   = paramNumFrames(config);
        size_t num_features = paramNumFeatures(config);

        Onnx::Session session(select("session"));

        std::vector<Onnx::IOSpecification> io_spec;
        io_spec.emplace_back(Onnx::IOSpecification{"features", Onnx::IODirection::INPUT, false, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::FLOAT}, {{-1, -1, static_cast<int64_t>(num_features)}}});
        io_spec.emplace_back(Onnx::IOSpecification{"features-size", Onnx::IODirection::INPUT, true, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::INT32}, {{-1}}});
        io_spec.emplace_back(Onnx::IOSpecification{"output", Onnx::IODirection::OUTPUT, false, {Onnx::ValueType::TENSOR}, {Onnx::ValueDataType::FLOAT}, {{-1, -1, -2}}});

        Onnx::IOMapping   mapping(select("io-map"), io_spec);
        Onnx::IOValidator validator(select("validator"));

        validator.validate(io_spec, mapping, session);

        std::vector<std::string> output_names;
        std::vector<Onnx::Value> outputs;

        std::vector<Math::FastMatrix<f32>> batch;
        size_t                             batch_size = 1ul;
        for (size_t b = 0ul; b < batch_size; b++) {
            batch.emplace_back(num_frames, num_features);
            for (size_t i = 0ul; i < num_frames; i++) {
                for (size_t j = 0ul; j < num_features; j++) {
                    batch.back()(i, j) = i * j;
                }
            }
        }

        std::vector<int32_t> seq_length;
        for (auto const& m : batch) {
            seq_length.push_back(m.nRows());
        }

        output_names.push_back(mapping.getOnnxName("output"));

        double total = 0.0;
        for (size_t i = 0ul; i < 20; i++) {
            // `Session::run` moves the values out of its inputs, so they have to be created anew for every run
            std::vector<std::pair<std::string, Onnx::Value>> inputs;
            inputs.emplace_back(mapping.getOnnxName("features"), Onnx::Value::create(batch));
            inputs.emplace_back(mapping.getOnnxName("features-size"), Onnx::Value::create(seq_length));

            auto start  = std::chrono::steady_clock::now();
            auto status = session.run(std::move(inputs), output_names, outputs);
            if (not status.ok()) {
                criticalError() << status.message();
            }
            auto                                      end      = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            total += duration.count();
            std::cerr << "run: " << duration.count() << "ms" << std::endl;
        }
        std::cerr << "total run: " << total << "ms" << std::endl;

        outputs[0].save<float>("output");

        return 0;
    }
};

APPLICATION(TestApplication)
