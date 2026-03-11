#include "nnpf.h"

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset13.hpp"

#include <fstream>
#include <chrono>
#include <thread>
#include <csignal>
#include <cassert>
#include <cstdlib>


template <typename T>
inline T clamp(T a, T min, T max)
{
    if (a < min)
        return min;
    if (a > max)
        return max;
    return a;
}

// Escape shell argument to prevent command injection
static std::string escapeShellArg(const std::string &arg)
{
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    escaped += '\'';
    for (char c : arg)
    {
        if (c == '\'')
            escaped += "'\\''"; // Replace ' with '\''
        else
            escaped += c;
    }
    escaped += '\'';
    return escaped;
}

const char *nnpf_version()
{
    return "1";
}

// works with "speedscope" web UI
struct Profiler
{
    std::ofstream s;
    typedef std::chrono::high_resolution_clock Clock;
    Clock::time_point time;
    void log(const char *what)
    {
        const auto now = std::chrono::high_resolution_clock::now();
        if (what)
        {
            typedef std::chrono::duration<int, std::ratio<1, 1000000>> Microseconds;
            const auto duration = std::chrono::duration_cast<Microseconds>(now - time).count();
            s << what << " " << duration << "\n";
        }
        time = now;
    }
};

struct Nnpf
{
    int width, height;

    int inferenceWidth;
    int inferenceHeight;
    int luminanceInferenceWidth;
    int luminanceInferenceHeight;

    Profiler profiler;

    std::string updatedModelPath;

    Nnpf(const char *model_path, int width, int height, const char *weights_update_path)
        : width(width),
          height(height)
    {
        if (false)
            profiler.s.open("profile.txt");

        assert(width % 2 == 0);
        assert(height % 2 == 0);

        const char *model_override = std::getenv("NNPF_MODEL");
        if (model_override)
        {
            model_path = model_override;
            std::cout << "Overriding model path with environment variable NNPF_MODEL: " << model_path << "\n";
            weights_update_path = nullptr; // Ignore weights update if model is overridden, to avoid confusion
        }

        // If weights_update_path is provided, run external script to generate updated model
        const char *actualModelPath = model_path;
        if (!model_override && weights_update_path && weights_update_path[0] != '\0')
        {
            // Generate output path by replacing extension with _updated.onnx
            std::string basePath(model_path);
            size_t extPos = basePath.rfind('.');
            if (extPos != std::string::npos)
                updatedModelPath = basePath.substr(0, extPos) + "_updated.onnx";
            else
                updatedModelPath = basePath + "_updated.onnx";

            // Build and execute the command with proper escaping
            std::string cmd = "python3 decode_weights_update.py";
            cmd += " --base_model " + escapeShellArg(model_path);
            cmd += " --bitstream " + escapeShellArg(weights_update_path);
            cmd += " --output_model " + escapeShellArg(updatedModelPath);
            cmd += " > decode_weights_update.log 2>&1"; // Redirect stdout and stderr

            std::cout << "Running weights update command:\n";
            std::cout << cmd << "\n";
            std::cout.flush();

            int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                std::cerr << "ERROR: Failed to decode weights update (exit code " << ret << ")\n";
                std::cerr << "Check decode_weights_update.log for details\n";
                std::raise(SIGTERM);
            }

            std::cout << "Weights update applied, using model: " << updatedModelPath << "\n";
            actualModelPath = updatedModelPath.c_str();
        }

        const int hsplit = 8;
        const int vsplit = 1;

        inferenceWidth = width / (2 * hsplit);
        inferenceHeight = height / (2 * vsplit);


        printf("loading %s\n", actualModelPath);

        {
            // Load model using OpenVINO
            modelOv = core.read_model(actualModelPath);
            
            std::cout << "Device info:\n";
            std::cout << core.get_versions(device) << "\n";
            std::cout.flush();

            // Get input parameter and output result from the model
            auto parameters = modelOv->get_parameters();
            auto results = modelOv->get_results();
            
            if (parameters.empty() || results.empty())
            {
                std::cerr << "Model must have at least one input and one output\n";
                std::raise(SIGTERM);
            }
            
            inputTensor = parameters[0];
            result = results[0];
            
            // Compile the model
            compiled_model = core.compile_model(modelOv, device, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
            const auto optimalNumber = compiled_model.get_property(ov::optimal_number_of_infer_requests);
            std::cout << "OpenVINO: optimal_number_of_infer_requests = " << optimalNumber << "\n";
            std::cout.flush();

            auto compiled_inputs = compiled_model.inputs();
            auto compiled_outputs = compiled_model.outputs();
            auto input_shape = compiled_inputs[0].get_shape();
            auto output_shape = compiled_outputs[0].get_shape();
            std::cout << "Input shape: " << input_shape << "\n";
            std::cout << "Output shape: " << output_shape << "\n";

            inferenceWidth = output_shape[3];
            inferenceHeight = output_shape[2];
        }

        luminanceInferenceWidth = 2 * inferenceWidth;
        luminanceInferenceHeight = 2 * inferenceHeight;

        for (int ab = 0; ab < 2; ++ab)
        {
            const auto patchesPerRow = (width + luminanceInferenceWidth - 1) / luminanceInferenceWidth;
            const auto patchesPerColumn = (height + luminanceInferenceHeight - 1) / luminanceInferenceHeight;
            transfers[ab].inferRequests.resize(patchesPerRow * patchesPerColumn);

            for (auto &inferRequest : transfers[ab].inferRequests)
                inferRequest = compiled_model.create_infer_request();

            {
                int w = width + 2 * Transfer::padX;
                int h = height + 2 * Transfer::padY;
                transfers[ab].planes[0].resize(w * h);
            }
            {
                int w = width / 2 + 2 * Transfer::padX;
                int h = height / 2 + 2 * Transfer::padY;
                transfers[ab].planes[1].resize(w * h);
                std::fill(transfers[ab].planes[1].begin(), transfers[ab].planes[1].end(), 512);
            }
            {
                int w = width / 2 + 2 * Transfer::padX;
                int h = height / 2 + 2 * Transfer::padY;
                transfers[ab].planes[2].resize(w * h);
                std::fill(transfers[ab].planes[2].begin(), transfers[ab].planes[2].end(), 512);
            }
        }

        profiler.log(nullptr);
    }

    ov::Core core;
    const char *device = getenv("NNPF_OPENVINO_DEVICE") ? getenv("NNPF_OPENVINO_DEVICE") : "CPU";
    std::shared_ptr<ov::Model> modelOv;
    ov::CompiledModel compiled_model;
    std::shared_ptr<ov::opset13::Parameter> inputTensor;
    std::shared_ptr<ov::opset13::Result> result;

    struct Transfer
    {
        static constexpr int padX = 32;
        static constexpr int padY = 132;

        typedef ov::InferRequest InferRequest;
        std::vector<InferRequest> inferRequests;
        std::vector<uint16_t> planes[3];
        float strength;
    };

    Transfer transfers[2];
    int ab = 0;

    static inline uint16_t convert(uint16_t &p, float delta)
    {
        int x = p;
        x += delta;
        if (x < 0)
            return 0;
        if (x > 1023)
            return 1023;
        return x;
    }

    void convertPatchesOpenVINO(int width, int height, Transfer *transferIn, Transfer *transferOut)
    {
        auto &inferRequestsIn = transferIn->inferRequests;
        auto &inferRequestsOut = transferOut->inferRequests;

        ptrdiff_t yuvStrides[3];
        yuvStrides[0] = width + 2 * Transfer::padX;
        yuvStrides[1] = width / 2 + 2 * Transfer::padX;
        yuvStrides[2] = width / 2 + 2 * Transfer::padX;

        uint16_t *yuvOutPointers[3];
        yuvOutPointers[0] = transferOut->planes[0].data() + Transfer::padY * yuvStrides[0] + Transfer::padX;
        yuvOutPointers[1] = transferOut->planes[1].data() + Transfer::padY * yuvStrides[1] + Transfer::padX;
        yuvOutPointers[2] = transferOut->planes[2].data() + Transfer::padY * yuvStrides[2] + Transfer::padX;

        uint16_t *yuvInPointers[3];
        yuvInPointers[0] = transferIn->planes[0].data() + Transfer::padY * yuvStrides[0] + Transfer::padX;
        yuvInPointers[1] = transferIn->planes[1].data() + Transfer::padY * yuvStrides[1] + Transfer::padX;
        yuvInPointers[2] = transferIn->planes[2].data() + Transfer::padY * yuvStrides[2] + Transfer::padX;

        const auto patchesPerRow = (width + luminanceInferenceWidth - 1) / luminanceInferenceWidth;
        const auto patchesPerColumn = (height + luminanceInferenceHeight - 1) / luminanceInferenceHeight;

        static constexpr auto bitDepth = 10;
        static constexpr auto multiplier = 1.f / ((1 << bitDepth) - 1);

        // Start async inference for all patches
        for (int patchY = 0; patchY < patchesPerColumn; ++patchY)
            for (int patchX = 0; patchX < patchesPerRow; ++patchX)
            {
                auto &inferRequest = inferRequestsIn[patchX + patchY * patchesPerRow];
                ov::Tensor input_tensor = inferRequest.get_input_tensor();
                auto input_shape = input_tensor.get_shape();

                // Expected input shape: [1][channels][height][width]
                float *inputData = input_tensor.data<float>();
                const auto channels = input_shape[1];
                const auto height_dim = input_shape[2];
                const auto width_dim = input_shape[3];
                const size_t spatial_size = height_dim * width_dim;

                for (size_t in_y = 0; in_y < height_dim; ++in_y)
                    for (size_t in_x = 0; in_x < width_dim; ++in_x)
                    {
                        const auto x = inferenceWidth * patchX + in_x - 4;
                        const auto y = inferenceHeight * patchY + in_y - 4;

                        // Access pattern for [1][C][H][W]: data[c * (H*W) + y * W + x]
                        const size_t pixel_offset = in_y * width_dim + in_x;

                        inputData[0 * spatial_size + pixel_offset] = multiplier * yuvInPointers[0][2 * x + 0 + (2 * y + 0) * yuvStrides[0]];
                        inputData[1 * spatial_size + pixel_offset] = multiplier * yuvInPointers[0][2 * x + 1 + (2 * y + 0) * yuvStrides[0]];
                        inputData[2 * spatial_size + pixel_offset] = multiplier * yuvInPointers[0][2 * x + 0 + (2 * y + 1) * yuvStrides[0]];
                        inputData[3 * spatial_size + pixel_offset] = multiplier * yuvInPointers[0][2 * x + 1 + (2 * y + 1) * yuvStrides[0]];
                        inputData[4 * spatial_size + pixel_offset] = multiplier * yuvInPointers[1][x + y * yuvStrides[1]];
                        inputData[5 * spatial_size + pixel_offset] = multiplier * yuvInPointers[2][x + y * yuvStrides[2]];
                        inputData[6 * spatial_size + pixel_offset] = transferIn->strength;
                        for (size_t c = 7; c < channels; ++c)
                            inputData[c * spatial_size + pixel_offset] = 0.f;
                    }

                profiler.log("caller;process_picture;convertPatches;copy_input");
                inferRequest.start_async();
                profiler.log("caller;process_picture;convertPatches;start_async");
            }

        static const auto g = std::getenv("NNPF_GAIN");
        static const float gain = g ? std::atof(g) : 1.f;

        static const char *s = std::getenv("NNPF_SYNCHRONOUS");
        static const bool synchronousInference = s ? std::atoi(s) == 1 : true;

        // In synchronous mode, wait for current frame's inference
        // In async mode, wait for previous frame's inference (which was started in the previous call)
        auto &inferRequestsToWait = synchronousInference ? inferRequestsIn : inferRequestsOut;

        for (int patchY = 0; patchY < patchesPerColumn; ++patchY)
            for (int patchX = 0; patchX < patchesPerRow; ++patchX)
            {
                auto &inferRequest = inferRequestsToWait[patchX + patchY * patchesPerRow];

                profiler.log("caller;process_picture;convertPatches");
                inferRequest.wait();
                profiler.log("caller;process_picture;convertPatches;wait");
            }

        for (int patchY = 0; patchY < patchesPerColumn; ++patchY)
            for (int patchX = 0; patchX < patchesPerRow; ++patchX)
            {
                auto &inferRequest = inferRequestsToWait[patchX + patchY * patchesPerRow];
                const ov::Tensor outputOv = inferRequest.get_output_tensor();
                auto output_shape = outputOv.get_shape();

                // Expected output shape: [1][6][height][width]
                const auto output_height = output_shape[2];
                const auto output_width = output_shape[3];
                const float *output_data = outputOv.data<float>();

                for (size_t out_y = 0; out_y < output_height; ++out_y)
                    for (size_t out_x = 0; out_x < output_width; ++out_x)
                    {
                        auto x = out_x + inferenceWidth * patchX;
                        auto y = out_y + inferenceHeight * patchY;

                        // Access pattern for [1][C][H][W]: data[c * (H*W) + y * W + x]
                        const size_t spatial_size = output_height * output_width;
                        const size_t pixel_offset = out_y * output_width + out_x;

                        const float p0 = output_data[0 * spatial_size + pixel_offset];
                        const float p1 = output_data[1 * spatial_size + pixel_offset];
                        const float p2 = output_data[2 * spatial_size + pixel_offset];
                        const float p3 = output_data[3 * spatial_size + pixel_offset];
                        const float p4 = output_data[4 * spatial_size + pixel_offset];
                        const float p5 = output_data[5 * spatial_size + pixel_offset];

                        if (synchronousInference && gain != 1.f)
                        {
                            // Apply gain: out = in + (model_out - in) * gain
                            auto apply_gain = [](float in_val, float model_out)
                            {
                                return in_val + (model_out - in_val) * gain;
                            };

                            float in0 = yuvInPointers[0][2 * x + 0 + (2 * y + 0) * yuvStrides[0]];
                            float in1 = yuvInPointers[0][2 * x + 1 + (2 * y + 0) * yuvStrides[0]];
                            float in2 = yuvInPointers[0][2 * x + 0 + (2 * y + 1) * yuvStrides[0]];
                            float in3 = yuvInPointers[0][2 * x + 1 + (2 * y + 1) * yuvStrides[0]];
                            float in4 = yuvInPointers[1][x + y * yuvStrides[1]];
                            float in5 = yuvInPointers[2][x + y * yuvStrides[2]];

                            yuvOutPointers[0][2 * x + 0 + (2 * y + 0) * yuvStrides[0]] = clamp(apply_gain(in0, 1023.f * p0), 0.f, 1023.f);
                            yuvOutPointers[0][2 * x + 1 + (2 * y + 0) * yuvStrides[0]] = clamp(apply_gain(in1, 1023.f * p1), 0.f, 1023.f);
                            yuvOutPointers[0][2 * x + 0 + (2 * y + 1) * yuvStrides[0]] = clamp(apply_gain(in2, 1023.f * p2), 0.f, 1023.f);
                            yuvOutPointers[0][2 * x + 1 + (2 * y + 1) * yuvStrides[0]] = clamp(apply_gain(in3, 1023.f * p3), 0.f, 1023.f);
                            yuvOutPointers[1][x + y * yuvStrides[1]] = clamp(apply_gain(in4, 1023.f * p4), 0.f, 1023.f);
                            yuvOutPointers[2][x + y * yuvStrides[2]] = clamp(apply_gain(in5, 1023.f * p5), 0.f, 1023.f);
                        }
                        else
                        {
                            yuvOutPointers[0][2 * x + 0 + (2 * y + 0) * yuvStrides[0]] = clamp(1023.f * p0, 0.f, 1023.f);
                            yuvOutPointers[0][2 * x + 1 + (2 * y + 0) * yuvStrides[0]] = clamp(1023.f * p1, 0.f, 1023.f);
                            yuvOutPointers[0][2 * x + 0 + (2 * y + 1) * yuvStrides[0]] = clamp(1023.f * p2, 0.f, 1023.f);
                            yuvOutPointers[0][2 * x + 1 + (2 * y + 1) * yuvStrides[0]] = clamp(1023.f * p3, 0.f, 1023.f);
                            yuvOutPointers[1][x + y * yuvStrides[1]] = clamp(1023.f * p4, 0.f, 1023.f);
                            yuvOutPointers[2][x + y * yuvStrides[2]] = clamp(1023.f * p5, 0.f, 1023.f);
                        }
                    }

                profiler.log("caller;process_picture;convertPatches;copy_output");
            }
    }

    void process_picture(uint16_t *in[3], int inStride[3], uint16_t *out[3], int outStride[3], float strength)
    {
        profiler.log("caller");
        // copy input buffer to transfer YUV
        for (int comp = 0; comp < 3; comp++)
        {
            int w = width >> !!comp;
            int h = height >> !!comp;

            int stride = w + 2 * Transfer::padX;

            for (int y = 0; y < h; ++y)
                std::copy(
                    &in[comp][0 + y * inStride[comp]],
                    &in[comp][w + y * inStride[comp]],
                    transfers[ab].planes[comp].data() + Transfer::padX + (Transfer::padY + y) * stride);
        }
        profiler.log("caller;process_picture;copy_input");

        transfers[ab].strength = strength;

        profiler.log("caller;process_picture");
        convertPatchesOpenVINO(width, height, &transfers[ab], &transfers[1 - ab]);
        profiler.log("caller;process_picture;convertPatches");

        ab = 1 - ab;

        // copy output
        for (int comp = 0; comp < 3; comp++)
        {
            int w = width >> !!comp;
            int h = height >> !!comp;

            int stride = w + 2 * Transfer::padX;
            auto data = transfers[ab].planes[comp].data() + stride * Transfer::padY + Transfer::padX;
            for (int y = 0; y < h; ++y)
            {
                // if (comp)
                // std::fill(data, data + w, 700);
                
                std::copy(
                    data,
                    data + w,
                    &out[comp][0 + y * outStride[comp]]);

                std::fill(data, data + w, 512); // unnecessary?

                data += stride;
            }
        }

        profiler.log("caller;process_picture;copy_output");
    }
};

void *nnpf_create(int width, int height, const char *model_path, const char *weights_update_path)
{
    return new Nnpf(model_path, width, height, weights_update_path);
}

void nnpf_destroy(void *p)
{
    delete reinterpret_cast<Nnpf *>(p);
}

void nnpf_process_picture(void *p, uint16_t *in[3], int inStride[3], uint16_t *out[3], int outStride[3], float strength)
{
    reinterpret_cast<Nnpf *>(p)->process_picture(in, inStride, out, outStride, strength);
}
