#include "placeretriever.h"
#include <iostream>
#include <c10/cuda/CUDACachingAllocator.h>

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------

PlaceRetriever::PlaceRetriever(const std::string& model_path)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    std::cout << "Loading EigenPlaces model from " << model_path
              << " (device: " << device_ << ")" << std::endl;
    model_ = torch::jit::load(model_path, device_);
    model_.eval();

    // Warmup: run a few dummy forward passes so kernels / memory pools
    // are fully primed before the main loop.
    {
        torch::InferenceMode guard;
        auto dummy = torch::zeros({1, 3, kInputSize, kInputSize}, device_);
        for (int i = 0; i < 3; ++i) {
            model_.forward({dummy});
        }
        if (device_.is_cuda()) {
            torch::cuda::synchronize();
            c10::cuda::CUDACachingAllocator::emptyCache();
        }
        std::cout << "EigenPlaces warmup complete" << std::endl;
    }
}

// -----------------------------------------------------------------------
// Preprocessing
// -----------------------------------------------------------------------

torch::Tensor PlaceRetriever::preprocess(const cv::Mat& image) const
{
    // Resize to 480x480
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_CUBIC);

    // BGR -> RGB and uint8 -> float32 [0,1]
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    // HWC -> CHW tensor
    auto tensor = torch::from_blob(rgb.data, {1, kInputSize, kInputSize, 3}, torch::kFloat32);
    tensor = tensor.permute({0, 3, 1, 2}).contiguous(); // (1, 3, H, W)

    // ImageNet normalisation
    auto mean = torch::tensor({0.485f, 0.456f, 0.406f}).view({1, 3, 1, 1});
    auto std  = torch::tensor({0.229f, 0.224f, 0.225f}).view({1, 3, 1, 1});
    tensor = (tensor - mean) / std;

    return tensor.to(device_);
}

// -----------------------------------------------------------------------
// Descriptor computation
// -----------------------------------------------------------------------

torch::Tensor PlaceRetriever::computeDescriptor(const cv::Mat& image)
{
    torch::InferenceMode guard;
    auto input = preprocess(image);
    auto output = model_.forward({input}).toTensor(); // (1, 512) on device_

    // Immediately move result to CPU and release GPU intermediates
    auto cpu_desc = output.detach().to(torch::kCPU);

    if (device_.is_cuda()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    return cpu_desc;
}

// -----------------------------------------------------------------------
// Database management
// -----------------------------------------------------------------------

void PlaceRetriever::buildDatabase(const std::vector<torch::Tensor>& descriptors)
{
    if (descriptors.empty()) {
        database_ = torch::Tensor();
        return;
    }
    std::vector<torch::Tensor> cpu_descs;
    cpu_descs.reserve(descriptors.size());
    for (const auto& d : descriptors) {
        cpu_descs.push_back(d.to(torch::kCPU).view({1, -1}));
    }
    database_ = torch::cat(cpu_descs, 0);  // (N, D), stays on CPU
    std::cout << "PlaceRetriever: built database with " << database_.size(0)
              << " descriptors (dim=" << database_.size(1) << ")" << std::endl;
}

int PlaceRetriever::databaseSize() const
{
    return database_.defined() ? static_cast<int>(database_.size(0)) : 0;
}

// -----------------------------------------------------------------------
// Retrieval
// -----------------------------------------------------------------------

std::vector<int> PlaceRetriever::findTopK(const torch::Tensor& query, int k) const
{
    if (!database_.defined() || database_.size(0) == 0) {
        return {};
    }

    // Both query and database are L2-normalised, so dot product == cosine sim
    auto q = query.to(torch::kCPU).view({1, -1});      // (1, D)
    auto similarities = torch::mm(q, database_.t());    // (1, N)

    int n = static_cast<int>(database_.size(0));
    k = std::min(k, n);

    auto [values, indices] = similarities.squeeze(0).topk(k); // (k,)
    auto idx_cpu = indices.to(torch::kCPU).to(torch::kInt32);
    auto idx_ptr = idx_cpu.data_ptr<int32_t>();

    return std::vector<int>(idx_ptr, idx_ptr + k);
}

// -----------------------------------------------------------------------
// Serialisation
// -----------------------------------------------------------------------

void PlaceRetriever::save(std::ostream& out) const
{
    bool has_db = database_.defined() && database_.size(0) > 0;
    out.write(reinterpret_cast<const char*>(&has_db), sizeof(bool));
    if (has_db) {
        auto cpu_db = database_.to(torch::kCPU);
        std::ostringstream buf(std::ios::binary);
        torch::save(cpu_db, buf);
        std::string data = buf.str();
        size_t sz = data.size();
        out.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
        out.write(data.data(), sz);
    }
}

void PlaceRetriever::load(std::istream& in)
{
    bool has_db;
    in.read(reinterpret_cast<char*>(&has_db), sizeof(bool));
    if (has_db) {
        size_t sz;
        in.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
        std::string data(sz, '\0');
        in.read(&data[0], sz);
        std::istringstream buf(data, std::ios::binary);
        torch::Tensor t;
        torch::load(t, buf);
        database_ = t.to(torch::kCPU);
        std::cout << "PlaceRetriever: loaded database with " << database_.size(0)
                  << " descriptors" << std::endl;
    } else {
        database_ = torch::Tensor();
    }
}
