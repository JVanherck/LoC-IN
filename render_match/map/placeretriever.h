#ifndef PLACERETRIEVER_H
#define PLACERETRIEVER_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>

/**
 * @brief Visual place retrieval using an EigenPlaces TorchScript model.
 *
 * The model runs on GPU for speed; descriptors and the database are kept
 * on CPU to minimise VRAM usage.  JIT profiling is disabled to avoid an
 * unbounded memory leak during long map-construction sessions.
 */
class PlaceRetriever
{
public:
    /**
     * @brief Load a traced EigenPlaces TorchScript model.
     * @param model_path  Path to the .pt TorchScript file.
     */
    explicit PlaceRetriever(const std::string& model_path);

    /**
     * @brief Compute the global place descriptor for an image.
     * @param image  BGR cv::Mat (any size; will be resized internally).
     * @return L2-normalised 512-dim descriptor tensor **on CPU**.
     */
    torch::Tensor computeDescriptor(const cv::Mat& image);

    /**
     * @brief (Re)build the database from a vector of descriptors.
     *
     * Stacks all descriptors into a single (N, D) tensor on CPU for
     * fast batch cosine-similarity lookup.
     */
    void buildDatabase(const std::vector<torch::Tensor>& descriptors);

    /**
     * @brief Retrieve the top-k most similar database entries.
     * @param query  A single L2-normalised descriptor (1, D) or (D,), on any device.
     * @param k      Number of results to return (clamped to database size).
     * @return Indices into the database sorted by descending similarity.
     */
    std::vector<int> findTopK(const torch::Tensor& query, int k = 5) const;

    /// Number of descriptors currently in the database.
    int databaseSize() const;

    /// Serialise the descriptor database to a binary stream.
    void save(std::ostream& out) const;

    /// Deserialise the descriptor database from a binary stream.
    void load(std::istream& in);

private:
    torch::jit::Module model_;
    torch::Device device_;            // GPU if available, else CPU
    torch::Tensor database_;          // (N, D) stacked descriptors, always on CPU

    /// Preprocess a BGR cv::Mat into a (1,3,480,480) normalised tensor on device_.
    torch::Tensor preprocess(const cv::Mat& image) const;

    static constexpr int kInputSize = 480;
};

#endif // PLACERETRIEVER_H
