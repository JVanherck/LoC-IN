#include "bowretriever.h"
#include <algorithm>
#include <numeric>
#include <iostream>

BowRetriever::BowRetriever(const std::string& vocabPath)
{
    vocab_.readFromFile(vocabPath);
    if (!vocab_.isValid()) {
        throw std::runtime_error("Failed to load FBoW vocabulary from: " + vocabPath);
    }
    std::cout << "Loaded FBoW vocabulary from " << vocabPath << std::endl;
}

fbow::BoWVector BowRetriever::computeBoW(const cv::Mat& descriptors)
{
    return vocab_.transform(descriptors);
}

void BowRetriever::addToDatabase(const fbow::BoWVector& bow)
{
    database_.push_back(bow);
}

std::vector<int> BowRetriever::findTopK(const fbow::BoWVector& query, int k) const
{
    const int n = static_cast<int>(database_.size());
    if (n == 0) return {};

    std::vector<std::pair<double, int>> scores(n);
    for (int i = 0; i < n; ++i) {
        scores[i] = { fbow::BoWVector::score(query, database_[i]), i };
    }

    int topk = std::min(k, n);
    std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> result;
    result.reserve(topk);
    for (int i = 0; i < topk; ++i) {
        result.push_back(scores[i].second);
    }
    return result;
}

void BowRetriever::buildDatabase(const std::vector<fbow::BoWVector>& bows)
{
    database_ = bows;
}

// ---------------------------------------------------------------------------
// Serialisation: store the database of BoW vectors
// ---------------------------------------------------------------------------

void BowRetriever::save(std::ostream& out) const
{
    size_t n = database_.size();
    out.write(reinterpret_cast<const char*>(&n), sizeof(size_t));
    for (const auto& bow : database_) {
        bow.toStream(out);
    }
}

void BowRetriever::load(std::istream& in)
{
    size_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
    database_.resize(n);
    for (auto& bow : database_) {
        bow.fromStream(in);
    }
}
