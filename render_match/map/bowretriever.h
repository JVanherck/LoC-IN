#ifndef BOWRETRIEVER_H
#define BOWRETRIEVER_H

#include <opencv2/opencv.hpp>
#include <fbow/fbow.h>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief Bag-of-Words retriever using FBoW for ORB descriptor candidate selection.
 *
 * Wraps an FBoW vocabulary to convert ORB descriptors into BoW vectors and
 * find the top-k most similar keyframes by L2-normalised BoW score.
 */
class BowRetriever
{
public:
    BowRetriever() = default;

    /// Load an FBoW vocabulary file (e.g. orb_vocab.fbow).
    explicit BowRetriever(const std::string& vocabPath);

    /// Convert ORB descriptors (CV_8U, Nx32) to a BoW vector.
    fbow::BoWVector computeBoW(const cv::Mat& descriptors);

    /// Add a BoW vector to the internal database.
    void addToDatabase(const fbow::BoWVector& bow);

    /// Return the top-k database indices by BoW score.
    std::vector<int> findTopK(const fbow::BoWVector& query, int k) const;

    /// Number of entries in the database.
    int databaseSize() const { return static_cast<int>(database_.size()); }

    /// Build the database from a vector of BoW vectors (replaces existing DB).
    void buildDatabase(const std::vector<fbow::BoWVector>& bows);

    // ---- Serialisation ------------------------------------------------
    void save(std::ostream& out) const;
    void load(std::istream& in);

    bool isValid() const { return vocab_.isValid(); }

private:
    fbow::Vocabulary vocab_;
    std::vector<fbow::BoWVector> database_;
};

#endif // BOWRETRIEVER_H
