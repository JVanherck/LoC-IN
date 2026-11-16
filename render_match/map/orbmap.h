#ifndef ORBMAP_H
#define ORBMAP_H

#include "map.h"
#include "RTRenderer/CameraCalibration.h"
#include "../feature/orbframe.h"
#include "placeretriever.h"
#include "bowretriever.h"
#include <memory>

class OrbMap : public Map
{
public:
    OrbMap(int nfeatures, CameraCalibration calibration);

    bool find_pose(cv::Mat& frame, cv::Matx44d& w2cam) override;
    void add_frame(cv::Mat& frame, cv::Mat& depth, cv::Matx44d w2cam, CameraCalibration calibration) override;

    void save_keyframes(const std::string& filename) const override;
    void load_keyframes(const std::string& filename) override;
    int getKeyframeCount() const override { return static_cast<int>(keyframes.size()); }

    /// Attach a neural place retriever (EigenPlaces) for top-k candidate selection.
    void setPlaceRetriever(std::shared_ptr<PlaceRetriever> retriever);

    /// Attach a BoW retriever (FBoW) for top-k candidate selection.
    void setBowRetriever(std::shared_ptr<BowRetriever> retriever);

private:
    std::vector<OrbFrame> keyframes;
    cv::Ptr<cv::ORB> detector;
    CameraCalibration calibration;
    std::shared_ptr<PlaceRetriever> place_retriever_;
    std::shared_ptr<BowRetriever> bow_retriever_;

    /// Precomputed BoW vectors for each keyframe (parallel to keyframes vector).
    std::vector<fbow::BoWVector> bow_vectors_;

    /// Mapping from BoW database index to keyframe index.
    /// Needed because keyframes with empty descriptors are skipped in the BoW DB.
    std::vector<int> bow_to_kf_idx_;

    static constexpr int kTopK = 10;

    /// Lowe's ratio test on knnMatch results (k=2).
    static std::vector<cv::DMatch> ratioTest(
        const std::vector<std::vector<cv::DMatch>>& knnMatches,
        float ratio = 0.75f);
};

#endif // ORBMAP_H
