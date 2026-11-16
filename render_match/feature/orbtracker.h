#ifndef ORBTRACKER_H
#define ORBTRACKER_H

#include "featuretracker.h"
#include "orbframe.h"

class OrbTracker : public FeatureTracker
{
public:
    OrbTracker(int nfeatures = 1000, float scaleFactor = 1.2f, int nlevels = 8);

    int find_pose(cv::Mat& img_cam, CameraCalibration& intrinsics, cv::Matx44d& w2cam) override;
    void update_keyframe(cv::Mat& img_pcd, cv::Mat& depth_pcd, CameraCalibration& intrinsics, cv::Matx44d& w2cam) override;

private:
    cv::Ptr<cv::ORB> detector;

    /// Lowe's ratio test on knnMatch results (k=2).
    static std::vector<cv::DMatch> ratioTest(
        const std::vector<std::vector<cv::DMatch>>& knnMatches,
        float ratio = 0.75f);

    OrbFrame last_frame_rendered;
};

#endif // ORBTRACKER_H
