#include "orbtracker.h"
#include "../utils.h"

OrbTracker::OrbTracker(int nfeatures, float scaleFactor, int nlevels)
    : FeatureTracker(),
      detector(cv::ORB::create(nfeatures, scaleFactor, nlevels,
                               31 /*edgeThreshold*/, 0 /*firstLevel*/,
                               2 /*WTA_K*/, cv::ORB::HARRIS_SCORE,
                               31 /*patchSize*/, 20 /*fastThreshold*/))
{
}

std::vector<cv::DMatch> OrbTracker::ratioTest(
    const std::vector<std::vector<cv::DMatch>>& knnMatches, float ratio)
{
    std::vector<cv::DMatch> good;
    good.reserve(knnMatches.size());
    for (const auto& m : knnMatches) {
        if (m.size() >= 2 && m[0].distance < ratio * m[1].distance) {
            good.push_back(m[0]);
        }
    }
    return good;
}

int OrbTracker::find_pose(cv::Mat& img_cam, CameraCalibration& calibration, cv::Matx44d& w2cam)
{
    OrbFrame last_keyframe_local;
    {
        std::lock_guard<std::mutex> lock(kf_mutex);
        last_keyframe_local = last_frame_rendered;
    }

    // Detect ORB features in the camera image
    cv::Mat gray;
    if (img_cam.channels() == 3)
        cv::cvtColor(img_cam, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img_cam;

    std::vector<cv::KeyPoint> kpts_cam;
    cv::Mat desc_cam;
    detector->detectAndCompute(gray, cv::noArray(), kpts_cam, desc_cam);

    if (static_cast<int>(kpts_cam.size()) < min_number_matches ||
        static_cast<int>(last_keyframe_local.keypoints.size()) < min_number_matches)
    {
        std::cout << "Not enough features detected in either current frame "
                  << kpts_cam.size() << " or last keyframe "
                  << last_keyframe_local.keypoints.size()
                  << ". Skipping pose estimation." << std::endl;
        return -1;
    }

    // knnMatch + Lowe's ratio test (much faster than cross-check BFMatcher)
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(desc_cam, last_keyframe_local.descriptors, knnMatches, 2);
    auto matches = ratioTest(knnMatches);

    if (static_cast<int>(matches.size()) < min_number_matches)
    {
        return -1;
    }

    // Build matched keypoint matrices and 3-D array
    cv::Mat mkpts_cam(static_cast<int>(matches.size()), 2, CV_32F);
    cv::Mat mkpts_kf(static_cast<int>(matches.size()), 2, CV_32F);
    std::vector<int> kf_indices;
    kf_indices.reserve(matches.size());

    for (int i = 0; i < static_cast<int>(matches.size()); ++i)
    {
        const auto& m = matches[i];
        mkpts_cam.at<float>(i, 0) = kpts_cam[m.queryIdx].pt.x;
        mkpts_cam.at<float>(i, 1) = kpts_cam[m.queryIdx].pt.y;
        mkpts_kf.at<float>(i, 0) = last_keyframe_local.keypoints[m.trainIdx].pt.x;
        mkpts_kf.at<float>(i, 1) = last_keyframe_local.keypoints[m.trainIdx].pt.y;
        kf_indices.push_back(m.trainIdx);
    }

    cv::Mat points_3d = Utils::selectByIndices(last_keyframe_local.features_3d, kf_indices);

    w2cam = last_keyframe_local.w2cam;
    return find_filtered_pose(mkpts_cam, mkpts_kf, points_3d, calibration, w2cam);
}

void OrbTracker::update_keyframe(cv::Mat& img_pcd, cv::Mat& depth_pcd,
                                 CameraCalibration& calibration, cv::Matx44d& w2cam)
{
    cv::Mat gray;
    if (img_pcd.channels() == 3)
        cv::cvtColor(img_pcd, gray, cv::COLOR_BGR2GRAY);
    else
        gray = img_pcd;

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    detector->detectAndCompute(gray, cv::noArray(), kpts, desc);

    // Project keypoints to 3-D
    cv::Mat kpts_mat = Utils::keypointsToMat(kpts);
    cv::Mat points_3d, mask_nan;
    Utils::project_3d(kpts_mat, depth_pcd, calibration.getIntrinsicsMatrix(), points_3d, mask_nan);

    // Filter out NaN / invalid points
    std::vector<cv::KeyPoint> valid_kpts;
    cv::Mat valid_desc;
    cv::Mat valid_3d;

    for (int i = 0; i < mask_nan.rows; ++i)
    {
        if (mask_nan.at<uchar>(i))
        {
            valid_kpts.push_back(kpts[i]);
            valid_desc.push_back(desc.row(i));
            valid_3d.push_back(points_3d.row(i));
        }
    }

    {
        std::lock_guard<std::mutex> lock(kf_mutex);
        last_frame_rendered = OrbFrame{w2cam, valid_kpts, valid_desc,
                                       calibration, valid_3d};
    }
}
