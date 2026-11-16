#include "orbmap.h"
#include "../utils.h"
#include <fstream>
#include <iostream>
#include <numeric>

OrbMap::OrbMap(int nfeatures, CameraCalibration calibration)
    : Map(),
      detector(cv::ORB::create(nfeatures, 1.2f, 8,
                                31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20)),
      calibration(calibration)
{
}

void OrbMap::setPlaceRetriever(std::shared_ptr<PlaceRetriever> retriever)
{
    place_retriever_ = retriever;
}

void OrbMap::setBowRetriever(std::shared_ptr<BowRetriever> retriever)
{
    bow_retriever_ = retriever;
}

std::vector<cv::DMatch> OrbMap::ratioTest(
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

// ---------------------------------------------------------------------------
// find_pose
// ---------------------------------------------------------------------------

bool OrbMap::find_pose(cv::Mat& req_frame, cv::Matx44d& w2cam)
{
    // Detect ORB in query frame
    cv::Mat gray;
    if (req_frame.channels() == 3)
        cv::cvtColor(req_frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = req_frame;

    std::vector<cv::KeyPoint> req_kpts;
    cv::Mat req_desc;
    detector->detectAndCompute(gray, cv::noArray(), req_kpts, req_desc);

    if (req_desc.empty() || req_desc.rows < min_number_matches) return false;

    // Determine candidate keyframes
    std::vector<int> candidates;

    if (bow_retriever_ && bow_retriever_->databaseSize() > 0) {
        // BoW candidate selection — translate DB indices to keyframe indices
        auto query_bow = bow_retriever_->computeBoW(req_desc);
        auto bow_indices = bow_retriever_->findTopK(query_bow, kTopK);
        candidates.reserve(bow_indices.size());
        for (int bi : bow_indices) {
            if (bi < static_cast<int>(bow_to_kf_idx_.size()))
                candidates.push_back(bow_to_kf_idx_[bi]);
        }
        std::cout << "BoW top-" << candidates.size() << " candidates:";
        for (int idx : candidates) std::cout << " " << idx;
        std::cout << std::endl;
    } else if (place_retriever_ && place_retriever_->databaseSize() > 0) {
        // Neural place retriever (EigenPlaces)
        auto query_desc = place_retriever_->computeDescriptor(req_frame);
        candidates = place_retriever_->findTopK(query_desc, kTopK);
        std::cout << "Top-" << candidates.size() << " candidates:";
        for (int idx : candidates) std::cout << " " << idx;
        std::cout << std::endl;
    } else {
        // Brute-force: try all keyframes
        candidates.resize(keyframes.size());
        std::iota(candidates.begin(), candidates.end(), 0);
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);

    for (int ki : candidates)
    {
        auto& kf = keyframes[ki];

        if (kf.descriptors.empty()) continue;

        // knnMatch + Lowe's ratio test
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(req_desc, kf.descriptors, knnMatches, 2);
        auto matches = ratioTest(knnMatches);

        if (static_cast<int>(matches.size()) < min_number_matches)
        {
            continue;
        }

        // Build matched keypoint matrices
        cv::Mat mkpts_req(static_cast<int>(matches.size()), 2, CV_32F);
        cv::Mat mkpts_kf(static_cast<int>(matches.size()), 2, CV_32F);
        std::vector<int> kf_indices;
        kf_indices.reserve(matches.size());

        for (int i = 0; i < static_cast<int>(matches.size()); ++i)
        {
            const auto& m = matches[i];
            mkpts_req.at<float>(i, 0) = req_kpts[m.queryIdx].pt.x;
            mkpts_req.at<float>(i, 1) = req_kpts[m.queryIdx].pt.y;
            mkpts_kf.at<float>(i, 0) = kf.keypoints[m.trainIdx].pt.x;
            mkpts_kf.at<float>(i, 1) = kf.keypoints[m.trainIdx].pt.y;
            kf_indices.push_back(m.trainIdx);
        }

        // Filter outliers with homography
        cv::Mat mask;
        cv::findHomography(mkpts_req, mkpts_kf, cv::USAC_MAGSAC,
                           reprojection_error, mask, iterations, confidence);

        cv::Mat indices;
        cv::findNonZero(mask, indices);

        if (indices.rows < min_number_matches)
        {
            continue;
        }

        cv::Mat points_3d = Utils::selectByIndices(kf.features_3d, kf_indices);
        cv::Mat points_3d_filtered = Utils::filterRowsByMask(points_3d, mask);
        cv::Mat mkpts_req_filtered = Utils::filterRowsByMask(mkpts_req, mask);

        cv::Mat rvec, tvec;
        cv::solvePnPRansac(points_3d_filtered, mkpts_req_filtered,
                           calibration.getIntrinsicsMatrix(),
                           calibration.getDistortionParameters(),
                           rvec, tvec, false, iterations);

        cv::Matx44d pcd2cam;
        Utils::make_transform(rvec, tvec, pcd2cam);

        w2cam = pcd2cam * kf.w2cam;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// add_frame
// ---------------------------------------------------------------------------

void OrbMap::add_frame(cv::Mat& frame, cv::Mat& depth, cv::Matx44d w2cam,
                       CameraCalibration calibration)
{
    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame;

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    detector->detectAndCompute(gray, cv::noArray(), kpts, desc);

    // Project keypoints to 3-D
    cv::Mat kpts_mat = Utils::keypointsToMat(kpts);
    cv::Mat points_3d, mask_nan;
    Utils::project_3d(kpts_mat, depth, calibration.getIntrinsicsMatrix(), points_3d, mask_nan);

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

    // Compute BoW vector if retriever available
    if (bow_retriever_ && bow_retriever_->isValid() && !valid_desc.empty()) {
        auto bow = bow_retriever_->computeBoW(valid_desc);
        bow_retriever_->addToDatabase(bow);
        bow_vectors_.push_back(bow);
        bow_to_kf_idx_.push_back(static_cast<int>(keyframes.size()));
    }

    // Compute EigenPlaces descriptor if retriever available
    if (place_retriever_) {
        place_retriever_->computeDescriptor(frame);
    }

    // Store a thumbnail for visualisation
    cv::Mat thumbnail;
    cv::resize(frame, thumbnail, cv::Size(120, 120));

    keyframes.emplace_back(w2cam, valid_kpts, valid_desc,
                           calibration, valid_3d, thumbnail);
}

// ---------------------------------------------------------------------------
// save / load
// ---------------------------------------------------------------------------

void OrbMap::save_keyframes(const std::string& filename) const
{
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    size_t num_keyframes = keyframes.size();
    out.write(reinterpret_cast<const char*>(&num_keyframes), sizeof(size_t));
    for (const auto& kf : keyframes) {
        kf.save(out);
    }

    // Persist BoW database
    bool has_bow = (bow_retriever_ && bow_retriever_->databaseSize() > 0);
    out.write(reinterpret_cast<const char*>(&has_bow), sizeof(bool));
    if (has_bow) {
        bow_retriever_->save(out);
    }

    // Persist place retriever database
    if (place_retriever_) {
        place_retriever_->save(out);
    }

    out.close();
}

void OrbMap::load_keyframes(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    size_t num_keyframes;
    in.read(reinterpret_cast<char*>(&num_keyframes), sizeof(size_t));
    keyframes.resize(num_keyframes);
    for (auto& kf : keyframes) {
        kf.load(in);
    }

    // Load BoW database
    bool has_bow = false;
    in.read(reinterpret_cast<char*>(&has_bow), sizeof(bool));
    if (has_bow && bow_retriever_) {
        bow_retriever_->load(in);
    } else if (has_bow) {
        // Skip BoW data if no retriever is attached
        // (read & discard — create temp retriever just to advance the stream)
        BowRetriever temp;
        temp.load(in);
    }

    // Load persisted place retriever database
    if (place_retriever_ && in.peek() != EOF) {
        place_retriever_->load(in);
    }

    in.close();

    // Rebuild BoW database from keyframe descriptors if needed
    if (bow_retriever_ && bow_retriever_->isValid() && bow_retriever_->databaseSize() == 0) {
        bow_vectors_.clear();
        bow_vectors_.reserve(keyframes.size());
        bow_to_kf_idx_.clear();
        for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
            if (!keyframes[i].descriptors.empty()) {
                auto bow = bow_retriever_->computeBoW(keyframes[i].descriptors);
                bow_vectors_.push_back(bow);
                bow_to_kf_idx_.push_back(i);
            }
        }
        bow_retriever_->buildDatabase(bow_vectors_);
        if (bow_vectors_.empty()) {
            std::cerr << "Warning: map has no ORB descriptors for BoW, "
                         "relocalization will use brute-force scan" << std::endl;
        }
    }

    // Always rebuild BoW→keyframe index mapping (even if DB was loaded from file)
    if (bow_retriever_ && bow_retriever_->databaseSize() > 0 && bow_to_kf_idx_.empty()) {
        bow_to_kf_idx_.clear();
        for (int i = 0; i < static_cast<int>(keyframes.size()); ++i) {
            if (!keyframes[i].descriptors.empty())
                bow_to_kf_idx_.push_back(i);
        }
    }
}
