#include "xfeatmap.h"
#include "../utils.h"
#include <fstream>
#include <iostream>
#include <numeric>

XFeatMap::XFeatMap(std::shared_ptr<XFeat::XFDetector> detector, CameraCalibration calibration): Map(), detector(detector), calibration(calibration)
{
}

void XFeatMap::setPlaceRetriever(std::shared_ptr<PlaceRetriever> retriever)
{
    place_retriever_ = retriever;
}

bool XFeatMap::find_pose(cv::Mat &req_frame, cv::Matx44d &w2cam)
{
    auto frame_tensor = detector->parseInput(req_frame);

    std::unordered_map<std::string, at::Tensor> features;
    detector->detectAndCompute(frame_tensor, features);

    auto& req_desc = features["descriptors"];
    auto& req_kpts = features["keypoints"];

    // Determine which keyframes to try
    std::vector<int> candidates;
    if (place_retriever_ && place_retriever_->databaseSize() > 0) {
        auto query_desc = place_retriever_->computeDescriptor(req_frame);
        candidates = place_retriever_->findTopK(query_desc, kTopK);
        std::cout << "Top-" << candidates.size() << " candidates:";
        for (int idx : candidates) std::cout << " " << idx;
        std::cout << std::endl;
    } else {
        // Fallback: try all keyframes
        candidates.resize(keyframes.size());
        std::iota(candidates.begin(), candidates.end(), 0);
    }

        // cv::imshow("Query Frame", req_frame);
        // cv::waitKey(0);

    for (int ki : candidates)
    {
        auto& kf = keyframes[ki];
        auto& desc = kf.features["descriptors"];
        auto& kpts = kf.features["keypoints"];

        // acquire matches
        torch::Tensor idxs_req, idxs;
        detector->match(req_desc, desc, idxs_req, idxs, -1.0);

        if(idxs.size(0) < min_number_matches)
        {
            continue;
        }

        torch::Tensor mkpts_req_tensor = req_kpts.index({idxs_req});
        torch::Tensor mkpts_tensor = kpts.index({idxs});

        cv::Mat mkpts_req = detector->tensorToMat(mkpts_req_tensor);
        cv::Mat mkpts = detector->tensorToMat(mkpts_tensor);

        // filter outliers
        cv::Mat mask;
        cv::findHomography(mkpts_req, mkpts, cv::USAC_MAGSAC, reprojection_error, mask, iterations, confidence);

        cv::Mat indices;
        cv::findNonZero(mask, indices);

        if(indices.rows < min_number_matches)
        {
            continue;
        }

        cv::Mat points_3d = Utils::selectByIndicesTensor(kf.features_3d, idxs);
        cv::Mat points_3d_filtered = Utils::filterRowsByMask(points_3d, mask);
        cv::Mat mkpts_req_filtered = Utils::filterRowsByMask(mkpts_req, mask);

        cv::Mat rvec, tvec;
        cv::solvePnPRansac(points_3d_filtered, mkpts_req_filtered, calibration.getIntrinsicsMatrix(), calibration.getDistortionParameters(), rvec, tvec, false, iterations);

        cv::Matx44d pcd2cam;
        Utils::make_transform(rvec, tvec, pcd2cam);

        w2cam = pcd2cam * kf.w2cam;

        return true;
    }
    return false;
}

void XFeatMap::add_frame(cv::Mat &frame, cv::Mat& depth, cv::Matx44d w2cam, CameraCalibration calibration)
{
    auto frame_tensor = detector->parseInput(frame);

    std::unordered_map<std::string, at::Tensor> features;
    detector->detectAndCompute(frame_tensor, features);

    cv::Mat points_3d, mask_nan;
    Utils::project_3d(detector->tensorToMat(features["keypoints"]), depth, calibration.getIntrinsicsMatrix(), points_3d, mask_nan);

    cv::Mat points_3d_filtered = Utils::filterRowsByMask(points_3d, mask_nan);

    std::vector<int64_t> valid_indices;
    for (int i = 0; i < mask_nan.rows; ++i) {
        if (mask_nan.at<uchar>(i)) {
            valid_indices.push_back(i);
        }
    }

    at::Tensor indices = torch::tensor(valid_indices, torch::dtype(torch::kLong));

    features["keypoints"] = features["keypoints"].index({indices});
    features["descriptors"] = features["descriptors"].index({indices});

    // Compute place descriptor if retriever is available
    torch::Tensor place_desc;
    if (place_retriever_) {
        place_desc = place_retriever_->computeDescriptor(frame);
    }

    // Store a thumbnail for visualization during relocalization
    cv::Mat thumbnail;
    cv::resize(frame, thumbnail, cv::Size(120, 120));

    keyframes.emplace_back(w2cam, features, calibration, points_3d_filtered, place_desc);
    keyframes.back().render = thumbnail;
}

void XFeatMap::save_keyframes(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    size_t num_keyframes = keyframes.size();
    out.write(reinterpret_cast<const char*>(&num_keyframes), sizeof(size_t));
    for (const auto& kf : keyframes) {
        kf.save(out);
    }

    // Persist place retriever database
    if (place_retriever_) {
        place_retriever_->save(out);
    }

    out.close();

    // Build the in-memory database so it's ready after map construction
    if (place_retriever_) {
        std::vector<torch::Tensor> descs;
        descs.reserve(keyframes.size());
        for (const auto& kf : keyframes) {
            if (kf.place_descriptor.defined() && kf.place_descriptor.numel() > 0) {
                descs.push_back(kf.place_descriptor);
            }
        }
        if (!descs.empty()) {
            place_retriever_->buildDatabase(descs);
        }
    }
}

void XFeatMap::load_keyframes(const std::string& filename) {
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

    // Load persisted place retriever database
    if (place_retriever_ && in.peek() != EOF) {
        place_retriever_->load(in);
    }

    in.close();

    // Rebuild place retriever database from loaded descriptors
    // (this handles the case where the DB was not persisted in the file
    //  but per-frame descriptors were)
    if (place_retriever_ && place_retriever_->databaseSize() == 0) {
        std::vector<torch::Tensor> descs;
        descs.reserve(keyframes.size());
        for (const auto& kf : keyframes) {
            if (kf.place_descriptor.defined() && kf.place_descriptor.numel() > 0) {
                descs.push_back(kf.place_descriptor);
            }
        }
        if (!descs.empty()) {
            place_retriever_->buildDatabase(descs);
        } else {
            std::cerr << "Warning: map has no place descriptors, "
                         "relocalization will use brute-force scan" << std::endl;
        }
    }
}
