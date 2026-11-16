#ifndef ORBFRAME_H
#define ORBFRAME_H

#include <opencv2/opencv.hpp>
#include "RTRenderer/CameraCalibration.h"

/**
 * @brief Data class for a single ORB feature frame.
 *
 * Parallel to XfeatFrame but uses pure OpenCV types — no LibTorch dependency.
 * Stores ORB keypoints, descriptors, optional 3-D back-projections, pose,
 * calibration, and an optional thumbnail for visualisation.
 */
class OrbFrame
{
public:
    cv::Matx44d w2cam;
    CameraCalibration calibration;

    std::vector<cv::KeyPoint> keypoints;   ///< Detected ORB keypoints
    cv::Mat descriptors;                   ///< ORB descriptors (CV_8U, Nx32)
    cv::Mat features_3d;                   ///< Back-projected 3-D points (Nx1 CV_32FC3)
    cv::Mat render;                        ///< Optional thumbnail for visualisation

    // ---- Constructors -------------------------------------------------

    OrbFrame() = default;

    OrbFrame(cv::Matx44d w2cam,
             std::vector<cv::KeyPoint> keypoints,
             cv::Mat descriptors,
             CameraCalibration calibration,
             cv::Mat features_3d,
             cv::Mat render = cv::Mat())
        : w2cam(w2cam), calibration(calibration),
          keypoints(std::move(keypoints)),
          descriptors(std::move(descriptors)),
          features_3d(std::move(features_3d)),
          render(std::move(render)) {}

    // ---- Binary serialisation -----------------------------------------

    void save(std::ostream& out) const
    {
        // Pose
        out.write(reinterpret_cast<const char*>(&w2cam), sizeof(cv::Matx44d));

        // Calibration
        calibration.save(out);

        // Keypoints
        size_t nKpts = keypoints.size();
        out.write(reinterpret_cast<const char*>(&nKpts), sizeof(size_t));
        for (const auto& kp : keypoints) {
            out.write(reinterpret_cast<const char*>(&kp.pt.x), sizeof(float));
            out.write(reinterpret_cast<const char*>(&kp.pt.y), sizeof(float));
            out.write(reinterpret_cast<const char*>(&kp.size), sizeof(float));
            out.write(reinterpret_cast<const char*>(&kp.angle), sizeof(float));
            out.write(reinterpret_cast<const char*>(&kp.response), sizeof(float));
            out.write(reinterpret_cast<const char*>(&kp.octave), sizeof(int));
            out.write(reinterpret_cast<const char*>(&kp.class_id), sizeof(int));
        }

        // Descriptors (cv::Mat, CV_8U)
        saveMat(out, descriptors);

        // 3-D features
        saveMat(out, features_3d);
    }

    void load(std::istream& in)
    {
        // Pose
        in.read(reinterpret_cast<char*>(&w2cam), sizeof(cv::Matx44d));

        // Calibration
        calibration.load(in);

        // Keypoints
        size_t nKpts;
        in.read(reinterpret_cast<char*>(&nKpts), sizeof(size_t));
        keypoints.resize(nKpts);
        for (auto& kp : keypoints) {
            in.read(reinterpret_cast<char*>(&kp.pt.x), sizeof(float));
            in.read(reinterpret_cast<char*>(&kp.pt.y), sizeof(float));
            in.read(reinterpret_cast<char*>(&kp.size), sizeof(float));
            in.read(reinterpret_cast<char*>(&kp.angle), sizeof(float));
            in.read(reinterpret_cast<char*>(&kp.response), sizeof(float));
            in.read(reinterpret_cast<char*>(&kp.octave), sizeof(int));
            in.read(reinterpret_cast<char*>(&kp.class_id), sizeof(int));
        }

        // Descriptors
        loadMat(in, descriptors);

        // 3-D features
        loadMat(in, features_3d);
    }

private:
    static void saveMat(std::ostream& out, const cv::Mat& mat)
    {
        bool empty = mat.empty();
        out.write(reinterpret_cast<const char*>(&empty), sizeof(bool));
        if (empty) return;

        int rows = mat.rows, cols = mat.cols, type = mat.type();
        out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        out.write(reinterpret_cast<const char*>(&type), sizeof(int));

        size_t dataSize = mat.total() * mat.elemSize();
        out.write(reinterpret_cast<const char*>(mat.data), dataSize);
    }

    static void loadMat(std::istream& in, cv::Mat& mat)
    {
        bool empty;
        in.read(reinterpret_cast<char*>(&empty), sizeof(bool));
        if (empty) { mat.release(); return; }

        int rows, cols, type;
        in.read(reinterpret_cast<char*>(&rows), sizeof(int));
        in.read(reinterpret_cast<char*>(&cols), sizeof(int));
        in.read(reinterpret_cast<char*>(&type), sizeof(int));

        mat.create(rows, cols, type);
        size_t dataSize = mat.total() * mat.elemSize();
        in.read(reinterpret_cast<char*>(mat.data), dataSize);
    }
};

#endif // ORBFRAME_H
