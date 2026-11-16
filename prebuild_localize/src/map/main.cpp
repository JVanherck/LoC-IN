/**
 * @file main.cpp
 * @brief Unified map builder supporting multiple SLAM backends with frame caching.
 * 
 * This application can:
 * 1. Render frames from a point cloud and save them to a cache
 * 2. Load cached frames and build a map with a specific backend
 * 
 * This allows building maps with different SLAM backends using the exact same frames.
 * 
 * Usage:
 *   Render frames only (no SLAM):
 *     ./prebuild_map_<backend> render <folder> <calib> <cloud>
 * 
 *   Build map from cached frames:
 *     ./prebuild_map_<backend> build [backend-options] <folder> <calib>
 * 
 *   Render and build in one step:
 *     ./prebuild_map_<backend> full [backend-options] <folder> <calib> <cloud>
 */

#include "map/slam_backend.h"
#include "map/frame_cache.h"
#include "common/pose_generator.h"
#include "common/pose_io.h"
#include "common/geometry.h"
#include "common/camera_utils.h"
#include "common/trajectory_io.h"
#include "utils.h"

#ifdef STELLA_AVAILABLE
#include "map/stella_backend.h"
#endif

#ifdef ORBSLAM3_AVAILABLE
#include "map/orbslam3_backend.h"
#endif

#include "RTRenderer/cloudreader.h"
#include "RTRenderer/project_cloud.h"
#include "DeviceFactory/CameraCalibration.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>

void printUsage(const char* progName) {
    std::cerr << "Usage: " << progName << " <mode> [options] <folder> <calib> [cloud]\n\n";
    std::cerr << "Modes:\n";
    std::cerr << "  render  - Render and cache frames only (no SLAM), requires <cloud>\n";
    std::cerr << "  build   - Build map from cached frames (no rendering)\n";
    std::cerr << "  full    - Render frames and build map in one step, requires <cloud>\n\n";
    
    std::cerr << "Rendering options (render/full modes):\n";
    std::cerr << "  --poses <path>    Load camera poses from file instead of interactive GUI\n";
    std::cerr << "  --no-neural       Shorthand for --render-mode projected\n";
    std::cerr << "  --render-mode <m> Renderer to use: neural (computeFull, default),\n";
    std::cerr << "                    projected (computeRGBD), filtered (computeFilteredRGBD)\n\n";

#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "Options for ORB-SLAM3 (build/full modes):\n";
    std::cerr << "  --voc <path>      Path to ORB vocabulary file (ORBvoc.txt)\n";
    std::cerr << "  --config <path>   Path to ORB-SLAM3 config template file\n\n";
#endif

    std::cerr << "Required arguments:\n";
    std::cerr << "  folder   Base folder containing all data\n";
    std::cerr << "  calib    Camera calibration file relative to folder\n";
    std::cerr << "  cloud    Point cloud file relative to folder (render/full modes)\n\n";
    
    std::cerr << "Examples:\n";
    std::cerr << "  # Step 1: Render frames once\n";
    std::cerr << "  " << progName << " render /data/scene calib.txt cloud.ply\n\n";
    std::cerr << "  # Step 2: Build map from cached frames\n";
#ifdef STELLA_AVAILABLE
    std::cerr << "  " << progName << " build /data/scene calib.txt\n";
#endif
#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "  " << progName << " build --voc /path/ORBvoc.txt --config /path/template.yaml /data/scene calib.txt\n";
#endif
    std::cerr << "\n  # Or do both in one step:\n";
#ifdef STELLA_AVAILABLE
    std::cerr << "  " << progName << " full /data/scene calib.txt cloud.ply\n";
#endif
#ifdef ORBSLAM3_AVAILABLE
    std::cerr << "  " << progName << " full --voc /path/ORBvoc.txt --config /path/template.yaml /data/scene calib.txt cloud.ply\n";
#endif
}

bool renderFrames(const std::filesystem::path& cloudFile,
                  const std::filesystem::path& cacheDir,
                  const CameraCalibration& calib,
                  FrameCache& cache,
                  const std::string& posesFile = "",
                  const std::string& renderMode = "neural") {

    std::cout << "\n=== Rendering Frames ===" << std::endl;
    std::cout << "Loading point cloud from: " << cloudFile << std::endl;
    std::cout << "Rendering mode: " << renderMode << std::endl;

    auto grid = CloudReader::loadCloud(cloudFile, cacheDir);

    // Only the neural renderer needs a TensorRT model; both point-projection
    // variants (plain and depth-filtered) run without one.
    std::string engineName = (renderMode == "neural")
        ? std::string("trt_") + std::to_string(calib.getWidth()) + "x" + std::to_string(calib.getHeight()) + ".ts"
        : std::string();
    ProjectCloud projector(grid, engineName);

    std::vector<cv::Matx44d> trajectory;
    if (!posesFile.empty()) {
        std::cout << "Loading camera poses from: " << posesFile << std::endl;
        trajectory = loadPoses(posesFile);
        if (trajectory.empty()) {
            std::cerr << "Failed to load poses or file is empty: " << posesFile << std::endl;
            return false;
        }
    } else {
        std::cout << "Generating camera poses interactively..." << std::endl;
        trajectory = PoseGenerator::getPoses(grid);
        // Save the generated poses so they can be reused later
        savePoses(cacheDir / "generated_poses.txt", trajectory);
    }
    std::cout << "Using " << trajectory.size() << " poses." << std::endl;

    if (!cache.initForWriting()) {
        std::cerr << "Failed to initialize frame cache for writing" << std::endl;
        return false;
    }

    int frameIdx = 0;
    for (const auto& w2cam : trajectory) {
        ++frameIdx;
        std::cout << "Rendering frame " << frameIdx << "/" << trajectory.size() << "\r" << std::flush;

        cv::Mat rgb(calib.getHeight(), calib.getWidth(), CV_8UC3);
        cv::Mat depth(calib.getHeight(), calib.getWidth(), CV_32F);
        
        if (renderMode == "neural")
            projector.computeFull(calib, w2cam, &rgb, &depth);
        else if (renderMode == "filtered")
            projector.computeFilteredRGBD(calib, w2cam, &rgb, &depth);
        else
            projector.computeRGBD(calib, w2cam, &rgb, &depth);

        double timestamp = static_cast<double>(frameIdx) / 30.0;  // Simulated 30 FPS
        Eigen::Matrix4d pose_cw = convertToEigen(w2cam);

        if (!cache.saveFrame(rgb, depth, pose_cw, timestamp)) {
            std::cerr << "\nFailed to save frame " << frameIdx << std::endl;
            return false;
        }
    }
    std::cout << std::endl;

    if (!cache.finalizeWriting()) {
        std::cerr << "Failed to finalize frame cache" << std::endl;
        return false;
    }

    cv::destroyAllWindows();
    return true;
}

bool buildMap(const std::filesystem::path& cacheDir,
              const std::filesystem::path& cloudFile,
              const std::filesystem::path& calibFile,
              const CameraCalibration& calib,
              FrameCache& cache,
              const std::string& renderMode,
              [[maybe_unused]] const std::string& vocPath,
              [[maybe_unused]] const std::string& configTemplate) {

    std::cout << "\n=== Building Map ===" << std::endl;
    std::string cloudStem = cloudFile.stem().string();
    std::string map_tag = cloudStem + "_" + renderMode;

    if (!cache.initForReading()) {
        std::cerr << "Failed to load frame cache. Run 'render' mode first." << std::endl;
        return false;
    }

    // Create backend
    std::unique_ptr<SlamBackend> backend;

#ifdef STELLA_AVAILABLE
    std::cout << "Initializing Stella VSLAM backend..." << std::endl;
    backend = std::make_unique<StellaBackend>(calib);
#endif

#ifdef ORBSLAM3_AVAILABLE
    std::cout << "Initializing ORB-SLAM3 backend..." << std::endl;
    
    std::filesystem::path orbConfig = cacheDir / "orbslam3_config.yaml";
    if (!createORBSLAM3Config(configTemplate, calibFile, orbConfig)) {
        std::cerr << "Failed to create ORB-SLAM3 config file" << std::endl;
        return false;
    }
    
    backend = std::make_unique<OrbSlam3Backend>(vocPath, orbConfig.string());
#endif

    if (!backend) {
        std::cerr << "No SLAM backend available!" << std::endl;
        return false;
    }

    std::cout << "Backend: " << backend->getName() << std::endl;
    std::cout << "Processing " << cache.getFrameCount() << " frames..." << std::endl;

    auto mapBuildStart = std::chrono::high_resolution_clock::now();

    int successCount = 0;
    for (int i = 0; i < cache.getFrameCount(); i++) {
        FrameCache::Frame frame;
        if (!cache.loadFrame(i, frame)) {
            std::cerr << "Failed to load frame " << i << std::endl;
            continue;
        }

        std::cout << "Frame " << (i + 1) << "/" << cache.getFrameCount() << "\r" << std::flush;

        bool success = backend->addKeyframe(frame.rgb, frame.depth, frame.timestamp, frame.pose_cw);
        
        if (success) {
            successCount++;
        }
    }
    std::cout << std::endl;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Finalizing map..." << std::endl;
    std::cout << "Total frames: " << cache.getFrameCount() << std::endl;
    std::cout << "Successful keyframes: " << successCount << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Construct descriptive map name: {cloud}_{rendering}
    std::string mapName = "map";
    if (!map_tag.empty()) {
        mapName = map_tag;
    }

    backend->finalize(cacheDir, mapName);

    // Write metadata sidecar (extension depends on backend)
    std::string meta_ext;
#ifdef STELLA_AVAILABLE
    meta_ext = "_stella.meta";
#endif
#ifdef ORBSLAM3_AVAILABLE
    meta_ext = "_orbslam3.meta";
#endif
    if (!meta_ext.empty()) {
        std::ofstream meta(cacheDir / (mapName + meta_ext));
        writeTrajectoryHeader(meta, "prebuild_map", backend->getName(),
                              cloudStem, "", renderMode);
        auto mapBuildEnd = std::chrono::high_resolution_clock::now();
        auto mapBuildMs = std::chrono::duration_cast<std::chrono::milliseconds>(mapBuildEnd - mapBuildStart).count();
        meta << "# num_keyframes: " << backend->getKeyframeCount() << "\n";
        meta << "# num_landmarks: " << backend->getLandmarkCount() << "\n";
        meta << "# map_construction_time_ms: " << mapBuildMs << "\n";
        meta.close();
    }
    
    std::cout << "  Final keyframes: " << backend->getKeyframeCount() << std::endl;
    std::cout << "  Final landmarks: " << backend->getLandmarkCount() << std::endl;
    std::cout << "\nMap building complete!" << std::endl;
    std::cout << "Output directory: " << cacheDir << std::endl;

    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printUsage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    
    if (mode != "render" && mode != "build" && mode != "full") {
        std::cerr << "Invalid mode: " << mode << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::string vocPath;
    std::string configTemplate;
    std::string posesFile;
    std::string renderMode = "neural";
    int argIdx = 2;
    
    while (argIdx < argc && argv[argIdx][0] == '-') {
        std::string arg = argv[argIdx];
        if (arg == "--voc" && argIdx + 1 < argc) {
            vocPath = argv[++argIdx];
        } else if (arg == "--config" && argIdx + 1 < argc) {
            configTemplate = argv[++argIdx];
        } else if (arg == "--poses" && argIdx + 1 < argc) {
            posesFile = argv[++argIdx];
        } else if (arg == "--no-neural") {
            renderMode = "projected";
        } else if (arg == "--render-mode" && argIdx + 1 < argc) {
            renderMode = argv[++argIdx];
            if (renderMode != "neural" && renderMode != "projected" && renderMode != "filtered") {
                std::cerr << "Unknown render mode: " << renderMode
                          << ". Supported: neural, projected, filtered" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        ++argIdx;
    }

    bool needsCloud = (mode == "render" || mode == "full");
    int requiredArgs = needsCloud ? 3 : 2;
    
    if (argc - argIdx < requiredArgs) {
        std::cerr << "Missing required arguments.\n";
        printUsage(argv[0]);
        return 1;
    }

    std::filesystem::path folder = argv[argIdx];
    std::filesystem::path calibFile = folder / argv[argIdx + 1];
    std::filesystem::path cloudFile = folder / argv[argIdx + 2];

    std::filesystem::path cacheDir = folder / ".cache";

#ifdef ORBSLAM3_AVAILABLE
    if ((mode == "build" || mode == "full") && (vocPath.empty() || configTemplate.empty())) {
        std::cerr << "ORB-SLAM3 backend requires --voc and --config options.\n";
        printUsage(argv[0]);
        return 1;
    }
#endif

    std::filesystem::create_directories(cacheDir);

    CameraCalibration calib;
    if (!calib.loadCalibration(calibFile)) {
        std::cerr << "Failed to load camera calibration from: " << calibFile << std::endl;
        return 1;
    }

    calib = getAlignedCalibration(calib, 16);

    std::string mapTag = cloudFile.stem().string() + "_" + renderMode;
    FrameCache cache(cacheDir, mapTag);

    if (mode == "render") {
        return renderFrames(cloudFile, cacheDir, calib, cache, posesFile, renderMode) ? 0 : 1;
    } 
    else if (mode == "build") {
        return buildMap(cacheDir, cloudFile, calibFile, calib, cache, renderMode, vocPath, configTemplate) ? 0 : 1;
    }
    else if (mode == "full") {
        if (!renderFrames(cloudFile, cacheDir, calib, cache, posesFile, renderMode)) {
            return 1;
        }
        FrameCache readCache(cacheDir, mapTag);
        return buildMap(cacheDir, cloudFile, calibFile, calib, readCache, renderMode, vocPath, configTemplate) ? 0 : 1;
    }

    return 0;
}
