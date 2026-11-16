# LoC-IN: Real-Time Localization of Cameras in Neural Rendered Prior LiDAR Maps

This is the implementation of our RA-L paper, **LoC-IN: Real-Time Localization of Cameras in Neural
Rendered Prior LiDAR Maps** (published in *IEEE Robotics and Automation Letters*, DOI
[10.1109/LRA.2026.3748685](https://doi.org/10.1109/LRA.2026.3748685)).

*Accurate camera localization is crucial for robotics and Extended Reality (XR), enabling reliable
navigation and alignment of virtual and real content. Existing visual methods often suffer from
drift, scale ambiguity, and depend on fiducials or loop closure. This work introduces a real-time
method for localizing a camera within a pre-captured, highly accurate colored LiDAR point cloud. By
rendering synthetic views from this cloud, 2D-3D correspondences are established between live frames
and the point cloud. A neural rendering technique narrows the domain gap between synthetic and real
images, reducing occlusion and background artifacts to improve feature matching. The result is
drift-free camera tracking with correct metric scale in the global LiDAR coordinate system. Two
real-time variants are presented: Online Render and Match (R&M), and Prebuild and Localize (P&L). We
demonstrate improved results on the ScanNet++ dataset and outperform existing SLAM pipelines.*

This repository contains only the code for our own methods (R&M and P&L, the latter with a
StellaVSLAM and an ORB-SLAM3 backend). Baseline methods we compared against in the paper are not
included here.

## Citation

```
@article{vanherck2026locin,
  author={Vanherck, Joni and Moonen, Steven and Zoomers, Brent and Werner, Kobe and Put, Jeroen and Jorissen, Lode and Michiels, Nick},
  journal={IEEE Robotics and Automation Letters}, 
  title={LoC-IN: Real-Time Localization of Cameras in Neural Rendered Prior LiDAR Maps}, 
  year={2026},
  volume={11},
  number={12},
  pages={14980-14987},
  keywords={Cameras;Rendering (computer graphics);Laser radar;Tracking;Location awareness;Clouds;Timing;Simultaneous localization and mapping;Accuracy;Trajectory;Localization;visual tracking;mapping},
  doi={10.1109/LRA.2026.3748685}
}
```

## Overview

Two variants are built from this repository:

- **Prebuild and Localize (P&L)** — build a SLAM map offline from the point cloud, then localize a
  video against it. Available with two interchangeable SLAM backends:
  - **StellaVSLAM** (binaries `map_stella` / `localize_stella`) — **recommended**: easiest to set
    up, no source patching required, and performs best in our evaluation.
  - **ORB-SLAM3** (binaries `map_orb` / `localize_orb`) — works, but needs some minor fixes to a
    stock checkout; StellaVSLAM is preferred unless you specifically need ORB-SLAM3.
- **Render and Match (R&M)** — one binary (`render_match`) that renders the point cloud directly
  against each incoming frame, with either XFeat or ORB features.

Both variants share the same offline-rendering/online-tracking split: a `render` step projects the
point cloud into cached frames, and a `map`/`build` step turns those cached frames into a queryable
map; `localize` always runs online against the live video.

## Installation

### Prerequisites (required for every variant)

- CMake ≥ 3.16, a C++17-capable compiler
- [OpenCV](https://opencv.org/) (≥ 4.x)
- [Eigen3](https://eigen.tuxfamily.org/) (≥ 3.4)
- [RTRenderer](https://github.com/EDM-Research/Real-time-Neural-Rendering-of-LiDAR-Point-Clouds.git)
  — the neural/projected point-cloud renderer shared by all variants

Clone with submodules (pulls in `DeviceFactory` and, for R&M, `xfeat_cpp`):

```
git clone --recurse-submodules https://github.com/UHasselt-DigitalFutureLab/LoC-IN.git LoC-IN
cd LoC-IN
```

### Prebuild & Localize — StellaVSLAM backend (recommended)

- [Stella_vSLAM with the Iridescence viewer](https://stella-cv.readthedocs.io/en/latest/overview.html#installation)

A slightly modified Stella_vSLAM repository for easier integration can be found [here](https://github.com/JVanherck/LoC-IN-stella_vslam).

### Prebuild & Localize — ORB-SLAM3 backend (optional)

- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (built as a shared library, `libORB_SLAM3.so`)
- Pangolin, Boost (`serialization` component), OpenSSL

**This backend needs some minor fixes applied to a vanilla ORB-SLAM3 checkout** before it will build
and run against this repository. A repository with fixes is provided [here](https://github.com/JVanherck/LoC-IN-ORB_SLAM3).

If no `find_package(ORB_SLAM3)` config is installed, point `ORB_SLAM3_ROOT_DIR` at your own checkout, 
or install an ORB-SLAM3 CMake package, as needed.

### Render & Match

- [libtorch](https://pytorch.org/) (`find_package(Torch)`)
- [fbow](https://github.com/rmsalinas/fbow)
- the `xfeat_cpp` submodule (pulled automatically with `--recurse-submodules`) and the bundled
  `render_match/scripts/eigenplaces_resnet50_512.pt` place-retrieval model

### Build

```
mkdir build && cd build
cmake ..
make
```

CMake auto-detects which SLAM backend(s) are available (`find_package(stella_vslam)` /
`find_package(ORB_SLAM3)`) and only builds the corresponding `map_*`/`localize_*` executables; at
least one backend must be found. `render_match` and `render_trajectory` are always built. Optionally
install with `sudo make install`.

### Bundled assets

The vocabulary/model files needed by each backend are already included in this repo:

- `prebuild_localize/configs/ORBvoc.txt` — standard ORB vocabulary, for the ORB-SLAM3 backend's
  `--voc` argument.
- `prebuild_localize/configs/orb_vocab.fbow` — StellaVSLAM's ORB vocabulary.
- `render_match/scripts/eigenplaces_resnet50_512.pt` — EigenPlaces place-retrieval model for R&M.

## Usage

### Prebuild and Localize

Both backends share the same `map_<backend>`/`localize_<backend>` CLI shape, where `<backend>` is
`stella` or `orb`.

**1. Build a map** — either as separate steps (so renders can be reused across backends/render
modes) or in one shot:

```
# Step 1: render & cache frames from the point cloud
map_<backend> render <folder> <calib> <cloud>

# Step 2: build the map from cached frames
map_<backend> build [backend-options] <folder> <calib>

# ...or do both at once:
map_<backend> full [backend-options] <folder> <calib> <cloud>
```

- `folder`: base folder containing all data; the map is written to `<folder>/.cache`.
- `calib`: calibration file relative to `<folder>` (see [calibration format](#calibration-format)).
- `cloud`: point cloud relative to `<folder>` (`.ply` or `.e57`), required for `render`/`full`.
- `--poses <path>`: reuse a previously generated camera-pose file instead of the interactive GUI.
- `--render-mode <neural|projected|filtered>` (default `neural`): which RTRenderer mode produces the
  cached frames; `--no-neural` is shorthand for `--render-mode projected`.
- ORB-SLAM3 only — `build`/`full` additionally require:
  - `--voc <path>`: path to `ORBvoc.txt`.
  - `--config <path>`: path to `prebuild_localize/configs/orbslam3_config_template.yaml`.

**2. Localize a video** against a built map, or build a new map directly from video:

```
localize_<backend> [options] <folder> <video> <calib> [map]
```

- `folder`, `calib`: as above.
- `video`: video to localize, relative to `<folder>`.
- `map` (optional): map file inside `<folder>/.cache` (`.msg` for Stella, `.osa` for ORB-SLAM3). If
  omitted, a new map is built live from the video instead of localizing against a prebuilt one.
- `--voc <path>` / `--config <path>`:
  - Stella: `prebuild_localize/configs/orb_vocab.fbow` / `prebuild_localize/configs/stella_config_template.yaml`.
  - ORB-SLAM3: `ORBvoc.txt` / `prebuild_localize/configs/orbslam3_config_template.yaml`.
- `--start-time <s>`: skip video frames until this timestamp before starting tracking (default 0).
- `--early-stop-frames <N>`: stop once N consecutive frames track successfully (default 0 = run to end).
- `--max-duration <s>`: give up if not relocalized within this much video time of `--start-time`
  (default 0 = no limit).

The estimated trajectory (and, for both backends, a companion per-frame matched-feature-count file)
is written to `<folder>/estimated/`.

### Render and Match

```
render_match <mode> [options] <arguments...>

render_match render   [opts] <folder> <cloud> <calibration>
render_match map      [opts] <folder> <cloud> <calibration>
render_match localize [opts] <folder> <cloud> <video> <calibration>
```

- `folder`, `cloud`, `calibration`: as above; `video` relative to `<folder>`.
- `--feature <xfeat|orb>` (default `xfeat`): feature backend used for matching.
- `--render-mode <neural|projected|filtered>` / `--no-neural`: as above.
- `--poses <path>`: reuse a previously generated pose file (`render`/`full` modes).
- `--eigenplaces-model <path>`: TorchScript place-retrieval model — pass the bundled
  `render_match/scripts/eigenplaces_resnet50_512.pt` to enable top-k relocalization candidates. Must
  be given at both `map` and `localize` time, or the map is built with zero place descriptors and
  falls back to a brute-force scan at load time.
- `--bow-vocab <path>`: FBoW vocabulary for ORB bag-of-words candidate selection (ORB feature mode).
- `--start-time <s>` / `--early-stop-frames <N>` / `--max-duration <s>`: as above (`localize` mode).

### Render trajectory

To visually inspect localization accuracy:

```
render_trajectory <folder> <cloud> <video_name> <calibration_file> <trajectory_file>
```

- `folder`: base folder; a new folder of side-by-side comparisons and blended images is created inside it.
- `cloud`: point cloud relative to `<folder>`.
- `video_name`: the video that was localized, relative to `<folder>`.
- `calibration_file`: as above.
- `trajectory_file`: the estimated trajectory relative to `<folder>`, in TUM format.

### Calibration format

If the calibration file is named `cameras.txt`, COLMAP output is expected:
`CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]`. Otherwise this format is expected:

```
W H
fx 0 cx
0 fy cy
0 0 1
k1 k2 p1 p2 k3
0
```

## About

![](/img/DFL_FlandersMake.jpg)

This work is a development by [Hasselt University](https://www.uhasselt.be/),
[Digital Future Lab](https://www.uhasselt.be/en/instituten-en/digitalfuturelab), funded by Hasselt
University and Research Foundation Flanders (FWO).
