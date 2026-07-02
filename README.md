# Stretch Dual Lidar Calibration

This repository contains the core calibration nodes and algorithms for aligning, floor-registering, and body-modeling dual LiDAR data on Hello Robot's Stretch.

## Installation

This is a standard ROS 2 Python package. You can install it using `colcon` or locally via `pip`:

```bash
# Via pip (editable mode) from the repository root
pip3 install -e . --break-system-packages
```

---

## Running Calibration

You should perform these calibration steps sequentially to properly configure your dual LiDAR setup. The calibration settings are automatically saved to your fleet directory (or `~/.stretch/calibration/`) so they can be securely loaded from any working directory later.

Prior to running body calibration, stow the robot’s arm by lowering the lift, retracting the telescoping arm, and configuring the wrist so that the wrist and end effector are within the footprint of the mobile base. This will enable the two LiDAR sensors to see the environment better.

You should also run the following ros2 launch files in separate terminals:

```
ros2 launch stretch_core stretch_driver.launch.py

ros2 launch stretch_core dual_hesai.launch.py  filter_type:=sor
```

### 1. Dual LiDAR Alignment Calibration

**Preconditions:** Prior to running dual LiDAR alignment calibration, make sure the robot is in a static environment without moving objects. The environment should also have enough geometric structure and size to enable scan matching to find a high-quality rigid body transform between scans taken from the left and right LiDAR.

```bash
ros_align_dual_lidar
```

### 2. Floor Calibration

**Preconditions:** Prior to running the floor calibration method, make sure that the robot is in the middle of a large flat floor that is visible to the robot. The larger, flatter, and more visible the floor is, the better the floor calibration will be.

```bash
ros_find_floor_calibration
```

### 3. Broadcast Calibration

**Preconditions:** Prior to running the visualizer or consuming data, you must broadcast the dual LiDAR alignment and the floor calibrations. You should leave this running in a separate terminal.

```bash
ros_broadcast_calibration
```

## Visualization

A dedicated node has been provided to publish a combined view of all the calibration outputs (unified point cloud, floor inliers):

```bash
ros_visualize_calibration
```

You can find the all-in-one RViz configuration in the `rviz` directory showing the unified views:

```bash
rviz2 -d rviz/dual_lidar_calibration.rviz
```

_(Ensure that `ros_broadcast_calibration` is running to resolve the `floor_plane` transform)._

## Example Calibration File

After using all of the calibration steps, you can find the calibration file at:

```bash
echo $HELLO_FLEET_PATH/$HELLO_FLEET_ID/calibration_dual_lidar/dual_lidar_calibration.yaml
```

An example of a calibration YAML file follows:

```yaml
right_to_left_transform:
  data:
    - - -0.5365967222717024
      - 0.6663353592972918
      - -0.5177462183323179
      - -0.12294484016345666
    - - -0.6646750085371393
      - 0.04424751619601933
      - 0.7458212187492647
      - 0.21060627682335334
    - - 0.5198760339479219
      - 0.7443381934716462
      - 0.4191533884700769
      - -0.11890210541737793
    - - 0.0
      - 0.0
      - 0.0
      - 1.0
  robot_id: stretch-se4-4010
  timestamp: "2026-03-19T14:28:32.548974"
floor_to_base_link_transform:
  data:
    - - 0.9999523981086859
      - 0.00018263214746540911
      - -0.009755417068829098
      - -2.710505431213761e-20
    - - 0.0
      - 0.9998248062402024
      - 0.018717821100267065
      - 5.421010862427522e-20
    - - 0.00975712645649783
      - -0.018716930096581413
      - 0.9997772126884428
      - 0.013265772694061559
    - - 0.0
      - 0.0
      - 0.0
      - 1.0
  robot_id: stretch-se4-4010
  timestamp: "2026-03-19T13:45:26.138201"
floor_model_params:
  data:
    normal:
      - 0.00975712645649783
      - -0.018716930096581413
      - 0.9997772126884428
    distance: -0.013265772694061559
    description: "Floor plane: normal [x,y,z] dot point + distance = 0"
  robot_id: stretch-se4-4010
  timestamp: "2026-03-19T13:45:26.138201"
```

## Calibration Details

This package uses a multi-step process to calibrate the dual LiDAR setup, align the robot with the floor, and define the robot's body mode.

### 1. Dual LiDAR Alignment (`ros_align_dual_lidar.py`)

The first step is to find the static rigid body transform between the two LiDARs ($T_{left \leftarrow right}$).

- **Data Collection**: The script collects synchronized scan pairs from both LiDARs.
- **Registration**: It uses **small_gicp** (Generalized Iterative Closest Point) to register the point cloud from the right LiDAR to the left LiDAR's frame.
- **Averaging**: To ensure robustness, the script performs this registration across multiple frames (default: 100 samples) and computes the average transform.
- **Result**: The computed transform is saved as `right_to_left_transform` in `dual_lidar_calibration.yaml` (located in the stretch_user calibration_dual_lidar directory).

### 2. Floor Plane Calibration (`ros_find_floor_calibration.py`)

The second step is to determine the floor plane relative to the robot's base link and compute the `base_footprint` frame. This ensures the robot's URDF model sits correctly on the ground.

- **Data Accumulation**: The script transforms point clouds from both LiDARs into the `base_link` frame (using the previously computed dual-lidar transform and URDF transforms) and accumulates them to form a dense representation of the scene.
- **Iterative Plane Fitting**: An iterative algorithm is used to robustly find the floor:
  1.  **Height Estimation**: A histogram of z-coordinates is used to find the approximate height of the floor, assuming it is the largest horizontal surface near the robot's feet.
  2.  **Outlier Rejection**: Points are filtered based on their distance from the current estimated plane model. The threshold significantly tightens over iterations (from 10cm down to 3mm).
  3.  **Normal Estimation**: **SVD (Singular Value Decomposition)** is performed on the inlying points to find the plane normal (the eigenvector corresponding to the smallest eigenvalue).
  4.  **Refinement**: The normal and height are iteratively updated until convergence.
- **Frame Computation**: The `base_footprint` frame is defined such that:
  - Its origin is on the floor plane, directly below the `base_link` origin.
  - Its Z-axis is aligned with the floor normal.
  - Its X-axis is aligned with the projection of the `base_link` X-axis onto the floor.
- **Result**: This transform maps points from the `base_link` frame to the `base_footprint` frame. The transform `floor_to_base_link_transform` is saved to `dual_lidar_calibration.yaml` and as a static calibration value (the `base_ref` joint) in `stretch_calibration_values.yaml`.

## Body Shape Modeling

The Body Shape Model provides an advanced mechanism for constructing bounds modeling the actual 2D boundaries of the robot during operation, resulting in an independent, highly tunable calibration file.

### Motivation

A manipulator like Stretch physically elongates its collision footprint as its telescoping arm extends outwards. A static circle wastes navigable space when the arm is stowed and might not accurately shield the arm when extended. By generalizing the tracking to various shapes (such as a circle, ellipse, or tapered capsule bounding box) and coupling it to kinematics states, obstacle detection can maintain tighter margins correctly matched to the actual extension.

### How It Works & Available Shapes

The shape model routine works by capturing LiDAR data across varying dynamic configurations:

1. **Data Collection**: The ROS node iteratively sweeps the robot through programmed configurations, extracting massive point cloud cross-sections of the robot mapped against its immediate kinematics state.
2. **Model Fitting**: For each kinematics step, LiDAR data is mapped to a high-resolution 2D floor histogram to remove noise. The script then applies an optimizer to construct a highly accurate bounding footprint contour matching the geometric scatter.
3. **Available Shape Constraints**:
   - **`circle`**: Finds the minimal enclosing circle wrapping the physical footprint. Constant, legacy-style behavior capable.
   - **`ellipse_opencv`**: Uses algebraic distance minimization. Fast and generally robust.
   - **`ellipse_min_enclosing`**: Computes an optimal minimal area bounding ellipse ensuring total coverage (Khachiyan's algorithm).
   - **`ellipse_axis_aligned`**: Solves for a minimal bounding ellipse whose axes are strictly aligned with the robot's base coordinate frame (X/Y axes). Ideal for bilateral symmetry and avoiding rotational jumps.
   - **`tapered_capsule` (Default)**: Finds the minimal bounding contour forming a tapered capsule (the convex hull of two circles). This shape is ideal for the Stretch base + arm geometry.
4. **Data Parameterization**: The configured footprints are logged into a dynamic parameters YAML file permitting real-time interpolation of limits relative to kinematics joints. Users can toggle dependencies on arm extension (`FIT_ARM_EXTENSION_DEPENDENT_BODY_SHAPE`) and wrist yaw (`FIT_WRIST_YAW_DEPENDENT_BODY_SHAPE`). Note that if properties are disabled, the shape simplifies dynamically down to a static rigid bounding box.

### Visualizing Shape Capabilities

The RViz inspection nodes draw highly detailed boundary profiles augmented by transparent shaded overlays to visually help analyze the fitted bounds and calibration tolerances in real-time. Three distinct boundaries and two critical polygon shaded regions are published dynamically:

- **Minimum Model**: Bounded by a thin blue line and shaded with a transparent blue interior, this marks the bare-metal output of the optimization algorithm outlining the mathematical minimal envelope strictly enclosing the physical bounds of the robot.
- **Inner Boundary**: The inner red curve defining the structural margin buffer. The gap extending between the inner boundary and the blue minimum model explicitly operates as a dead zone to account for modeling errors or sensor noise; any LiDAR hits landing within this bare gap are completely ignored.
- **Obstacle Detection Region**: Shaded with a transparent red overlay between the inner and outer red boundary curves, this spans the active "Stop Zone". Any LiDAR reflections striking within this highlighted red region trigger proximity detections distinguishing against collision obstacles.

### Tuning & Parameter Configuration

The entire sequence is deeply tunable. Configurable aspects include iteration ranges for spatial sweeps, stop zone margins, and algorithmic curve solvers. **To customize these features or switch to a different fitting algorithm, edit the variables defined within the settings code module:**
`stretch_dual_lidar_calibration/stretch_dual_lidar_calibration/body_shape_calibration_params.py`

### Usage Directions

This procedure requires access to unified data streams; guarantee that the background `ros_broadcast_calibration` is already active to relay the core `/tf` transforms.

1. **Collect Data**: Execute the automated collection sequence. By default, it manages folder naming automatically, but you can explicitly specify an output directory identifier.
2. **Fit the Shapes**: Route the resulting timestamped directory through to the shape engine.
3. **Verify the Output**: Launch RViz and inspect the new limits bounding curves.

**Example Sequence:**

_Terminal 1:_

```bash
# This creates a folder like 'collected_body_shape_data_2026xxxx_xxxxxx'
# Note: You can also specify a custom directory name as a first argument.
ros_collect_body_shape_data

fit_body_shape_model ./collected_body_shape_data_.../

ros_visualize_body_shape_calibration ./tapered_capsule_body_model_TIMESTAMP.yaml
```

_Terminal 2:_

```bash
rviz2 -d ./rviz/body_shape_calibration.rviz
```

---

## Acknowledgments

### small_gicp

stretch_robosense makes use of [small_gicp](https://github.com/koide3/small_gicp), which is a library for fast 3D lidar scan registration. Kenji Koide from National Institute of Advanced Industrial Science and Technology (AIST) is the creator of small_gicp. The repository was released in 2024 with an MIT License.

If you use small*gicp as part of stretch_dual_lidar, please cite it using the following citation and consider leaving a comment [here](https://github.com/koide3/small_gicp/issues/) as requested by Kenji Koide. *"It would help the author receive recognition in his organization and keep working on this project."\_ - [small_gicp GitHub repository](https://github.com/koide3/small_gicp),

```
@article{small_gicp,
author = {Kenji Koide},
title = {{small\_gicp: Efficient and parallel algorithms for point cloud registration}},
journal = {Journal of Open Source Software},
month = aug,
number = {100},
pages = {6948},
volume = {9},
year = {2024},
doi = {10.21105/joss.06948}
}
```

Thank you Kenji Koide and other contributors for this helpful code!
