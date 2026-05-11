"""
body_shape_calibration_params.py

This configuration file defines the parameters used for collecting data, fitting,
and utilizing the dynamically scaling shape body model for the robot.
"""
import numpy as np

# -----------------------------------------------------------------------------
# Data Collection Parameters
# -----------------------------------------------------------------------------

# Number of samples and ranges for joint configurations
# These dictate the grid of configurations the robot will move through to collect data.

# Lift positions: [MIN (m), MAX (m), SAMPLES]
LIFT_CONFIG_RANGE = [0.1, 1.0, 6]

# Arm extensions: [MIN (m), MAX (m), SAMPLES]
ARM_CONFIG_RANGE = [0.0, 0.49, 6] 

# Wrist yaw (degrees): [MIN (deg), MAX (deg), SAMPLES]
WRIST_YAW_RANGE = [-60, 120, 6] #[-20.0, 20.0, 3]

# Wrist pitch (degrees): [MIN (deg), MAX (deg), SAMPLES]
WRIST_PITCH_RANGE = [0.0, 0.0, 1]

# Wrist roll (degrees): [MIN (deg), MAX (deg), SAMPLES]
WRIST_ROLL_RANGE = [0.0, 0.0, 1]

# Delay in seconds to wait after commanding joints before capturing constraints (settle time)
SETTLE_TIME_S = 0.5

# Generate evenly spaced arrays from the ranges above:
LIFT_POSITIONS = np.linspace(LIFT_CONFIG_RANGE[0], LIFT_CONFIG_RANGE[1], int(LIFT_CONFIG_RANGE[2])).tolist()
ARM_POSITIONS = np.linspace(ARM_CONFIG_RANGE[0], ARM_CONFIG_RANGE[1], int(ARM_CONFIG_RANGE[2])).tolist()

# Conversions for wrist angles from degrees to radians
WRIST_YAW_POSITIONS = np.deg2rad(np.linspace(WRIST_YAW_RANGE[0], WRIST_YAW_RANGE[1], int(WRIST_YAW_RANGE[2]))).tolist()
WRIST_PITCH_POSITIONS = np.deg2rad(np.linspace(WRIST_PITCH_RANGE[0], WRIST_PITCH_RANGE[1], int(WRIST_PITCH_RANGE[2]))).tolist()
WRIST_ROLL_POSITIONS = np.deg2rad(np.linspace(WRIST_ROLL_RANGE[0], WRIST_ROLL_RANGE[1], int(WRIST_ROLL_RANGE[2]))).tolist()


# -----------------------------------------------------------------------------
# Shape Model Fitting Parameters
# -----------------------------------------------------------------------------

# Bounding box for floor/robot points. Since the arm extends up to 0.4m (plus margin), we widen x_max 
# considerably compared to the static circular model (which used x_max=0.2).
FIT_X_MIN = -0.5
FIT_X_MAX = 1.5
FIT_Y_MIN = -0.5
FIT_Y_MAX = 0.5

# Z height boundaries used to classify points as belonging to the robot body
# Heights must be greater than MIN and less than MAX
FIT_Z_MIN = 0.06
FIT_Z_MAX = 1.6

# Histogram mapping constants used for robust ellipse contouring
HISTOGRAM_RESOLUTION_M = 0.03 #0.01  # Meters per pixel for histogram rendering
HISTOGRAM_THRESHOLD = 20 #5    # Minimum point count inside a bin to be considered part of the body

# Fitting algorithm configuration
# Available methods:
# 'circle' : Finds a minimal area bounding circle enclosing the robot.
# 'ellipse_opencv' : Uses cv2.fitEllipse, which minimizes algebraic distance. Usually robust and fast.
# 'ellipse_min_enclosing' : Finds a minimal area bounding ellipse using Khachiyan's algorithm. 
#                   Slower but ensures a guaranteed bounding shape.
# 'ellipse_axis_aligned' : Finds a minimal area bounding ellipse whose principal axes are strictly
#                  aligned with the robot's base coordinate frame (X/Y axes). Ideal for bilaterally
#                  symmetric robots to preserve mathematical alignment and avoid rotational spinning.
# 'tapered_capsule' : Finds a minimal area bounding tapered capsule shape (convex hull of two circles).
SHAPE_FITTING_METHOD = 'tapered_capsule'

# Interpolation configuration
# Options: 'linear', 'spline'
# This dictates how intermediate arm extensions are approximated
INTERPOLATION_METHOD = 'spline' #'linear'

# If True, the generated body shape parameters will depend on the arm extension.
# If False, the generated body shape model will be constant regardless of arm extension.
FIT_ARM_EXTENSION_DEPENDENT_BODY_SHAPE = True

# If True, the generated body shape parameters will depend on BOTH the arm extension
# (if enabled) and wrist yaw.
# If False, the shape will not depend on the wrist yaw.
FIT_WRIST_YAW_DEPENDENT_BODY_SHAPE = True

# -----------------------------------------------------------------------------
# Obstacle Detection Model Parameters
# -----------------------------------------------------------------------------

# Radial margins dynamically added to the axes of the fitted min_shape to build obstacle detector zones
BODY_SHAPE_MARGIN_M = 0.05  # Distance expanded evenly all around to form the `inner_shape`
STOP_ZONE_M = 0.1           # Added to the margin to form the outer boundary `outer_shape`

# Height limits for general obstacle detection
MIN_POSITIVE_OBSTACLE_HEIGHT_M = 0.06
MAX_POSITIVE_OBSTACLE_HEIGHT_M = 1.6
MAX_NEGATIVE_OBSTACLE_HEIGHT_M = -0.06
