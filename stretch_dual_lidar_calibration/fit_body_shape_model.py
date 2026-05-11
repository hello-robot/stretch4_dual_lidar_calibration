#!/usr/bin/env python3
import numpy as np
import cv2
import os
import yaml
import argparse
from scipy.interpolate import interp1d, CubicSpline

from stretch_dual_lidar_calibration.dual_lidar_calibration import DualLidarCalibration
from stretch_dual_lidar_calibration.histogram_utils import (
    create_floor_histogram,
    floor_histogram_bins_to_points
)

import time
from stretch_dual_lidar_calibration.body_shape_calibration_params import (
    FIT_X_MIN, FIT_X_MAX, FIT_Y_MIN, FIT_Y_MAX,
    FIT_Z_MIN, FIT_Z_MAX,
    HISTOGRAM_RESOLUTION_M, HISTOGRAM_THRESHOLD,
    SHAPE_FITTING_METHOD,
    INTERPOLATION_METHOD
)
import stretch_dual_lidar_calibration.body_shape_calibration_params as params_module

def khachiyan_algorithm(P, tolerance=1e-3):
    """
    Finds the minimum area enclosing ellipse of the 2D point cloud P.
    P: (2, N) numpy array of points.
    Returns: center_x, center_y, width, height, angle_degrees
    Angle is rotated clockwise from X axis, similar to cv2.fitEllipse output.
    """
    d, N = P.shape
    Q = np.vstack([P, np.ones((1, N))])
    count = 1
    err = 1.0
    u = np.ones(N) / N
    # Prevent infinite loop for rank deficient point sets
    while err > tolerance and count < 1000:
        X = Q @ np.diag(u) @ Q.T
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            break
        M = np.diag(Q.T @ X_inv @ Q)
        max_idx = np.argmax(M)
        step_size = (M[max_idx] - d - 1) / ((d + 1) * (M[max_idx] - 1))
        new_u = (1 - step_size) * u
        new_u[max_idx] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
        count += 1
    
    center = P @ u
    U = np.diag(u) - np.outer(u, u)
    try:
        M2 = P @ U @ P.T
        M2_inv = np.linalg.inv(M2)
        A = (1 / d) * M2_inv
    except np.linalg.LinAlgError:
        return float(center[0]), float(center[1]), 0.0, 0.0, 0.0
        
    w, v = np.linalg.eigh(A)
    # The axes lengths are 2/sqrt(w_i)
    # Add epsilon to prevent division by zero for totally singular sets
    axes = 2.0 / np.sqrt(abs(w) + 1e-12)
    
    # In cv2.fitEllipse, angle is usually between [0, 180) representing
    # the angle to the major axis.
    # v[:, 0] is the eigenvector for the first axis.
    angle_rad = np.arctan2(v[1, 0], v[0, 0])
    angle_deg = np.rad2deg(angle_rad)
    
    return float(center[0]), float(center[1]), float(axes[0]), float(axes[1]), float(angle_deg % 180)

def axis_aligned_min_enclosing_ellipse(P):
    """
    Finds the minimum area axis-aligned bounding ellipse using SLSQP constrained optimization.
    P: (2, N) numpy array of points.
    Returns: cx, cy, width, height, 0.0
    """
    from scipy.optimize import minimize
    
    if P.shape[1] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    x = P[0, :]
    y = P[1, :]
    
    # Initial guess bounds
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    cx0 = (min_x + max_x) / 2.0
    cy0 = (min_y + max_y) / 2.0
    
    # A circle bounding a square of side L has radius (L * sqrt(2) / 2).
    # We use 1.5 as a safely circumscribing initial scale.
    a0 = max(1e-3, (max_x - min_x) / 2.0 * 1.5)
    b0 = max(1e-3, (max_y - min_y) / 2.0 * 1.5)
    
    def objective(v):
        # Minimize Area ~ a * b
        return v[2] * v[3]
        
    def constraint(v):
        # Enforce all points lie within the ellipse formula
        cx, cy, a, b = v
        return 1.0 - (((x - cx)**2) / (a**2) + ((y - cy)**2) / (b**2))
        
    bounds = [(None, None), (None, None), (1e-4, None), (1e-4, None)]
    cons = {'type': 'ineq', 'fun': constraint}
    
    # Optimization takes ~1-2 iterations since N is small from the Convex Hull
    res = minimize(
        objective, 
        x0=[cx0, cy0, a0, b0], 
        bounds=bounds, 
        constraints=cons, 
        method='SLSQP',
        options={'ftol': 1e-6, 'maxiter': 500}
    )
    
    if res.success:
        cx, cy, a, b = res.x
        return float(cx), float(cy), float(a * 2.0), float(b * 2.0), 0.0
    else:
        # Fallback to the circumscribed bounding box if solver fails
        return float(cx0), float(cy0), float(a0 * 2.0), float(b0 * 2.0), 0.0

def min_enclosing_tapered_capsule(P):
    """
    Finds the minimal area enclosing tapered capsule of 2D points.
    P: (2, N) numpy array of points.
    Returns: x1, y1, r1, x2, y2, r2
    """
    from scipy.optimize import minimize
    
    if P.shape[1] == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    x = P[0, :]
    y = P[1, :]
    
    # Initial guess bounds
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    wx = max_x - min_x
    wy = max_y - min_y
    if wx > wy:
        x10, x20 = min_x + wx*0.25, max_x - wx*0.25
        y10, y20 = cy, cy
        r10, r20 = wy/2.0, wy/2.0
    else:
        x10, x20 = cx, cx
        y10, y20 = min_y + wy*0.25, max_y - wy*0.25
        r10, r20 = wx/2.0, wx/2.0
        
    r10 = max(1e-3, r10 * 1.2)
    r20 = max(1e-3, r20 * 1.2)
    
    def objective(v):
        x1, y1, r1, x2, y2, r2 = v
        d = np.sqrt((x2-x1)**2 + (y2-y1)**2) + 1e-12
        if d <= abs(r1 - r2):
            return np.pi * max(r1, r2)**2
        diff = r1 - r2
        s = np.clip(diff / d, -1.0, 1.0)
        theta = np.arcsin(s)
        return np.pi/2 * (r1**2 + r2**2) + theta * (r1**2 - r2**2) + np.sqrt(max(0, d**2 - diff**2)) * (r1 + r2)
        
    def constraint(v):
        x1, y1, r1, x2, y2, r2 = v
        ba_x, ba_y = x2 - x1, y2 - y1
        ba_sq = ba_x**2 + ba_y**2
        dr = r2 - r1
        a = ba_sq - dr**2
        pa_x, pa_y = x - x1, y - y1
        b = (pa_x * ba_x + pa_y * ba_y) + r1 * dr
        
        h = np.zeros_like(b)
        mask = a > 1e-8
        h[mask] = b[mask] / a
        h = np.clip(h, 0.0, 1.0)
        
        pa_sq = pa_x**2 + pa_y**2
        f_h = a * (h**2) - 2 * b * h + pa_sq - r1**2
        return -f_h
        
    bounds = [(None, None), (None, None), (1e-4, None), (None, None), (None, None), (1e-4, None)]
    cons = {'type': 'ineq', 'fun': constraint}
    
    res = minimize(objective, x0=[x10, y10, r10, x20, y20, r20], bounds=bounds, constraints=cons, method='SLSQP', options={'ftol': 1e-6, 'maxiter': 500})
    if res.success:
        return tuple(float(val) for val in res.x)
    return float(x10), float(y10), float(r10), float(x20), float(y20), float(r20)

def convert_to_native(data):
    if isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: convert_to_native(v) for k, v in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [convert_to_native(v) for v in data]
    else:
        return data

def main():
    parser = argparse.ArgumentParser(description="Fit shape body models to collected data.")
    parser.add_argument('data_dir', type=str, help='Directory containing the collected npz and metadata.yaml.')
    args = parser.parse_args()
    
    metadata_path = os.path.join(args.data_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
        
    # Check if model should be dependent on properties
    arm_dependent = getattr(params_module, 'FIT_ARM_EXTENSION_DEPENDENT_BODY_SHAPE', True)
    yaw_dependent = getattr(params_module, 'FIT_WRIST_YAW_DEPENDENT_BODY_SHAPE', False)
    
    samples_by_config = {}
    for sample in metadata['samples']:
        arm_ext = round(sample['commanded']['arm'], 3)
        wrist_yaw = round(sample['commanded'].get('wrist_yaw', 0.0), 3)
        
        if arm_dependent and yaw_dependent:
            key = (arm_ext, wrist_yaw)
        elif arm_dependent:
            key = arm_ext
        elif yaw_dependent:
            key = wrist_yaw
        else:
            key = "constant"
            
        if key not in samples_by_config:
            samples_by_config[key] = []
        samples_by_config[key].append(sample)
        
    arm_extensions = sorted(list(set([round(s['commanded']['arm'], 3) for s in metadata['samples']])))
    wrist_yaws = sorted(list(set([round(s['commanded'].get('wrist_yaw', 0.0), 3) for s in metadata['samples']])))
    
    config_keys = []
    if arm_dependent and yaw_dependent:
        print(f"Found {len(samples_by_config)} unique combinations of arm extension and wrist yaw.")
        for a in arm_extensions:
            for y in wrist_yaws:
                config_keys.append((a, y))
    elif arm_dependent:
        print(f"Found {len(samples_by_config)} unique arm extensions.")
        config_keys = arm_extensions
    elif yaw_dependent:
        print(f"Found {len(samples_by_config)} unique wrist yaws.")
        config_keys = wrist_yaws
    else:
        print(f"Combining all samples into a single constant model.")
        config_keys = ["constant"]
    
    shapes = []
    
    cuboid = {
        'x_min': FIT_X_MIN, 'x_max': FIT_X_MAX,
        'y_min': FIT_Y_MIN, 'y_max': FIT_Y_MAX
    }
    
    for config_key in config_keys:
        if arm_dependent and yaw_dependent:
            print(f"Processing arm extension: {config_key[0]} m, wrist yaw: {config_key[1]} rad ...")
        elif arm_dependent:
            print(f"Processing arm extension: {config_key} m ...")
        elif yaw_dependent:
            print(f"Processing wrist yaw: {config_key} rad ...")
        else:
            print(f"Processing constant shape ...")
            
        all_points = []
        if config_key in samples_by_config:
            for sample in samples_by_config[config_key]:
                npz_file = os.path.join(args.data_dir, sample['file'])
                if not os.path.exists(npz_file):
                    continue
                data = np.load(npz_file)
                pts = data['points']
                
                # Filter for robot body candidates (Z range)
                z_mask = (pts[:, 2] > FIT_Z_MIN) & (pts[:, 2] < FIT_Z_MAX)
                pts_z = pts[z_mask]
                
                # Filter X, Y
                xy_mask = (pts_z[:, 0] > FIT_X_MIN) & (pts_z[:, 0] < FIT_X_MAX) & \
                          (pts_z[:, 1] > FIT_Y_MIN) & (pts_z[:, 1] < FIT_Y_MAX)
                
                valid_pts = pts_z[xy_mask]
                if len(valid_pts) > 0:
                    all_points.append(valid_pts)
                
        if len(all_points) == 0:
            if arm_dependent and yaw_dependent:
                print(f"  Warning: No points found for config (arm: {config_key[0]}, yaw: {config_key[1]})!")
            elif arm_dependent:
                print(f"  Warning: No points found for arm extension {config_key}!")
            elif yaw_dependent:
                print(f"  Warning: No points found for wrist yaw {config_key}!")
            else:
                print(f"  Warning: No points found for constant shape!")
                
            if SHAPE_FITTING_METHOD.startswith('ellipse'):
                shapes.append({'center_x': 0.0, 'center_y': 0.0, 'axes_x': 0.0, 'axes_y': 0.0, 'angle': 0.0})
            elif SHAPE_FITTING_METHOD == 'circle':
                shapes.append({'center_x': 0.0, 'center_y': 0.0, 'radius': 0.0})
            else:
                shapes.append({'x1': 0.0, 'y1': 0.0, 'r1': 0.0, 'x2': 0.0, 'y2': 0.0, 'r2': 0.0})
            continue
            
        combined_points = np.vstack(all_points)
        print(f"  Total points: {len(combined_points)}")
        
        # 2D Histogram downsampling
        histogram, _, _, _, _ = create_floor_histogram(combined_points, cuboid, HISTOGRAM_RESOLUTION_M)
        valid_bins_mask = histogram >= HISTOGRAM_THRESHOLD
        valid_indices = np.argwhere(valid_bins_mask)
        
        if len(valid_indices) < 5:
            print(f"  Warning: Not enough valid histogram bins for shape fitting (count={len(valid_indices)})")
            if SHAPE_FITTING_METHOD.startswith('ellipse'):
                shapes.append({'center_x': 0.0, 'center_y': 0.0, 'axes_x': 0.0, 'axes_y': 0.0, 'angle': 0.0})
            elif SHAPE_FITTING_METHOD == 'circle':
                shapes.append({'center_x': 0.0, 'center_y': 0.0, 'radius': 0.0})
            else:
                shapes.append({'x1': 0.0, 'y1': 0.0, 'r1': 0.0, 'x2': 0.0, 'y2': 0.0, 'r2': 0.0})
            continue
            
        points_for_ellipse = floor_histogram_bins_to_points(valid_indices, cuboid, HISTOGRAM_RESOLUTION_M)
        
        if SHAPE_FITTING_METHOD == 'ellipse_opencv':
            pts_cv = points_for_ellipse[:, :2].astype(np.float32).reshape(-1, 1, 2)
            (cx, cy), (ax_w, ax_h), angle = cv2.fitEllipse(pts_cv)
        elif SHAPE_FITTING_METHOD == 'ellipse_min_enclosing':
            pts_cv = points_for_ellipse[:, :2].astype(np.float32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts_cv)
            pts_k = hull.reshape(-1, 2).T
            cx, cy, ax_w, ax_h, angle = khachiyan_algorithm(pts_k)
        elif SHAPE_FITTING_METHOD == 'ellipse_axis_aligned':
            pts_cv = points_for_ellipse[:, :2].astype(np.float32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts_cv)
            pts_k = hull.reshape(-1, 2).T
            cx, cy, ax_w, ax_h, angle = axis_aligned_min_enclosing_ellipse(pts_k)
        elif SHAPE_FITTING_METHOD == 'circle':
            pts_cv = points_for_ellipse[:, :2].astype(np.float32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts_cv)
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
        elif SHAPE_FITTING_METHOD == 'tapered_capsule':
            pts_cv = points_for_ellipse[:, :2].astype(np.float32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts_cv)
            pts_k = hull.reshape(-1, 2).T
            x1, y1, r1, x2, y2, r2 = min_enclosing_tapered_capsule(pts_k)
            # Canonicalize: Ensure sorting x1 <= x2 for stable interpolation
            if x1 > x2:
                x1, y1, r1, x2, y2, r2 = x2, y2, r2, x1, y1, r1
        else:
            raise ValueError(f"Unknown fitting method: {SHAPE_FITTING_METHOD}")
            
        if SHAPE_FITTING_METHOD.startswith('ellipse'):
            if len(shapes) > 0:
                prev_w = shapes[-1]['axes_x']
                prev_h = shapes[-1]['axes_y']
                prev_angle = shapes[-1]['angle']
                
                candidates = []
                for k in [-2, -1, 0, 1, 2]:
                    candidates.append((ax_w, ax_h, angle + 180.0 * k))
                    candidates.append((ax_h, ax_w, angle + 90.0 + 180.0 * k))
                    
                best_dist = float('inf')
                best_w, best_h, best_angle = ax_w, ax_h, angle
                
                for cw, ch, c_theta in candidates:
                    dist = (cw - prev_w)**2 + (ch - prev_h)**2 + ((c_theta - prev_angle) * 0.01)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_w, best_h, best_angle = cw, ch, c_theta
                        
                ax_w, ax_h, angle = best_w, best_h, best_angle
    
            print(f"  Result -> Center: ({cx:.3f}, {cy:.3f}), Axes: ({ax_w:.3f}, {ax_h:.3f}), Angle: {angle:.2f}")
            shapes.append({
                'center_x': float(cx),
                'center_y': float(cy),
                'axes_x': float(ax_w),
                'axes_y': float(ax_h),
                'angle': float(angle)
            })
        elif SHAPE_FITTING_METHOD == 'circle':
            print(f"  Result -> Center: ({cx:.3f}, {cy:.3f}), Radius: {radius:.3f}")
            shapes.append({
                'center_x': float(cx),
                'center_y': float(cy),
                'radius': float(radius)
            })
        else:
            print(f"  Result -> C1: ({x1:.3f}, {y1:.3f}, {r1:.3f}), C2: ({x2:.3f}, {y2:.3f}, {r2:.3f})")
            shapes.append({
                'x1': float(x1), 'y1': float(y1), 'r1': float(r1),
                'x2': float(x2), 'y2': float(y2), 'r2': float(r2)
            })
        
    print(f"\nConstructing continuous {INTERPOLATION_METHOD} model...")
    
    # We will save the exact discrete data and use interpolation natively in the obstacle detector.
    # Saving the parameters and explicit grid makes the YAML clean and easy to interpret.
    
    model_params = {
        'arm_dependent': arm_dependent,
        'yaw_dependent': yaw_dependent,
        'arm_extensions': [float(x) for x in arm_extensions],
        'wrist_yaws': [float(y) for y in wrist_yaws],
        'interpolation_method': INTERPOLATION_METHOD,
        'shape_fitting_method': SHAPE_FITTING_METHOD,
        'parameters': shapes,
        'dataset_metadata': {
            'robot_id': metadata.get('robot_id', 'unknown'),
            'tool_id': metadata.get('tool_id', 'unknown'),
            'timestamp': metadata.get('collection_timestamp', 0)
        },
        'calibration_params_snapshot': {
            k: convert_to_native(getattr(params_module, k)) for k in dir(params_module)
            if not k.startswith('_') 
            and not callable(getattr(params_module, k)) 
            and type(getattr(params_module, k)).__name__ != 'module'
            and k not in [
                'BODY_SHAPE_MARGIN_M', 'STOP_ZONE_M', 
                'MIN_POSITIVE_OBSTACLE_HEIGHT_M', 'MAX_POSITIVE_OBSTACLE_HEIGHT_M', 
                'MAX_NEGATIVE_OBSTACLE_HEIGHT_M'
            ]
        }
    }
    
    model_data = {
        'custom_body_shape_model_params': model_params
    }
    
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    dep_str_parts = []
    if arm_dependent: dep_str_parts.append('arm')
    if yaw_dependent: dep_str_parts.append('yaw')
    dep_str = '_'.join(dep_str_parts) + "_dependent" if dep_str_parts else "constant"
    output_yaml = f"{SHAPE_FITTING_METHOD}_{dep_str}_body_model_{timestamp_str}.yaml"
    
    with open(output_yaml, 'w') as f:
        yaml.dump(convert_to_native(model_data), f, default_flow_style=False)
        
    print(f"Successfully saved {output_yaml}")

if __name__ == '__main__':
    main()
