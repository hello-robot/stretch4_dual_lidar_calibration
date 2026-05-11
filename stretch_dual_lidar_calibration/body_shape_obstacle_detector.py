import numpy as np
from scipy.interpolate import interp1d, CubicSpline, RegularGridInterpolator
from stretch_dual_lidar_calibration.body_shape_calibration_params import (
    MIN_POSITIVE_OBSTACLE_HEIGHT_M,
    MAX_POSITIVE_OBSTACLE_HEIGHT_M,
    MAX_NEGATIVE_OBSTACLE_HEIGHT_M,
    BODY_SHAPE_MARGIN_M,
    STOP_ZONE_M
)

class ShapeObstacleDetector:
    def __init__(self, shape_model_params):
        """
        shape_model_params is the dict loaded from the yaml under 'custom_body_shape_model_params'
        """
        self.arm_dependent = shape_model_params.get('arm_dependent', True)
        self.yaw_dependent = shape_model_params.get('yaw_dependent', False)
        
        self.arm_extensions = np.array(shape_model_params.get('arm_extensions', [0.0])) if self.arm_dependent else np.array([])
        if self.yaw_dependent:
            self.wrist_yaws = np.array(shape_model_params.get('wrist_yaws', [0.0]))
        else:
            self.wrist_yaws = np.array([])
            
        self.method = shape_model_params.get('interpolation_method', 'linear')
        self.shape_fitting_method = shape_model_params.get('shape_fitting_method', 'ellipse_axis_aligned')
        self.is_ellipse = self.shape_fitting_method.startswith('ellipse')
        self.is_circle = self.shape_fitting_method == 'circle'
        
        params = shape_model_params.get('parameters', [])
        
        if self.is_ellipse:
            if not params:
                params = [{'center_x':0, 'center_y':0, 'axes_x':0, 'axes_y':0, 'angle':0}]
            self.cxs = np.array([p.get('center_x', 0) for p in params])
            self.cys = np.array([p.get('center_y', 0) for p in params])
            self.axs = np.array([p.get('axes_x', 0) for p in params])
            self.ays = np.array([p.get('axes_y', 0) for p in params])
            self.angles = np.array([p.get('angle', 0) for p in params])
        elif self.is_circle:
            if not params:
                params = [{'center_x':0, 'center_y':0, 'radius':0}]
            self.cxs = np.array([p.get('center_x', 0) for p in params])
            self.cys = np.array([p.get('center_y', 0) for p in params])
            self.rs = np.array([p.get('radius', 0) for p in params])
        else:
            if not params:
                params = [{'x1':0, 'y1':0, 'r1':0, 'x2':0, 'y2':0, 'r2':0}]
            self.x1s = np.array([p.get('x1', 0) for p in params])
            self.y1s = np.array([p.get('y1', 0) for p in params])
            self.r1s = np.array([p.get('r1', 0) for p in params])
            self.x2s = np.array([p.get('x2', 0) for p in params])
            self.y2s = np.array([p.get('y2', 0) for p in params])
            self.r2s = np.array([p.get('r2', 0) for p in params])
        
        # Obstacle detection model parameters are pulled directly from body_shape_calibration_params.py 
        # at runtime, as they should be user-tunable without refitting the model.
        self.body_margin = BODY_SHAPE_MARGIN_M
        self.stop_zone = STOP_ZONE_M
        
        self._setup_interpolators()

    def _setup_interpolators(self):
        m = 'cubic' if self.method == 'spline' else 'linear'
        
        if self.yaw_dependent and self.arm_dependent:
            if len(self.arm_extensions) > 1 and len(self.wrist_yaws) > 1:
                Na = len(self.arm_extensions)
                Ny = len(self.wrist_yaws)
                grid = (self.arm_extensions, self.wrist_yaws)
                
                def make_2d_interp(arr):
                    arr_2d = arr.reshape((Na, Ny))
                    return RegularGridInterpolator(grid, arr_2d, method=m, bounds_error=False, fill_value=None)
                
                if self.is_ellipse:
                    self.interp_cx = make_2d_interp(self.cxs)
                    self.interp_cy = make_2d_interp(self.cys)
                    self.interp_ax = make_2d_interp(self.axs)
                    self.interp_ay = make_2d_interp(self.ays)
                    self.interp_angle = make_2d_interp(self.angles)
                elif self.is_circle:
                    self.interp_cx = make_2d_interp(self.cxs)
                    self.interp_cy = make_2d_interp(self.cys)
                    self.interp_r = make_2d_interp(self.rs)
                else:
                    self.interp_x1 = make_2d_interp(self.x1s)
                    self.interp_y1 = make_2d_interp(self.y1s)
                    self.interp_r1 = make_2d_interp(self.r1s)
                    self.interp_x2 = make_2d_interp(self.x2s)
                    self.interp_y2 = make_2d_interp(self.y2s)
                    self.interp_r2 = make_2d_interp(self.r2s)
        elif self.arm_dependent or self.yaw_dependent:
            dim_arr = self.arm_extensions if self.arm_dependent else self.wrist_yaws
            if len(dim_arr) > 1:
                if self.is_ellipse:
                    if self.method == 'spline':
                        self.interp_cx = CubicSpline(dim_arr, self.cxs, extrapolate=True)
                        self.interp_cy = CubicSpline(dim_arr, self.cys, extrapolate=True)
                        self.interp_ax = CubicSpline(dim_arr, self.axs, extrapolate=True)
                        self.interp_ay = CubicSpline(dim_arr, self.ays, extrapolate=True)
                        self.interp_angle = CubicSpline(dim_arr, self.angles, extrapolate=True)
                    else: # linear
                        self.interp_cx = interp1d(dim_arr, self.cxs, fill_value="extrapolate")
                        self.interp_cy = interp1d(dim_arr, self.cys, fill_value="extrapolate")
                        self.interp_ax = interp1d(dim_arr, self.axs, fill_value="extrapolate")
                        self.interp_ay = interp1d(dim_arr, self.ays, fill_value="extrapolate")
                        self.interp_angle = interp1d(dim_arr, self.angles, fill_value="extrapolate")
                elif self.is_circle:
                    if self.method == 'spline':
                        self.interp_cx = CubicSpline(dim_arr, self.cxs, extrapolate=True)
                        self.interp_cy = CubicSpline(dim_arr, self.cys, extrapolate=True)
                        self.interp_r = CubicSpline(dim_arr, self.rs, extrapolate=True)
                    else: # linear
                        self.interp_cx = interp1d(dim_arr, self.cxs, fill_value="extrapolate")
                        self.interp_cy = interp1d(dim_arr, self.cys, fill_value="extrapolate")
                        self.interp_r = interp1d(dim_arr, self.rs, fill_value="extrapolate")
                else:
                    if self.method == 'spline':
                        self.interp_x1 = CubicSpline(dim_arr, self.x1s, extrapolate=True)
                        self.interp_y1 = CubicSpline(dim_arr, self.y1s, extrapolate=True)
                        self.interp_r1 = CubicSpline(dim_arr, self.r1s, extrapolate=True)
                        self.interp_x2 = CubicSpline(dim_arr, self.x2s, extrapolate=True)
                        self.interp_y2 = CubicSpline(dim_arr, self.y2s, extrapolate=True)
                        self.interp_r2 = CubicSpline(dim_arr, self.r2s, extrapolate=True)
                    else: # linear
                        self.interp_x1 = interp1d(dim_arr, self.x1s, fill_value="extrapolate")
                        self.interp_y1 = interp1d(dim_arr, self.y1s, fill_value="extrapolate")
                        self.interp_r1 = interp1d(dim_arr, self.r1s, fill_value="extrapolate")
                        self.interp_x2 = interp1d(dim_arr, self.x2s, fill_value="extrapolate")
                        self.interp_y2 = interp1d(dim_arr, self.y2s, fill_value="extrapolate")
                        self.interp_r2 = interp1d(dim_arr, self.r2s, fill_value="extrapolate")
                
    def get_shape_params(self, arm_ext, wrist_yaw=0.0):
        if len(self.arm_extensions) == 0 and len(self.wrist_yaws) == 0:
            if self.is_ellipse: return float(self.cxs[0]), float(self.cys[0]), float(self.axs[0]), float(self.ays[0]), float(self.angles[0])
            elif self.is_circle: return float(self.cxs[0]), float(self.cys[0]), float(self.rs[0])
            else: return float(self.x1s[0]), float(self.y1s[0]), float(self.r1s[0]), float(self.x2s[0]), float(self.y2s[0]), float(self.r2s[0])
            
        if self.yaw_dependent and self.arm_dependent:
            if len(self.arm_extensions) == 1 or len(self.wrist_yaws) == 1:
                # Fallback to nearest/first shape for safety
                if self.is_ellipse: return float(self.cxs[0]), float(self.cys[0]), float(self.axs[0]), float(self.ays[0]), float(self.angles[0])
                elif self.is_circle: return float(self.cxs[0]), float(self.cys[0]), float(self.rs[0])
                else: return float(self.x1s[0]), float(self.y1s[0]), float(self.r1s[0]), float(self.x2s[0]), float(self.y2s[0]), float(self.r2s[0])
            
            clamped_a = np.clip(arm_ext, self.arm_extensions[0], self.arm_extensions[-1])
            clamped_y = np.clip(wrist_yaw, self.wrist_yaws[0], self.wrist_yaws[-1])
            pt = (clamped_a, clamped_y)
            
            if self.is_ellipse:
                return float(self.interp_cx(pt)), float(self.interp_cy(pt)), float(self.interp_ax(pt)), float(self.interp_ay(pt)), float(self.interp_angle(pt))
            elif self.is_circle:
                return float(self.interp_cx(pt)), float(self.interp_cy(pt)), float(self.interp_r(pt))
            else:
                return float(self.interp_x1(pt)), float(self.interp_y1(pt)), float(self.interp_r1(pt)), float(self.interp_x2(pt)), float(self.interp_y2(pt)), float(self.interp_r2(pt))
        elif self.arm_dependent or self.yaw_dependent:
            dim_arr = self.arm_extensions if self.arm_dependent else self.wrist_yaws
            val = arm_ext if self.arm_dependent else wrist_yaw
            
            if len(dim_arr) == 1:
                if self.is_ellipse: return float(self.cxs[0]), float(self.cys[0]), float(self.axs[0]), float(self.ays[0]), float(self.angles[0])
                elif self.is_circle: return float(self.cxs[0]), float(self.cys[0]), float(self.rs[0])
                else: return float(self.x1s[0]), float(self.y1s[0]), float(self.r1s[0]), float(self.x2s[0]), float(self.y2s[0]), float(self.r2s[0])
                
            clamped = np.clip(val, dim_arr[0], dim_arr[-1])
            
            if self.is_ellipse:
                return float(self.interp_cx(clamped)), float(self.interp_cy(clamped)), float(self.interp_ax(clamped)), float(self.interp_ay(clamped)), float(self.interp_angle(clamped))
            elif self.is_circle:
                return float(self.interp_cx(clamped)), float(self.interp_cy(clamped)), float(self.interp_r(clamped))
            else:
                return float(self.interp_x1(clamped)), float(self.interp_y1(clamped)), float(self.interp_r1(clamped)), float(self.interp_x2(clamped)), float(self.interp_y2(clamped)), float(self.interp_r2(clamped))
        
    def _is_in_ellipse(self, points_xy, cx, cy, ax, ay, angle_deg):
        if len(points_xy) == 0:
            return np.zeros(0, dtype=bool)
        shifted = points_xy - np.array([cx, cy])
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        rotated_x = shifted[:, 0] * cos_a - shifted[:, 1] * sin_a
        rotated_y = shifted[:, 0] * sin_a + shifted[:, 1] * cos_a
        sa_x = ax / 2.0
        sa_y = ay / 2.0
        if sa_x <= 0 or sa_y <= 0:
            return np.zeros(len(points_xy), dtype=bool)
        val = (rotated_x / sa_x)**2 + (rotated_y / sa_y)**2
        return val <= 1.0

    def _is_in_circle(self, points_xy, cx, cy, r):
        if len(points_xy) == 0:
            return np.zeros(0, dtype=bool)
        shifted = points_xy - np.array([cx, cy])
        val = shifted[:, 0]**2 + shifted[:, 1]**2
        return val <= r**2

    def _is_in_tapered_capsule(self, points_xy, x1, y1, r1, x2, y2, r2):
        if len(points_xy) == 0:
            return np.zeros(0, dtype=bool)
        ba_x, ba_y = x2 - x1, y2 - y1
        ba_sq = ba_x**2 + ba_y**2
        dr = r2 - r1
        a = ba_sq - dr**2
        
        pa_x = points_xy[:, 0] - x1
        pa_y = points_xy[:, 1] - y1
        b = (pa_x * ba_x + pa_y * ba_y) + r1 * dr
        
        h = np.zeros_like(b)
        mask = a > 1e-8
        h[mask] = b[mask] / a
        h = np.clip(h, 0.0, 1.0)
        
        pa_sq = pa_x**2 + pa_y**2
        f_h = a * (h**2) - 2 * b * h + pa_sq - r1**2
        return f_h <= 0

    def process_cloud(self, floor_points, arm_ext, wrist_yaw=0.0, min_h=MIN_POSITIVE_OBSTACLE_HEIGHT_M, max_h=MAX_POSITIVE_OBSTACLE_HEIGHT_M, max_neg_h=MAX_NEGATIVE_OBSTACLE_HEIGHT_M):
        params = self.get_shape_params(arm_ext, wrist_yaw)
        
        if self.is_ellipse:
            cx, cy, ax, ay, angle = params
            min_shape = {'type': 'ellipse', 'center': [cx, cy], 'axes': [ax, ay], 'angle': angle}
            
            inner_ax = ax + 2.0 * self.body_margin
            inner_ay = ay + 2.0 * self.body_margin
            inner_shape = {'type': 'ellipse', 'center': [cx, cy], 'axes': [inner_ax, inner_ay], 'angle': angle}
            
            outer_ax = inner_ax + 2.0 * self.stop_zone
            outer_ay = inner_ay + 2.0 * self.stop_zone
            outer_shape = {'type': 'ellipse', 'center': [cx, cy], 'axes': [outer_ax, outer_ay], 'angle': angle}
        elif self.is_circle:
            cx, cy, r = params
            min_shape = {'type': 'circle', 'center': [cx, cy], 'r': r}
            inner_shape = {'type': 'circle', 'center': [cx, cy], 'r': r + self.body_margin}
            outer_shape = {'type': 'circle', 'center': [cx, cy], 'r': r + self.body_margin + self.stop_zone}
        else:
            x1, y1, r1, x2, y2, r2 = params
            min_shape = {'type': 'tapered_capsule', 'x1': x1, 'y1': y1, 'r1': r1, 'x2': x2, 'y2': y2, 'r2': r2}
            
            i_r1 = r1 + self.body_margin
            i_r2 = r2 + self.body_margin
            inner_shape = {'type': 'tapered_capsule', 'x1': x1, 'y1': y1, 'r1': i_r1, 'x2': x2, 'y2': y2, 'r2': i_r2}
            
            o_r1 = i_r1 + self.stop_zone
            o_r2 = i_r2 + self.stop_zone
            outer_shape = {'type': 'tapered_capsule', 'x1': x1, 'y1': y1, 'r1': o_r1, 'x2': x2, 'y2': y2, 'r2': o_r2}

        # Ensure array is correctly shaped even if empty
        if len(floor_points) == 0:
            return {
                'positive_obstacle_points': np.zeros((0, 3)),
                'negative_obstacle_points': np.zeros((0, 3)),
                'body_points': np.zeros((0, 3)),
                'ring_points': {
                    'positive': np.zeros((0, 3)),
                    'negative': np.zeros((0, 3))
                },
                'outer_shape': outer_shape,
                'inner_shape': inner_shape,
                'min_shape': min_shape,
                'is_ellipse': self.is_ellipse
            }
            
        heights = floor_points[:, 2]
        pos_obs = floor_points[(heights > min_h) & (heights < max_h)]
        neg_obs = floor_points[(heights < max_neg_h)]
        
        if self.is_ellipse:
            in_outer_pos = self._is_in_ellipse(pos_obs[:, :2], cx, cy, outer_ax, outer_ay, angle)
            in_inner_pos = self._is_in_ellipse(pos_obs[:, :2], cx, cy, inner_ax, inner_ay, angle)
            in_outer_neg = self._is_in_ellipse(neg_obs[:, :2], cx, cy, outer_ax, outer_ay, angle)
            in_inner_neg = self._is_in_ellipse(neg_obs[:, :2], cx, cy, inner_ax, inner_ay, angle)
            in_inner_body = self._is_in_ellipse(pos_obs[:, :2], cx, cy, inner_ax, inner_ay, angle)
        elif self.is_circle:
            in_outer_pos = self._is_in_circle(pos_obs[:, :2], cx, cy, r + self.body_margin + self.stop_zone)
            in_inner_pos = self._is_in_circle(pos_obs[:, :2], cx, cy, r + self.body_margin)
            in_outer_neg = self._is_in_circle(neg_obs[:, :2], cx, cy, r + self.body_margin + self.stop_zone)
            in_inner_neg = self._is_in_circle(neg_obs[:, :2], cx, cy, r + self.body_margin)
            in_inner_body = self._is_in_circle(pos_obs[:, :2], cx, cy, r + self.body_margin)
        else:
            in_outer_pos = self._is_in_tapered_capsule(pos_obs[:, :2], x1, y1, o_r1, x2, y2, o_r2)
            in_inner_pos = self._is_in_tapered_capsule(pos_obs[:, :2], x1, y1, i_r1, x2, y2, i_r2)
            in_outer_neg = self._is_in_tapered_capsule(neg_obs[:, :2], x1, y1, o_r1, x2, y2, o_r2)
            in_inner_neg = self._is_in_tapered_capsule(neg_obs[:, :2], x1, y1, i_r1, x2, y2, i_r2)
            in_inner_body = self._is_in_tapered_capsule(pos_obs[:, :2], x1, y1, i_r1, x2, y2, i_r2)
            
        ring_pos = pos_obs[in_outer_pos & ~in_inner_pos]
        ring_neg = neg_obs[in_outer_neg & ~in_inner_neg]
        body_points = pos_obs[in_inner_body]

        return {
            'positive_obstacle_points': pos_obs,
            'negative_obstacle_points': neg_obs,
            'body_points': body_points,
            'ring_points': {
                'positive': ring_pos,
                'negative': ring_neg
            },
            'outer_shape': outer_shape,
            'inner_shape': inner_shape,
            'min_shape': min_shape,
            'is_ellipse': self.is_ellipse
        }
