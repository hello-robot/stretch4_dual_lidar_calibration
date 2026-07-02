import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import time
import threading
import tf2_ros
import scipy.spatial.transform

import os
from stretch_dual_lidar_calibration.dual_lidar_calibration import DualLidarCalibration
import stretch_dual_lidar_calibration.fit_plane as fit_plane

class FloorCalibrationNode(Node):
    def __init__(self):
        super().__init__('floor_calibration_node')
        
        # Parameters
        self.declare_parameter('left_lidar_topic', '/lidar_points_left')
        self.left_lidar_topic = self.get_parameter('left_lidar_topic').value
        
        self.declare_parameter('right_lidar_topic', '/lidar_points_right')
        self.right_lidar_topic = self.get_parameter('right_lidar_topic').value
        
        self.declare_parameter('base_link_frame', 'base_link')
        self.base_link_frame = self.get_parameter('base_link_frame').value
        
        self.declare_parameter('left_lidar_frame', 'lidar_left_link')
        self.left_lidar_frame = self.get_parameter('left_lidar_frame').value
        
        self.declare_parameter('num_samples', 30)
        self.num_samples = self.get_parameter('num_samples').value
        
        self.declare_parameter('mode', 'accumulate_points') 
        self.mode = self.get_parameter('mode').value
        
        self.declare_parameter('fit_method', 'svd')
        self.fit_method = self.get_parameter('fit_method').value
        
        self.calibration = DualLidarCalibration()
        if not self.calibration.load():
            self.get_logger().warn("Could not load existing dual lidar calibration.")
            
        self.left_sub = Subscriber(self, PointCloud2, self.left_lidar_topic)
        self.right_sub = Subscriber(self, PointCloud2, self.right_lidar_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.right_sub, self.left_sub],
            queue_size=10,
            slop=0.05,
            allow_headerless=False
        )
        self.ts.registerCallback(self.cloud_callback)
        
        # TF Buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.collected_samples = [] 
        self.collecting = True
        
        self.get_logger().info(f"Floor Calibration Node started. Mode: {self.mode}. Collection {self.num_samples} samples.")

    def get_transform(self, target_frame, source_frame):
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time())
            return t
        except tf2_ros.TransformException:
            return None

    def transform_to_matrix(self, transform_stamped):
        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation
        
        translation = np.array([t.x, t.y, t.z])
        rotation = scipy.spatial.transform.Rotation.from_quat([q.x, q.y, q.z, q.w])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = translation
        return matrix

    def cloud_callback(self, right_msg, left_msg):
        if not self.collecting:
            return
            
        # Get TF from Left Lidar to Base Link
        # We assume Left Lidar points are in 'lidar_left_link' (or whatever left_msg.header.frame_id is?)
        # Let's rely on params or header.
        source_frame = left_msg.header.frame_id if left_msg.header.frame_id else self.left_lidar_frame
        target_frame = self.base_link_frame
        
        tf_stamped = self.get_transform(target_frame, source_frame)
        if tf_stamped is None:
            self.get_logger().warn(f"Waiting for TF {source_frame} -> {target_frame}", throttle_duration_sec=2.0)
            return
            
        T_bl_left = self.transform_to_matrix(tf_stamped)

        # Convert to numpy
        left_points = self.msg_to_numpy(left_msg)
        right_points = self.msg_to_numpy(right_msg)
        
        # Align right to left if calibration exists
        if self.calibration.right_to_left_transform is not None:
             right_points = self.calibration.apply(right_points)
             
        # Merge clouds (Now in Left Lidar Frame)
        merged_points_left = np.vstack((left_points, right_points))
        
        # Transform to Base Link Frame
        # Points P_bl = T_bl_left * P_left
        merged_points_bl = self.calibration.apply(merged_points_left, transform=T_bl_left)
        
        if self.mode == 'accumulate_points':
            self.collected_samples.append(merged_points_bl)
            self.get_logger().info(f"Collected sample {len(self.collected_samples)}/{self.num_samples}")
            
            if len(self.collected_samples) >= self.num_samples:
                self.collecting = False
                self.process_accumulated_points()
                
        elif self.mode == 'average_transforms':
            # Perform fit on single frame
            self.get_logger().info(f"Processing frame {len(self.collected_samples) + 1}/{self.num_samples}...")
            try:
                # Use verbose=False to reduce spam
                T, params = fit_plane.fit_floor_iterative(merged_points_bl, verbose=False, fit_method=self.fit_method)
                self.collected_samples.append(T)
                # We might want to save params too? For averaging, averaging params (normal, d) is tricky.
                # Transform averaging handles normal (rotation) well. 'd' is in translation Z roughly.
            except Exception as e:
                self.get_logger().warn(f"Fit failed for frame: {e}")
                
            if len(self.collected_samples) >= self.num_samples:
                self.collecting = False
                self.process_mid_transforms()

    def process_accumulated_points(self):
        self.get_logger().info("Processing accumulated points...")
        # Stack all points
        all_points = np.vstack(self.collected_samples)
        
        # Downsample if too huge?
        if all_points.shape[0] > 100000:
             # Simple random choice
             indices = np.random.choice(all_points.shape[0], 100000, replace=False)
             all_points = all_points[indices]
             
        T, params, rmse = fit_plane.fit_floor_iterative(all_points, verbose=True, fit_method=self.fit_method)
        self.finish(T, params, rmse)
        
    def process_mid_transforms(self):
        self.get_logger().info("Averaging transforms...")
        T_avg = DualLidarCalibration.average_homogeneous_transforms(self.collected_samples)
        
        # Params? 
        # We can extract params from T_avg if we assume T_avg represents the floor frame.
        # T_avg is base_link -> base_footprint (or whatever fit_floor_iterative returns).
        # T definition: P_fp = T * P_bl
        # Z-axis of P_fp is [0, 0, 1]. In P_bl frame, it is R_bl_fp * [0,0,1]?
        # Wait, T has R usually defined as R_fp_to_bl? No.
        # fit_floor_iterative returns T constructed from R.T, where R cols are X_f, Y_f, Z_f.
        # So R is R_bl_fp (rotation matrix with columns as basis vectors of fp in bl).
        # So Z_f (normal) is the 3rd column of R?
        # Yes. R = [X_f, Y_f, Z_f].
        # T[:3,:3] = R.T
        # So T's rotation part is R.T.
        # To get Z_f from T:
        # R = T[:3,:3].T
        # Z_f = R[:, 2]
        
        R_matrix = T_avg[:3, :3].T
        Z_f = R_matrix[:, 2]
        
        # d?
        # t = T[:3, 3] = -R.T @ t_origin
        # t_origin (translation of origin) = - R @ t
        # d is distance along normal?
        # In fit_floor_iterative: t_vec = d * Z_f.
        # So t_origin should be roughly along Z_f.
        
        t_origin = - R_matrix @ T_avg[:3, 3]
        d = np.dot(t_origin, Z_f)
        
        params = [Z_f[0], Z_f[1], Z_f[2], d]
        
        # For averaged transforms, RMSE is harder to define without original points.
        # We'll use None or 0.0 for now if averaging.
        self.finish(T_avg, params, rmse=None)

    def finish(self, transform, params, rmse=None):
        self.get_logger().info("Floor calibration finished.")
        self.get_logger().info(f"Transform:\n{transform}")
        self.get_logger().info(f"Params: {params}")
        if rmse is not None:
             self.get_logger().info(f"RMSE: {rmse:.6f} m")
        
        robot_id = os.environ.get('HELLO_FLEET_ID', 'unknown_robot')
        if self.calibration.save(floor_to_base_link_transform=transform, 
                                 floor_model_params=params, 
                                 robot_id=robot_id,
                                 fit_method=self.fit_method,
                                 rmse=rmse):
            self.get_logger().info("Saved to dual_lidar_calibration.yaml")
        else:
            self.get_logger().error("Failed to save.")
            
        # Exit
        raise SystemExit

    def msg_to_numpy(self, msg):
        return point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)

def main(args=None):
    rclpy.init(args=args)
    node = FloorCalibrationNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
