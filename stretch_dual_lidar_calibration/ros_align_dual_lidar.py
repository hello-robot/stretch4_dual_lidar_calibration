import os
import sys

import numpy as np
import rclpy
import scipy.spatial.transform
import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from scipy.spatial import KDTree
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# Import local scan_matcher
import stretch_dual_lidar_calibration.scan_matcher as sm
from stretch_dual_lidar_calibration.dual_lidar_calibration import \
    DualLidarCalibration


class DualLidarCalibrator(Node):
    def __init__(self):
        super().__init__('dual_lidar_calibrator')
        
        self.state = "optimizing" # "optimizing", "finished"

        # Parameters
        self.declare_parameter('downsampling_resolution_m', 0.1)
        self.downsampling_resolution_m = self.get_parameter('downsampling_resolution_m').value
        
        self.declare_parameter('max_iterations', 100)
        self.max_iterations = self.get_parameter('max_iterations').value
        
        self.declare_parameter('max_correspondence_distance', 0.4)
        self.max_correspondence_distance = self.get_parameter('max_correspondence_distance').value
        
        self.declare_parameter('num_samples', 100)
        self.num_samples = self.get_parameter('num_samples').value
        
        self.declare_parameter('left_lidar_topic', '/lidar_points_left')
        self.left_lidar_topic = self.get_parameter('left_lidar_topic').value
        
        self.declare_parameter('right_lidar_topic', '/lidar_points_right')
        self.right_lidar_topic = self.get_parameter('right_lidar_topic').value

        self.get_logger().info(f"Parameters: res={self.downsampling_resolution_m}, iter={self.max_iterations}, dist={self.max_correspondence_distance}, samples={self.num_samples}")

        # Scan Matcher
        self.scan_matcher = sm.ScanMatcher(
            downsampling_resolution_m=self.downsampling_resolution_m,
            num_threads=4,
            max_iterations=self.max_iterations,
            max_correspondence_distance=self.max_correspondence_distance
        )

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Calibration State
        self.corrective_transforms = []
        self.rmses = []
        self.T_dom = None # Initial transform from TF
        self.T_final = None # Final calibrated transform
        
        # Subscriptions
        # Using ApproximateTimeSynchronizer as in calibrate_dual_airy_relative_pose.py logic
        # and ros_lidar_test.py
        qos = rclpy.qos.QoSProfile(depth=10)
        
        self.left_sub = Subscriber(self, PointCloud2, self.left_lidar_topic)
        self.right_sub = Subscriber(self, PointCloud2, self.right_lidar_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.right_sub, self.left_sub],
            queue_size=10,
            slop=0.05,
            allow_headerless=False
        )
        self.ts.registerCallback(self.cloud_callback)
        
        # Publishers for visualization
        self.aligned_pub = self.create_publisher(PointCloud2, 'lidar_aligned', 10)
        
        self.get_logger().info("DualLidarCalibrator initialized. Waiting for scans...")
        
        # Calibration helper
        self.calib_helper = DualLidarCalibration()

    def get_transform(self, target_frame, source_frame):
        try:
            t = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time())
            return t
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Could not transform {source_frame} to {target_frame}: {ex}')
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
        if self.state != "optimizing":
            return
            
        # We will treat LEFT as DOMINANT (target) and RIGHT as NONDOMINANT (source)
        # We want to find T that aligns RIGHT to LEFT.
        # Ideally, T_left_right (transform from right to left frame).
        
        # 1. Get initial transform from TF if not already got
        if self.T_dom is None:
            # Look up transform from right to left
            # We want points in RIGHT frame to be transformed to LEFT frame.
            # So we need transform: target=left, source=right
            tf_stamped = self.get_transform('lidar_left_link', 'lidar_right_link')
            if tf_stamped is None:
                return
            self.T_dom = self.transform_to_matrix(tf_stamped)
            self.get_logger().info("Received initial TF transform.")
            return # Wait for next callback to be clean

        # 2. Convert to numpy
        # Format: x, y, z. Ignoring intensity for alignment geometry usually, but small_gicp might use it?
        # small_gicp likely just needs xyz.
        
        left_points = self.msg_to_numpy(left_msg)
        right_points = self.msg_to_numpy(right_msg)
        
        if left_points.shape[0] < 100 or right_points.shape[0] < 100:
            self.get_logger().warn("Too few points in scan.")
            return

        # 3. Transform RIGHT scan to LEFT frame using initial TF (T_dom)
        # Points are Nx3
        # T_dom is 4x4
        # (R @ right.T).T + t
        
        R_dom = self.T_dom[:3, :3]
        t_dom = self.T_dom[:3, 3]
        
        right_points_in_left = (R_dom @ right_points.T).T + t_dom
        
        # 4. Refine using scan matching
        # estimate_transform(scan, target_scan) -> align scan TO target_scan
        # We want to align right_points_in_left TO left_points
        # The result T_corr will be such that T_corr * right_points_in_left ~= left_points
        
        try:
            result = self.scan_matcher.estimate_transform(right_points_in_left, left_points)
            T_corr = result.T_target_source
            
            # Calculate RMSE for this sample
            # Transform source points by T_corr
            R_corr = T_corr[:3, :3]
            t_corr = T_corr[:3, 3]
            aligned_points = (R_corr @ right_points_in_left.T).T + t_corr
            
            # Find nearest neighbors in target cloud for RMSE
            # We can use a simple KDTree or just rely on fitness if small_gicp provided it.
            # To be safe and independent of small_gicp version, we calculate it here.
            tree = KDTree(left_points)
            dists, _ = tree.query(aligned_points, k=1)
            rmse = np.sqrt(np.mean(dists**2))
            
            self.corrective_transforms.append(T_corr)
            self.rmses.append(rmse)
            
            self.get_logger().info(f"Sample {len(self.corrective_transforms)}/{self.num_samples} collected. RMSE: {rmse:.6f} m")
            
            if len(self.corrective_transforms) >= self.num_samples:
                self.finish_calibration()
                
        except Exception as e:
            self.get_logger().error(f"Scan matching failed: {e}")

    def finish_calibration(self):
        self.get_logger().info("Computing average transform...")
        
        T_avg = self.calib_helper.average_homogeneous_transforms(self.corrective_transforms)
        avg_rmse = np.mean(self.rmses) if self.rmses else None
        
        # Final transform T_total = T_avg * T_dom
        # Meaning: P_left = T_avg * (T_dom * P_right)
        
        self.T_final = T_avg @ self.T_dom
        
        self.get_logger().info("Calibration Complete!")
        self.get_logger().info(f"Initial TF:\n{self.T_dom}")
        self.get_logger().info(f"Corrective:\n{T_avg}")
        self.get_logger().info(f"Final Transform (Right -> Left):\n{self.T_final}")
        if avg_rmse is not None:
             self.get_logger().info(f"Average RMSE: {avg_rmse:.6f} m")
             
        # Save to YAML
        # Use the helper class to save
        robot_id = os.environ.get('HELLO_FLEET_ID', 'unknown_robot')
        self.calib_helper.save(right_to_left_transform=self.T_final, 
                               robot_id=robot_id,
                               fit_method='gicp',
                               rmse=avg_rmse)
        self.get_logger().info(f"Saved calibration to {os.path.abspath(self.calib_helper.filename)}")
        
        self.state = "finished"

    def msg_to_numpy(self, msg):
        # PointCloud2 to Nx3 numpy array
        points = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        # points is Nx3 (or Nx4 if intensity included? read_points_numpy with field_names returns structured or unstructured?)
        # read_points_numpy returns separate fields if structured? No, it usually returns a structured array if using read_points. 
        # But sensor_msgs_py specific read_points_numpy returns a standard numpy array?
        # Let's check ros_lidar_test.py usage:
        # rcloud_mat = point_cloud2.read_points_numpy(right_cloud, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
        # points = np.vstack([rcloud_mat, lcloud_mat]) -> suggests it returns (N, 4) in that case.
        
        # I asked for x,y,z so it should return (N, 3)
        return points

def main(args=None):
    rclpy.init(args=args)
    node = DualLidarCalibrator()
    
    try:
        while rclpy.ok() and node.state != "finished":
            rclpy.spin_once(node, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
