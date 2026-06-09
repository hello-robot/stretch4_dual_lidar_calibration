#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from visualization_msgs.msg import Marker, MarkerArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros
import numpy as np
import os
import yaml
import sys

from stretch_dual_lidar_calibration.dual_lidar_calibration import DualLidarCalibration
from stretch_dual_lidar_calibration.lidar_utils import LidarProcessor
from stretch_dual_lidar_calibration.body_shape_obstacle_detector import ShapeObstacleDetector
from std_msgs.msg import Header
from geometry_msgs.msg import Point

from stretch_dual_lidar_calibration.visualization_utils import create_shape_marker_array

class ShapeCalibrationVisualizerNode(Node):
    def __init__(self, provided_yaml_path=None):
        super().__init__('ros_visualize_body_shape_calibration')
        
        self.declare_parameter('left_lidar_topic', '/lidar_points_left')
        self.declare_parameter('right_lidar_topic', '/lidar_points_right')
        self.declare_parameter('floor_vertical_distance_threshold', 0.05)
        self.declare_parameter('shape_model_yaml', 'body_shape_model.yaml')
        
        self.left_topic = self.get_parameter('left_lidar_topic').value
        self.right_topic = self.get_parameter('right_lidar_topic').value
        yaml_path = provided_yaml_path or self.get_parameter('shape_model_yaml').value
        
        self.calibration = DualLidarCalibration()
        if not self.calibration.load():
            self.get_logger().warn("Standard calibration file not found. Visualizer might throw errors.")
            
        self.lidar_processor = LidarProcessor(self.calibration)
        
        self.obstacle_detector = None
        if os.path.exists(yaml_path):
             with open(yaml_path, 'r') as f:
                 data = yaml.safe_load(f)
                 # attempt new or fallback legacy keys
                 params_key = 'custom_body_shape_model_params' if 'custom_body_shape_model_params' in data else 'custom_elliptical_body_model_params'
                 if params_key in data:
                     self.obstacle_detector = ShapeObstacleDetector(data[params_key])
                     self.get_logger().info(f"Loaded shape body model from {yaml_path}")
                 else:
                     self.get_logger().error(f"Missing parameter key in {yaml_path}")
                     sys.exit(1)
        else:
             self.get_logger().error(f"Shape body model YAML '{yaml_path}' missing or not provided. Please provide a valid model YAML file path.")
             sys.exit(1)
             
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.arm_ext = 0.0
        self.wrist_yaw = 0.0
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_states_callback, 10)
        
        self.left_sub = Subscriber(self, PointCloud2, self.left_topic)
        self.right_sub = Subscriber(self, PointCloud2, self.right_topic)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.right_sub, self.left_sub],
            queue_size=10,
            slop=0.05,
            allow_headerless=False
        )
        self.ts.registerCallback(self.cloud_callback)
        
        # Publishers
        self.pub_combined = self.create_publisher(PointCloud2, 'lidar_combined', 10)
        self.pub_inliers = self.create_publisher(PointCloud2, 'lidar_floor_inliers', 10)
        self.pub_positive = self.create_publisher(PointCloud2, 'obstacles_positive', 10)
        self.pub_negative = self.create_publisher(PointCloud2, 'obstacles_negative', 10)
        self.pub_body = self.create_publisher(PointCloud2, 'body_points', 10)
        self.pub_markers = self.create_publisher(MarkerArray, 'obstacle_markers', 10)
        self.get_logger().info("Body Shape Calibration Visualizer Node started.")

    def joint_states_callback(self, msg):
        if 'arm_joint' in msg.name:
            idx = msg.name.index('arm_joint')
            self.arm_ext = msg.position[idx]
        if 'joint_wrist_yaw' in msg.name:
            idx = msg.name.index('joint_wrist_yaw')
            self.wrist_yaw = msg.position[idx]

    def cloud_callback(self, right_msg, left_msg):
        points_fp = self.lidar_processor.unify_clouds(left_msg, right_msg, self.tf_buffer)
        if points_fp is None: return
        header = self.get_header()
        
        msg_combined = LidarProcessor.create_cloud(header, points_fp)
        self.pub_combined.publish(msg_combined)
        
        z_threshold = self.get_parameter('floor_vertical_distance_threshold').value
        z_values = points_fp[:, 2]
        inlier_mask = np.abs(z_values) < z_threshold
        points_inliers = points_fp[inlier_mask]
        
        msg_inliers = LidarProcessor.create_cloud(header, points_inliers)
        self.pub_inliers.publish(msg_inliers)
        
        if self.obstacle_detector is not None:
            results = self.obstacle_detector.process_cloud(points_fp, arm_ext=self.arm_ext, wrist_yaw=self.wrist_yaw)
            self.publish_clouds(header, results)
            self.publish_markers(header, results)

    def publish_clouds(self, header, results):
        def pub_cloud(publisher, points):
            msg = LidarProcessor.create_cloud(header, points) if len(points) > 0 else LidarProcessor.create_cloud(header, [])
            publisher.publish(msg)

        pub_cloud(self.pub_positive, results['ring_points']['positive'])
        pub_cloud(self.pub_negative, results['ring_points']['negative'])
        pub_cloud(self.pub_body, results.get('body_points', []))

    def publish_markers(self, header, results):
        ma = create_shape_marker_array(
            header,
            results['inner_shape'],
            results['outer_shape'],
            results['min_shape']
        )
        self.pub_markers.publish(ma)

    def get_header(self):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = 'base_footprint'
        return h

def main(args=None):
    rclpy.init(args=args)
    
    # Try to extract the yaml path from argv directly since the user may pass it positionally
    provided_yaml = None
    if sys.argv and len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if not arg.startswith('__') and not arg.startswith('--') and (arg.endswith('.yaml') or '/' in arg):
                provided_yaml = arg
                break

    node = ShapeCalibrationVisualizerNode(provided_yaml_path=provided_yaml)
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
