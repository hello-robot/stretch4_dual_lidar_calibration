#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros
import numpy as np
import os
import sys

from stretch_dual_lidar_calibration.dual_lidar_calibration import DualLidarCalibration
from stretch_dual_lidar_calibration.lidar_utils import LidarProcessor
from std_msgs.msg import Header

class CalibrationVisualizerNode(Node):
    def __init__(self):
        super().__init__('ros_visualize_calibration')
        
        self.declare_parameter('left_lidar_topic', '/lidar_points_left')
        self.declare_parameter('right_lidar_topic', '/lidar_points_right')
        self.declare_parameter('floor_vertical_distance_threshold', 0.05)
        
        self.left_topic = self.get_parameter('left_lidar_topic').value
        self.right_topic = self.get_parameter('right_lidar_topic').value
        
        # Load Calibration
        self.calibration = DualLidarCalibration()
        if not self.calibration.load():
            self.get_logger().warn("Calibration file not found or invalid. Visualizer might throw errors if processing.")
            
        self.lidar_processor = LidarProcessor(self.calibration)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
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
        
        self.get_logger().info("Calibration Visualizer Node started.")

    def cloud_callback(self, right_msg, left_msg):
        # 1. Unify Point Clouds
        points_fp = self.lidar_processor.unify_clouds(left_msg, right_msg, self.tf_buffer)
        
        if points_fp is None:
            return
            
        header = self.get_header()
        
        # Publish combined cloud
        msg_combined = LidarProcessor.create_cloud(header, points_fp)
        self.pub_combined.publish(msg_combined)
        
        # 2. Floor Inliers Filter
        z_threshold = self.get_parameter('floor_vertical_distance_threshold').value
        z_values = points_fp[:, 2]
        inlier_mask = np.abs(z_values) < z_threshold
        points_inliers = points_fp[inlier_mask]
        
        msg_inliers = LidarProcessor.create_cloud(header, points_inliers)
        self.pub_inliers.publish(msg_inliers)

    def get_header(self):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = 'base_footprint'
        return h

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationVisualizerNode()
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
