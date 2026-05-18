import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np
import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
import os
import sys

from stretch_dual_lidar_calibration.dual_lidar_calibration import DualLidarCalibration

class FloorplaneBroadcaster(Node):
    def __init__(self):
        super().__init__('floorplane_broadcaster')
        
        self.declare_parameter('base_footprint_frame', 'base_footprint')
        self.base_footprint_frame = self.get_parameter('base_footprint_frame').value
    
        self.declare_parameter('floor_plane_frame', 'floor_plane')
        self.floor_plane_frame = self.get_parameter('floor_plane_frame').value
        
        self.calibration = DualLidarCalibration()
        if not self.calibration.load():
            self.get_logger().error("Could not load calibration file.")
            sys.exit(1)
            
        if self.calibration.floor_plane_to_base_footprint_transform is None:
            self.get_logger().error("Calibration missing floor_plane_to_base_footprint_transform.")
            sys.exit(1)
            
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_transforms()
        
    def broadcast_transforms(self):
        # The calibration gives T_fp_bf directly, 
        # which satisfies P_fp = T_fp_bf * P_bf
        # A static transform from Parent=floor_plane to Child=base_footprint
        # uses exactly this matrix.
        
        T_fp_bf = self.calibration.floor_plane_to_base_footprint_transform
        
        # Create Message
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.floor_plane_frame
        ts.child_frame_id = self.base_footprint_frame
        
        ts.transform.translation.x = T_fp_bf[0, 3]
        ts.transform.translation.y = T_fp_bf[1, 3]
        ts.transform.translation.z = T_fp_bf[2, 3]

        quat = R.from_matrix(T_fp_bf[:3, :3]).as_quat() # x, y, z, w
        ts.transform.rotation.x = quat[0]
        ts.transform.rotation.y = quat[1]
        ts.transform.rotation.z = quat[2]
        ts.transform.rotation.w = quat[3]
        
        self.broadcaster.sendTransform(ts)
        self.get_logger().info(f"Broadcasted static transform {self.floor_plane_frame} -> {self.base_footprint_frame}")

def main(args=None):
    rclpy.init(args=args)
    node = FloorplaneBroadcaster()
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
