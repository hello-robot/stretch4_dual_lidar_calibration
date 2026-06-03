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

class FloorPlaneBroadcaster(Node):
    def __init__(self):
        super().__init__('floor_plane_broadcaster')
        
        self.declare_parameter('base_footprint_frame', 'base_footprint')
        self.base_footprint_frame = self.get_parameter('base_footprint_frame').value
        
        self.declare_parameter('floor_plane_frame', 'floor_plane')
        self.floor_plane_frame = self.get_parameter('floor_plane_frame').value
        
        self.base_footprint_to_base_link_calibration = DualLidarCalibration()
        if not self.base_footprint_to_base_link_calibration.load():
            self.get_logger().error("Could not load calibration file.")
            sys.exit(1)
            
        if self.base_footprint_to_base_link_calibration.floor_to_base_link_transform is None:
            self.get_logger().error("Calibration missing floor_to_base_link_transform.")
            sys.exit(1)
            
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_transforms()

    def broadcast_transforms(self):
        # 1. floor_to_base_link_transform (T_fp_bl)
        # P_fp = T_fp_bl * P_bl
        # We want to publish Parent=base_link, Child=base_footprint
        # Msg transform T_msg should satisfy P_bl = T_msg * P_fp
        # So T_msg = inv(T_fp_bl)
        
        T_footprint_to_base_link = self.base_footprint_to_base_link_calibration.floor_to_base_link_transform
        
        # Place holder for a TF perceived live 
        T_base_link_to_floorplane = np.linalg.inv(T_footprint_to_base_link)
        
        T_footprint_to_floorplane = T_footprint_to_base_link @ T_base_link_to_floorplane

        # Create Message
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.base_footprint_frame
        ts.child_frame_id = self.floor_plane_frame
        
        ts.transform.translation.x = T_footprint_to_floorplane[0, 3]
        ts.transform.translation.y = T_footprint_to_floorplane[1, 3]
        ts.transform.translation.z = T_footprint_to_floorplane[2, 3]
        
        quat = R.from_matrix(T_footprint_to_floorplane[:3, :3]).as_quat() # x, y, z, w
        ts.transform.rotation.x = quat[0]
        ts.transform.rotation.y = quat[1]
        ts.transform.rotation.z = quat[2]
        ts.transform.rotation.w = quat[3]

        self.broadcaster.sendTransform(ts)
        self.get_logger().info(f"Broadcasted static transform {self.base_footprint_frame} -> {self.floor_plane_frame}")

def main(args=None):
    rclpy.init(args=args)
    node = FloorPlaneBroadcaster()
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
