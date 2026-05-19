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

class BaseFootprintBroadcaster(Node):
    def __init__(self):
        super().__init__('base_footprint_broadcaster')
        
        self.declare_parameter('base_footprint_frame', 'base_footprint')
        self.base_footprint_frame = self.get_parameter('base_footprint_frame').value
        self.declare_parameter('base_link_frame', 'base_link')
        self.base_link_frame = self.get_parameter('base_link_frame').value
        
        self.calibration = DualLidarCalibration()
        if not self.calibration.load():
            self.get_logger().error("Could not load calibration file.")
            sys.exit(1)
        if self.calibration.base_link_to_base_footprint_transform is None:
            self.get_logger().error("Calibration missing base_link_to_base_footprint_transform.")
            sys.exit(1)
            
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcast_transforms()
    def broadcast_transforms(self):
        # The calibration gives T_bf_from_bl directly, 
        # which satisfies P_bf = T_bf_from_bl * P_bl
        
        T_bf_from_bl = self.calibration.base_link_to_base_footprint_transform
        
        # To broadcast a static transform from Parent=base_link to Child=base_footprint,
        # tf2 requires the pose of base_footprint in base_link (T_bl_from_bf).
        # We compute the inverse of T_bf_from_bl mathematically.
        R_bf_from_bl = T_bf_from_bl[:3, :3]
        t_bf_from_bl = T_bf_from_bl[:3, 3]
        
        R_bl_from_bf = R_bf_from_bl.T
        t_bl_from_bf = -R_bl_from_bf @ t_bf_from_bl
        
        T_bl_from_bf = np.eye(4)
        T_bl_from_bf[:3, :3] = R_bl_from_bf
        T_bl_from_bf[:3, 3] = t_bl_from_bf
        
        # Create Message
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = self.base_link_frame
        ts.child_frame_id = self.base_footprint_frame
        
        ts.transform.translation.x = T_bl_from_bf[0, 3]
        ts.transform.translation.y = T_bl_from_bf[1, 3]
        ts.transform.translation.z = T_bl_from_bf[2, 3]

        quat = R.from_matrix(T_bl_from_bf[:3, :3]).as_quat() # x, y, z, w
        ts.transform.rotation.x = quat[0]
        ts.transform.rotation.y = quat[1]
        ts.transform.rotation.z = quat[2]
        ts.transform.rotation.w = quat[3]
        
        self.broadcaster.sendTransform(ts)
        self.get_logger().info(f"Broadcasted static transform {self.base_link_frame} -> {self.base_footprint_frame}")

def main(args=None):
    rclpy.init(args=args)
    node = BaseFootprintBroadcaster()
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
