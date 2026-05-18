import numpy as np
import scipy.spatial.transform
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import rclpy
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped

class LidarProcessor:
    def __init__(self, calibration):
        """
        calibration: Instance of DualLidarCalibration
        """
        self.calibration = calibration

    def get_transform_matrix(self, tf_buffer, target_frame, source_frame):
        try:
            # Look up recent transform
            t = tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time())
            return self.transform_to_matrix(t)
        except Exception as e:
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

    def msg_to_numpy(self, msg):
        field_names = [f.name for f in msg.fields]
        if 'intensity' in field_names:
            return point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
        else:
            return point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)

    @staticmethod
    def create_cloud(header, points):
        if len(points) == 0:
            return point_cloud2.create_cloud_xyz32(header, [])
        if points.shape[1] >= 4:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            return point_cloud2.create_cloud(header, fields, points)
        else:
            return point_cloud2.create_cloud_xyz32(header, points[:, :3])

    def unify_clouds(self, left_msg, right_msg, tf_buffer, left_lidar_frame='lidar_left_link', base_link_frame='base_link', base_footprint_frame='base_footprint'):
        """
        Returns unified point cloud in base_footprint frame (numpy Nx3).
        """
        # 1. Get T_bl_left (Base Link <- Left Lidar)
        source_frame = left_msg.header.frame_id if left_msg.header.frame_id else left_lidar_frame
        target_frame = base_link_frame
        
        try:
            t_stamped = tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            T_bl_left = self.transform_to_matrix(t_stamped)
        except Exception:
            # If we can't get transform, return None
            return None

        # 2. Convert to numpy
        left_points = self.msg_to_numpy(left_msg)
        right_points = self.msg_to_numpy(right_msg)
        
        # 3. Align (Right to Left) using calibration
        if self.calibration.right_to_left_transform is not None:
             right_points = self.calibration.apply(right_points)
        
        # 4. Merge (now in Left Frame)
        points_left_frame = np.vstack((left_points, right_points))
        
        # 5. Transform to Base Footprint via TF
        # We need T_fp_left (Left -> Footprint)
        # We can look this up directly from TF if the broadcaster is running!
        try:
            t_stamped = tf_buffer.lookup_transform(base_footprint_frame, source_frame, rclpy.time.Time())
            T_fp_left = self.transform_to_matrix(t_stamped)
            
            # Apply
            points_fp = self.calibration.apply(points_left_frame, transform=T_fp_left)
            return points_fp
            
        except Exception as e:
            # print(f"TF Lookup failed: {e}")
            return None
