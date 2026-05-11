#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros
import numpy as np
import os
import time
import argparse
import yaml
import stretch_body_ii.robot.robot_client as rc

from stretch_dual_lidar_calibration.dual_lidar_calibration import DualLidarCalibration
from stretch_dual_lidar_calibration.lidar_utils import LidarProcessor

# Import parameters
from stretch_dual_lidar_calibration.body_shape_calibration_params import (
    LIFT_POSITIONS,
    ARM_POSITIONS,
    WRIST_YAW_POSITIONS,
    WRIST_PITCH_POSITIONS,
    WRIST_ROLL_POSITIONS,
    SETTLE_TIME_S
)

def convert_to_native(data):
    """Recursively converts numpy types in dictionaries to native python types for clean YAML serialization."""
    if isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {k: convert_to_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native(v) for v in data]
    else:
        return data

class BodyShapeDataCollection(Node):
    def __init__(self, output_dir):
        super().__init__('ros_collect_body_shape_data')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.robot = rc.RobotClient()
        self.robot.startup()
        
        if not self.robot.is_homed():
            self.get_logger().info("Homing robot...")
            self.robot.home()
            
        self.robot.lift.set_guarded_contact_sensitivity('sensitivity_low')
        self.robot.arm.set_guarded_contact_sensitivity('sensitivity_low')
        
        self.declare_parameter('left_lidar_topic', '/lidar_points_left')
        self.declare_parameter('right_lidar_topic', '/lidar_points_right')
        self.left_topic = self.get_parameter('left_lidar_topic').value
        self.right_topic = self.get_parameter('right_lidar_topic').value

        # Load Calibration just for LidarProcessor (to unify clouds)
        self.calibration = DualLidarCalibration()
        if not self.calibration.load():
             self.get_logger().error("Calibration file not found or invalid.")
             raise SystemExit
        
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
        
        self.latest_cloud = None
        
    def cloud_callback(self, right_msg, left_msg):
        points_fp = self.lidar_processor.unify_clouds(left_msg, right_msg, self.tf_buffer)
        if points_fp is not None:
            self.latest_cloud = points_fp
            
    def get_joint_state(self):
        self.robot.pull_status()
        status = self.robot.status
        joint_state_dict = {}
        if 'lift' in status:
            joint_state_dict['joint_lift'] = status['lift']['pos']
        # stretch_body stores the actual arm extension in 'arm' 
        if 'arm' in status:
            joint_state_dict['joint_arm'] = status['arm']['pos'] 
        if 'end_of_arm' in status:
            eoa = status['end_of_arm']
            if 'wrist_yaw' in eoa:
                joint_state_dict['joint_wrist_yaw'] = eoa['wrist_yaw']['pos']
            if 'wrist_pitch' in eoa:
                joint_state_dict['joint_wrist_pitch'] = eoa['wrist_pitch']['pos']
            if 'wrist_roll' in eoa:
                joint_state_dict['joint_wrist_roll'] = eoa['wrist_roll']['pos']
        return joint_state_dict
        
    def wait_until_joint_close(self, joint_name, target, is_eoa=False, threshold=0.015):
        joint_state = self.get_joint_state()
        val = joint_state.get('joint_arm') if is_eoa else joint_state.get(joint_name, 0.0)
        
        last_print_time = time.time()
        start_time = time.time()
        while val is not None and abs(val - target) >= threshold:
            if time.time() - last_print_time > 1.0:
                self.get_logger().info(f"Waiting for {joint_name} to reach {target}. Current: {val:.4f}")
                last_print_time = time.time()
                
            if time.time() - start_time > 15.0: # Timeout
                self.get_logger().warn(f"Timeout waiting for {joint_name} to reach {target}.")
                break
                
            joint_state = self.get_joint_state()
            val = joint_state.get('joint_arm') if is_eoa else joint_state.get(joint_name, 0.0)
            time.sleep(0.05)

    def move_to_config(self, lift_pos, arm_pos, wrist_yaw, wrist_pitch, wrist_roll):
        state = self.get_joint_state()
        def is_moved(val, target):
            return val is None or abs(val - target) >= 0.015
            
        if is_moved(state.get('joint_lift'), lift_pos):
            self.robot.lift.move_to(lift_pos)
        if is_moved(state.get('joint_arm'), arm_pos):
            self.robot.arm.move_to(arm_pos)
        if is_moved(state.get('joint_wrist_yaw'), wrist_yaw):
            self.robot.end_of_arm.move_to('wrist_yaw', wrist_yaw)
        if 'joint_wrist_pitch' in state and is_moved(state.get('joint_wrist_pitch'), wrist_pitch):
            self.robot.end_of_arm.move_to('wrist_pitch', wrist_pitch)
        if 'joint_wrist_roll' in state and is_moved(state.get('joint_wrist_roll'), wrist_roll):
            self.robot.end_of_arm.move_to('wrist_roll', wrist_roll)
            
        self.robot.push_command()
        
        self.wait_until_joint_close('joint_lift', lift_pos)
        self.wait_until_joint_close('joint_arm', arm_pos, is_eoa=True)
        self.wait_until_joint_close('joint_wrist_yaw', wrist_yaw)
        if 'joint_wrist_pitch' in state:
            self.wait_until_joint_close('joint_wrist_pitch', wrist_pitch)
        if 'joint_wrist_roll' in state:
            self.wait_until_joint_close('joint_wrist_roll', wrist_roll)
            
        # Give time for the robot vibrations to stop
        time.sleep(SETTLE_TIME_S)
        
    def run_collection(self):
        tool_id = 'unknown'
        if hasattr(self.robot, 'end_of_arm') and hasattr(self.robot.end_of_arm, 'name'):
            tool_id = self.robot.end_of_arm.name
        elif hasattr(self.robot, 'status') and 'end_of_arm' in self.robot.status and 'name' in self.robot.status['end_of_arm']:
             tool_id = self.robot.status['end_of_arm']['name']
             
        metadata = {
            'collection_timestamp': time.time(),
            'robot_id': os.environ.get('HELLO_FLEET_ID', 'unknown_robot'),
            'tool_id': tool_id,
            'samples': []
        }
        
        sample_id = 0
        total_samples = len(LIFT_POSITIONS) * len(ARM_POSITIONS) * len(WRIST_YAW_POSITIONS) * len(WRIST_PITCH_POSITIONS) * len(WRIST_ROLL_POSITIONS)
        self.get_logger().info(f"Starting collection of {total_samples} samples.")
        
        for lift_pos in LIFT_POSITIONS:
            for arm_pos in ARM_POSITIONS:
                for wrist_yaw in WRIST_YAW_POSITIONS:
                    for wrist_pitch in WRIST_PITCH_POSITIONS:
                        for wrist_roll in WRIST_ROLL_POSITIONS:
                            
                            self.get_logger().info(f"Moving to config {sample_id}/{total_samples}: lift={lift_pos:.2f}, arm={arm_pos:.2f}, yaw={wrist_yaw:.2f}, pitch={wrist_pitch:.2f}, roll={wrist_roll:.2f}")
                            self.move_to_config(lift_pos, arm_pos, wrist_yaw, wrist_pitch, wrist_roll)
                            
                            # Flush out any old point cloud messages queued while the robot was moving
                            for _ in range(15):
                                rclpy.spin_once(self, timeout_sec=0.01)
                            
                            # Capture a fresh scan
                            self.latest_cloud = None
                            # Wait for a unified cloud
                            while self.latest_cloud is None and rclpy.ok():
                                rclpy.spin_once(self, timeout_sec=0.1)
                                
                            points = self.latest_cloud
                            
                            base_filename = f"sample_{sample_id:04d}"
                            npz_path = os.path.join(self.output_dir, f"{base_filename}.npz")
                            
                            # Save as compressed NumPy file
                            np.savez_compressed(npz_path, points=points)
                            
                            # Actual joined states
                            actual_state = self.get_joint_state()
                            
                            metadata['samples'].append({
                                'id': sample_id,
                                'commanded': {
                                    'lift': lift_pos,
                                    'arm': arm_pos,
                                    'wrist_yaw': wrist_yaw,
                                    'wrist_pitch': wrist_pitch,
                                    'wrist_roll': wrist_roll
                                },
                                'actual': actual_state,
                                'file': f"{base_filename}.npz"
                            })
                            self.get_logger().info(f"Saved {base_filename}.npz")
                            sample_id += 1
                            
        # Save yaml, making sure types are native
        with open(os.path.join(self.output_dir, 'metadata.yaml'), 'w') as f:
            yaml.dump(convert_to_native(metadata), f, default_flow_style=False)
            
        self.robot.stop()
        self.get_logger().info(f"Data collection complete! Saved {sample_id} samples to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Collect dual lidar point clouds over a range of arm extensions to model the clipping shape.")
    parser.add_argument('base_dir_name', type=str, nargs='?', default='collected_body_shape_data', help='Base directory name to save data (timestamp will be appended).')
    args, unknown = parser.parse_known_args() # allows skipping ros args
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"{args.base_dir_name}_{timestamp}"
    
    rclpy.init()
    try:
        node = BodyShapeDataCollection(output_dir=out_dir)
        node.run_collection()
        node.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
