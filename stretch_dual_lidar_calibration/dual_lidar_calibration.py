from datetime import datetime
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import yaml

class DualLidarCalibration:
    def __init__(self, filename=None):
        if filename is None:
            fleet_path = os.environ.get('HELLO_FLEET_PATH')
            fleet_id = os.environ.get('HELLO_FLEET_ID')
            if fleet_path and fleet_id:
                self.filename = os.path.join(fleet_path, fleet_id, 'calibration_dual_lidar', 'dual_lidar_calibration.yaml')
            else:
                self.filename = os.path.expanduser('~/.stretch/calibration/dual_lidar_calibration.yaml')
        else:
            self.filename = filename
        self.right_to_left_transform = None
        self.right_to_left_transform_metadata = None
        self.base_link_to_base_footprint_transform = None
        self.base_link_to_base_footprint_transform_metadata = None
        self.floor_model_params = None
        self.floor_model_params_metadata = None
        self.robot_id = None
        self.timestamp = None

    def _robust_load(self, f):
        """
        Attempt to load YAML safely, falling back to unsafe load if needed
        to recover from numpy tags (e.g. scalar).
        """
        content = f.read()
        f.seek(0)
        try:
            return yaml.safe_load(content)
        except yaml.constructor.ConstructorError:
            print("Warning: Failed to safe_load YAML (likely numpy tags). Attempting unsafe load to recover data.")
            try:
                # Fallback for numpy objects saved directly
                return yaml.load(content, Loader=yaml.UnsafeLoader)
            except AttributeError:
                # Older pyyaml
                return yaml.load(content)

    def save(self, right_to_left_transform=None, base_link_to_base_footprint_transform=None, floor_model_params=None, body_model_params=None, robot_id=None):
        """
        Save the calibration data to a YAML file.
        Updates provided fields, keeps existing ones if not provided.
        """
        # Load existing data first to preserve what's not being updated
        current_data = {}
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    current_data = self._robust_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load existing file for merging: {e}")

        data = current_data
        timestamp = datetime.now().isoformat()
        
        def pack(val, rid, ts):
            return {'data': val, 'robot_id': rid, 'timestamp': ts}

        if right_to_left_transform is not None:
            self.right_to_left_transform = right_to_left_transform
            if hasattr(right_to_left_transform, 'tolist'):
                 val = right_to_left_transform.tolist()
            else:
                 val = right_to_left_transform
            data['right_to_left_transform'] = pack(val, robot_id, timestamp)
            
        if base_link_to_base_footprint_transform is not None:
            self.base_link_to_base_footprint_transform = base_link_to_base_footprint_transform
            if hasattr(base_link_to_base_footprint_transform, 'tolist'):
                 val = base_link_to_base_footprint_transform.tolist()
            else:
                 val = base_link_to_base_footprint_transform
            data['base_link_to_base_footprint_transform'] = pack(val, robot_id, timestamp)

        if floor_model_params is not None:
            self.floor_model_params = floor_model_params
            # Convert to readable dict for YAML
            # Internal format: [nx, ny, nz, d]
            # YAML format: {'normal': [nx, ny, nz], 'distance': d, 'description': ...}
            
            # Ensure float values
            p = []
            for val in floor_model_params:
                if hasattr(val, 'item'): p.append(val.item())
                else: p.append(float(val))
            
            val = {
                'normal': [p[0], p[1], p[2]],
                'distance': p[3],
                'description': 'Floor plane: normal [x,y,z] dot point + distance = 0'
            }
            data['floor_model_params'] = pack(val, robot_id, timestamp)
            
            data['floor_model_params'] = pack(val, robot_id, timestamp)
            
        try:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, 'w') as f:
                yaml.dump(data, f, sort_keys=False)
            return True
        except Exception as e:
            print(f"Failed to save calibration: {e}")
            return False

    def load(self):
        """Load the calibration from the YAML file."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = self._robust_load(f)
                    if not data:
                        return False
                    
                    def unpack(key):
                        val = data.get(key)
                        if val is None: return None, None
                        
                        if isinstance(val, dict) and 'data' in val:
                            return val['data'], {'robot_id': val.get('robot_id'), 'timestamp': val.get('timestamp')}
                        return val, None

                    d, m = unpack('right_to_left_transform')
                    if d is not None:
                        self.right_to_left_transform = np.array(d)
                        self.right_to_left_transform_metadata = m
                        
                    d, m = unpack('base_link_to_base_footprint_transform')
                    if d is not None:
                        self.base_link_to_base_footprint_transform = np.array(d)
                        self.base_link_to_base_footprint_transform_metadata = m
                        
                    d, m = unpack('floor_model_params')
                    if d is not None:
                        # Convert dict back to list if needed
                        if isinstance(d, dict):
                             n = d.get('normal', [0,0,1])
                             dist = d.get('distance', 0.0)
                             self.floor_model_params = [n[0], n[1], n[2], dist]
                        else:
                             self.floor_model_params = d
                        self.floor_model_params_metadata = m
                        
                        self.floor_model_params_metadata = m
                        
                    return True
            except Exception as e:
                print(f"Failed to load calibration: {e}")
                print(f"Data dump: {data}") # Debug
                return False
        return False

    def validate(self, current_robot_id=None, max_age_days=2.0, verbose=True):
        """
        Validate the calibration.
        Checks:
        1. Essential fields exist.
        2. robot_id matches (if provided) for each field.
        3. timestamp is recent (if provided) for each field.
        """
        missing = []
        if self.floor_model_params is None: missing.append('floor_model_params')
        if self.base_link_to_base_footprint_transform is None: missing.append('base_link_to_base_footprint_transform')
        if self.right_to_left_transform is None: missing.append('right_to_left_transform')
        
        if missing:
            if verbose: print(f"ERROR: Missing calibration data: {missing}")
            return False
            
        # Check metadata
        items = [
            ('floor_model_params', self.floor_model_params_metadata),
            ('base_link_to_base_footprint_transform', self.base_link_to_base_footprint_transform_metadata),
            ('right_to_left_transform', self.right_to_left_transform_metadata)
        ]
        
        valid = True
        for name, meta in items:
            if meta:
                rid = meta.get('robot_id')
                ts_str = meta.get('timestamp')
                
                if current_robot_id and rid:
                    if rid != current_robot_id:
                        if verbose: print(f"WARNING: {name} calibration for robot '{rid}' but running on '{current_robot_id}'")
                        
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        age = datetime.now() - ts
                        if age.days > max_age_days:
                            if verbose: print(f"WARNING: {name} calibration is old ({age.days} days). Limit is {max_age_days} days.")
                    except ValueError:
                        if verbose: print(f"WARNING: Invalid timestamp for {name}.")
                else:
                    if verbose: print(f"WARNING: No timestamp for {name}.")
            else:
                 if verbose: print(f"WARNING: No metadata for {name}.")
                 
        return True

    def get_transform(self):
        return self.right_to_left_transform

    def apply(self, points, transform=None):
        """
        Apply the transform to a set of points (Nx3 or Nx4).
        If transform is None, uses the loaded calibration.
        """
        if transform is None:
            transform = self.right_to_left_transform
        
        if transform is None:
            raise ValueError("No transform available to apply.")

        # Handle Nx3 or Nx4 (e.g. xyz or xyzi)
        points_xyz = points[:, :3]
        extra_channels = points[:, 3:]
        
        # Homogeneous coordinates
        ones = np.ones((points_xyz.shape[0], 1))
        points_h = np.hstack([points_xyz, ones])
        
        # Apply transform: (T @ P.T).T
        transformed_xyz = (transform @ points_h.T).T[:, :3]
        
        if extra_channels.shape[1] > 0:
            return np.hstack([transformed_xyz, extra_channels])
        else:
            return transformed_xyz

    @staticmethod
    def transform_to_matrix(transform_stamped):
        """Convert a ROS 2 TransformStamped message to a 4x4 numpy matrix."""
        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation
        
        translation = np.array([t.x, t.y, t.z])
        rotation = R.from_quat([q.x, q.y, q.z, q.w])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = translation
        return matrix

    @staticmethod
    def average_homogeneous_transforms(transforms):
        """
        Compute the average of a list of 4x4 homogeneous transformation matrices.
        Averages translation vectors and rotation matrices (using scipy).
        """
        if not transforms:
            return None
            
        translations = [t[:3, 3] for t in transforms]
        rotations = [t[:3, :3] for t in transforms]
        
        # Average rotation using scipy
        r = R.from_matrix(rotations)
        try:
            # SciPy's mean() for rotation is present in newer versions. 
            avg_rotation = r.mean()
            avg_rotation_matrix = avg_rotation.as_matrix()
        except AttributeError:
             # Fallback if .mean() is missing
             print("scipy.spatial.transform.Rotation.mean() not found. Using simple average of matrices.")
             avg_rotation_matrix = np.mean(rotations, axis=0)
             # Orthonormalize
             u, _, vt = np.linalg.svd(avg_rotation_matrix)
             avg_rotation_matrix = u @ vt

        avg_translation = np.mean(translations, axis=0)
        
        avg_transform = np.eye(4)
        avg_transform[:3, :3] = avg_rotation_matrix
        avg_transform[:3, 3] = avg_translation
        
        return avg_transform
