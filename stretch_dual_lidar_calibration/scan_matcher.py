
import small_gicp as sg
import numpy as np

# Odometry estimation based on scan-to-scan matching
class ScanMatcher:
    def __init__(self, downsampling_resolution_m, num_threads=4, max_iterations=20, max_correspondence_distance=1.0):

        self.downsampling_resolution_m = downsampling_resolution_m
        self.num_threads = num_threads
        
        # "Maximum number of iterations for the optimization algorithm."
        self.max_iterations = max_iterations

        # "Maximum distance for corresponding point pairs."
        self.max_correspondence_distance = max_correspondence_distance

        # These are apparently not supported with the pip package available as of August 13, 2025
        # small_gicp  1.0.0
        
        # # "Convergence criteria for rotation change"
        # self.rotation_epsilon_default = 0.1 * (np.pi/180.0) # 0.1 deg
        # self.rotation_epsilon = self.rotation_epsilon_default / 2.0
        
        # # "Convergence criteria for transformation change"
        # self.translation_epsilon_default = 0.001 # 1 mm
        # self.translation_epsilon = self.translation_epsilon_default / 2.0
        

    def estimate_transform(self, scan, target_scan):
        target_downsampled, target_tree = sg.preprocess_points(target_scan, self.downsampling_resolution_m, num_threads=self.num_threads)
        scan_downsampled, scan_tree = sg.preprocess_points(scan, self.downsampling_resolution_m, num_threads=self.num_threads)

        initial_transform = np.identity(4)
        result = sg.align(target_downsampled,
                          scan_downsampled,
                          target_tree,
                          initial_transform,
                          num_threads=self.num_threads,
                          max_iterations=self.max_iterations,
                          #rotation_epsilon=self.rotation_epsilon,
                          #translation_epsilon=self.translation_epsilon,
                          max_correspondence_distance=self.max_correspondence_distance)

        return result

#########################
# from running small_gicp pip package version 1.0.0 with the wrong arguments on August 13, 2025

# align(): The following argument types are supported:


# 1. (target_points: numpy.ndarray[numpy.float64[m, n]], source_points: numpy.ndarray[numpy.float64[m, n]], init_T_target_source: numpy.ndarray[numpy.float64[4, 4]] = array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), registration_type: str = 'GICP', voxel_resolution: float = 1.0, downsampling_resolution: float = 0.25, max_correspondence_distance: float = 1.0, num_threads: int = 1, max_iterations: int = 20, verbose: bool = False) -> small_gicp.RegistrationResult
    

# 2. (target: small_gicp.PointCloud, source: small_gicp.PointCloud, target_tree: small_gicp.KdTree = None, init_T_target_source: numpy.ndarray[numpy.float64[4, 4]] = array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), registration_type: str = 'GICP', max_correspondence_distance: float = 1.0, num_threads: int = 1, max_iterations: int = 20, verbose: bool = False) -> small_gicp.RegistrationResult
    

# 3. (target_voxelmap: small_gicp.GaussianVoxelMap, source: small_gicp.PointCloud, init_T_target_source: numpy.ndarray[numpy.float64[4, 4]] = array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]), max_correspondence_distance: float = 1.0, num_threads: int = 1, max_iterations: int = 20, verbose: bool = False) -> small_gicp.RegistrationResult
#########################
    

###########################################
# Comments below copied from GitHub on August 13, 2025
# https://github.com/koide3/small_gicp/blob/master/src/python/align.cpp

    
###########################################
# Align two point clouds using specified ICP-like algorithms, utilizing point cloud and KD-tree inputs.
# Input point clouds are assumed to be preprocessed (downsampled, normals estimated, KD-tree built).
# See also: :class:`voxelgrid_sampling` :class:`estimate_normals` :class:`preprocess_points`

# Parameters
# ----------
# target : :class:`PointCloud`
#     Target point cloud.
# source : :class:`PointCloud`
#     Source point cloud.
# target_tree : :class:`KdTree`, optional
#     KdTree for the target point cloud. If not given, a new KdTree is built.
# init_T_target_source : numpy.ndarray[np.float64]
#     4x4 matrix representing the initial transformation from target to source.
# registration_type : str = 'GICP'
#     Type of registration algorithm to use ('ICP', 'PLANE_ICP', 'GICP').
# max_correspondence_distance : float = 1.0
#     Maximum distance for corresponding point pairs.
# num_threads : int = 1
#     Number of threads to use for computation.
# max_iterations : int = 20
#     Maximum number of iterations for the optimization algorithm.
# rotation_epsilon: double = 0.1 * M_PI / 180.0
#     Convergence criteria for rotation change
# translation_epsilon: double = 1e-3
#     Convergence criteria for transformation change
# verbose : bool = False
#     If True, print debug information during the optimization process.

# Returns
# -------
# result : :class:`RegistrationResult`
#     Object containing the final transformation matrix and convergence status.


###########################################

# OTHER ALIGN FUNCTIONS THAT ARE AVAILABLE

###########################################

###########################################
# Align two point clouds using various ICP-like algorithms.
# This function first performs preprocessing (downsampling, normal estimation, KdTree construction) and then estimates the transformation.

# See also: :class:`voxelgrid_sampling` :class:`estimate_normals` :class:`preprocess_points`

# Parameters
# ----------
# target_points : :class:`numpy.ndarray[np.float64]`
#     Nx3 or Nx4 matrix representing the target point cloud.
# source_points : numpy.ndarray[np.float64]
#     Nx3 or Nx4 matrix representing the source point cloud.
# init_T_target_source : numpy.ndarray[np.float64]
#     4x4 matrix representing the initial transformation from target to source.
# registration_type : str = 'GICP'
#     Type of registration algorithm to use ('ICP', 'PLANE_ICP', 'GICP', 'VGICP').
# voxel_resolution : float = 1.0
#     Resolution of voxels used for correspondence search (used only in VGICP).
# downsampling_resolution : float = 0.25
#     Resolution for downsampling the point clouds.
#     Input points out of the 21bit range after discretization will be ignored (See also: :class:`voxelgrid_sampling`).
# max_correspondence_distance : float = 1.0
#     Maximum distance for matching points between point clouds.
# num_threads : int = 1
#     Number of threads to use for parallel processing.
# max_iterations : int = 20
#     Maximum number of iterations for the optimization algorithm.
# rotation_epsilon: double = 0.1 * M_PI / 180.0
#     Convergence criteria for rotation change
# translation_epsilon: double = 1e-3
#     Convergence criteria for transformation change
# verbose : bool = False
#     If True, print debug information during the optimization process.

# Returns
# -------
# result : :class:`RegistrationResult`
#     Object containing the final transformation matrix and convergence status.


###########################################
# Align two point clouds using voxel-based GICP algorithm, utilizing a Gaussian Voxel Map.
# Input source point cloud is assumed to be preprocessed (downsampled, normals estimated, KD-tree built).
# See also: :class:`voxelgrid_sampling` :class:`estimate_normals` :class:`preprocess_points`

# Parameters
# ----------
# target_voxelmap : :class:`GaussianVoxelMap`
#     Voxel map constructed from the target point cloud.
# source : :class:`PointCloud`
#     Source point cloud to align to the target.
# init_T_target_source : numpy.ndarray[np.float64]
#     4x4 matrix representing the initial transformation from target to source.
# max_correspondence_distance : float = 1.0
#     Maximum distance for corresponding point pairs.
# num_threads : int = 1
#     Number of threads to use for computation.
# max_iterations : int = 20
#     Maximum number of iterations for the optimization algorithm.
# rotation_epsilon: double = 0.1 * M_PI / 180.0
#     Convergence criteria for rotation change
# translation_epsilon: double = 1e-3
#     Convergence criteria for transformation change
# verbose : bool = False
#     If True, print debug information during the optimization process.

# Returns
# -------
# result : :class:`RegistrationResult`
#     Object containing the final transformation matrix and convergence status.



