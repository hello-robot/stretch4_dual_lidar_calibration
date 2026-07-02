#!/usr/bin/env python3

import numpy as np

import scipy.spatial.transform

def svd_fit(points, verbose=False): 
    # calculate and subtract the mean
    center = np.mean(points, axis=0)

    if verbose:
        print( 'center =', center )

    # make the point distribution have zero mean
    points_zero_mean = points - center

    if verbose: 
        print( 'points_zero_mean[:5] =', points_zero_mean[:5] )
        print( 'points_zero_mean.shape =', points_zero_mean.shape )

    # find the covariance matrix, C, for the data
    C = np.cov(points_zero_mean.transpose())

    # find the SVD of the covariance matrix
    u, s, vh = np.linalg.svd(C)

    e0 = np.reshape(u[:, 0], (3,1))
    e1 = np.reshape(u[:, 1], (3,1))
    e2 = np.reshape(u[:, 2], (3,1))
    
    center = np.reshape(center, (3,1))
        
    return center, e0, e1, e2


def fit_plane_least_squares(points):
    """
    Perform a least squares fit of a plane to the points.
    Find the 3 element vector a for the equation
    aX ~= z where X[:,i] = [x_i, y_i, 1]^T, z[i] = z_i and a=[alpha,
    beta, gamma] such that alpha*x + beta*y + gamma ~= z .
    """
    # Formulate A matrix: [x, y, 1]
    # We solve (A^T A) x = A^T b for efficiency with large N
    
    # Points is (N, 3)
    xy = points[:, :2]
    z = points[:, 2]
    N = points.shape[0]

    # Compute A^T A and A^T b directly to avoid constructing large A
    # A = [x, y, 1]
    # A^T A = [[sum(x^2), sum(xy), sum(x)],
    #          [sum(xy), sum(y^2), sum(y)],
    #          [sum(x), sum(y), N]]
    
    sum_x = np.sum(xy[:, 0])
    sum_y = np.sum(xy[:, 1])
    sum_xx = np.dot(xy[:, 0], xy[:, 0])
    sum_yy = np.dot(xy[:, 1], xy[:, 1])
    sum_xy = np.dot(xy[:, 0], xy[:, 1])
    
    sum_z = np.sum(z)
    sum_xz = np.dot(xy[:, 0], z)
    sum_yz = np.dot(xy[:, 1], z)
    
    ATA = np.array([
        [sum_xx, sum_xy, sum_x],
        [sum_xy, sum_yy, sum_y],
        [sum_x,  sum_y,  float(N)]
    ])
    
    ATb = np.array([sum_xz, sum_yz, sum_z])
    
    try:
        a = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        A = np.c_[xy, np.ones(N)]
        a = np.linalg.lstsq(A, z, rcond=None)[0]
        
    return a

def fit_plane_error(a, points):
    """
    Calculate the fit error for the plane.
    """
    xy = points[:, :2]
    z = points[:, 2]
    z_fit = np.matmul(a, np.c_[xy, np.ones(points.shape[0])].transpose())
    fit_error = z - z_fit
    return fit_error, z_fit


def estimate_floor_height(points_world_frame):
    # Use a height histogram to estimate the height of the floor
    select_floor_height_points = None
    floor_points = points_world_frame
    floor_points_z = floor_points[:,2]
    mn = floor_points_z.min()
    mx = floor_points_z.max()
        
    # Set the allowed height range for points to be considered part of the floor in meters.
    floor_range = 0.2
    num_bins_for_height_range = 10.0
    ideal_bin_width_m = floor_range / num_bins_for_height_range

    num_bins = np.round((mx - mn) / ideal_bin_width_m).astype(np.uint64)

    if num_bins > 3:
        bin_width_m = (mx - mn) / num_bins
        floor_z_hist, bin_edges = np.histogram(floor_points_z, bins=num_bins, range=(mn, mx), density=False)
        # smooth the 1-D height histogram
        floor_z_hist = np.correlate(floor_z_hist, [0.05, 0.2, 0.5, 0.2, 0.05], 'same')
        max_index = np.argmax(floor_z_hist)
        max_bin_z = (bin_edges[max_index] + bin_edges[max_index + 1])/2.0
        select_floor_height_points = np.abs(floor_points_z - max_bin_z) < (floor_range/2.0)
        floor_z = np.median(floor_points_z[select_floor_height_points])
    else:
        print(f'WARNING: Too few histogram bins for reliable floor height estimation ({num_bins=}). Using median height across all floor points instead.')
        floor_z = np.median(floor_points_z)
    return floor_z

def estimate_floor_normal(points_world_frame):
    # Fit a plane to the floor to estimate the floor's normal vector
    fit_plane_to_floor = True
    if fit_plane_to_floor:
        floor_points = points_world_frame[:,:3]
        center, e0, e1, e2 = svd_fit(floor_points)
        debug_plane_fit = False
        if debug_plane_fit:
            print(f'{center=}')
            print(f'{e0=}')
            print(f'{e1=}')
            print(f'{e2=}')
        center = center.flatten()
        floor_surface_normal = e2.flatten()
        
    return floor_surface_normal

def fit_floor_iterative(points, verbose=True, fit_method='least_squares'):
    """
    Iteratively fit a floor plane to the points.
    
    Algorithm:
    1. Estimate floor height using histogram.
    2. Remove outliers based on height.
    3. Estimate normal using SVD.
    4. Enforce normal "up" direction.
    5. Construct approx base_footprint.
    6. Iterate until convergence.
    7. Final fit using Least Squares or SVD.
    
    Args:
        points (Nx3): Point cloud data
        verbose (bool): Print debug info
        fit_method (str): 'least_squares' or 'svd' for the final refinement step.
    
    Returns:
        transform (4x4): base_link -> base_footprint transform
        plane_params (4): [a, b, c, d] for n.x = d
    """
    
    # Copy points to avoid modifying original
    p = points[:, :3].copy()
    
    # Initial estimate
    floor_z_est = estimate_floor_height(p)
    if verbose:
        print(f"Initial floor height estimate: {floor_z_est}")

    # Iteration parameters
    max_iters = 10
    tolerance_normal = 1e-3
    tolerance_height = 1e-3
    
    prev_normal = np.array([0.0, 0.0, 1.0]) # Assume up
    prev_height = floor_z_est
    
    current_normal = prev_normal
    current_height = prev_height
    
    # Outlier threshold (tightening?)
    outlier_threshold = 0.1
    min_threshold = 0.03
    decay = 0.7
    
    final_inliers = None
    
    for i in range(max_iters):
        # 1. Filter outliers
        # dist to plane p0=[0,0,current_height], normal=current_normal
        dist = np.dot(p - np.array([0, 0, current_height]), current_normal)
        inliers_mask = np.abs(dist) < outlier_threshold
        inliers = p[inliers_mask]
        
        if len(inliers) < 100:
            print("Warning: Too few inliers found.")
            break
            
        # 2. Estimate normal on inliers
        current_normal = estimate_floor_normal(inliers)
        
        # Enforce up
        if current_normal[2] < 0:
            current_normal = -current_normal
            
        # 3. Rotate/Align logic
        d_plane = np.dot(np.mean(inliers, axis=0), current_normal)
        
        # Height update
        current_height = d_plane 
        
        # Check convergence
        norm_diff = np.linalg.norm(current_normal - prev_normal)
        height_diff = np.abs(current_height - prev_height)
        
        if verbose:
            print(f"Iter {i}: Inliers={len(inliers)}, Normal={current_normal}, Height={current_height}, Thresh={outlier_threshold:.4f}")
            
        if norm_diff < tolerance_normal and height_diff < tolerance_height and outlier_threshold <= min_threshold + 1e-4:
            if verbose:
                print("Converged.")
            final_inliers = inliers
            break
            
        prev_normal = current_normal
        prev_height = current_height
        
        # Tighten threshold
        outlier_threshold = max(min_threshold, outlier_threshold * decay)
        
    if final_inliers is None:
        final_inliers = inliers # maximal effort

    # 4. Final Fit
    if fit_method == 'least_squares':
        if verbose: print("Performing final Least Squares fit.")
        a_ls = fit_plane_least_squares(final_inliers)
        
        # Construct normal from LS result: z = ax + by + c => ax + by - z + c = 0
        # Normal = [a, b, -1]
        n_ls = np.array([a_ls[0], a_ls[1], -1.0])
        n_ls = n_ls / np.linalg.norm(n_ls)
        
        # Ensure it points up
        if n_ls[2] < 0:
            n_ls = -n_ls
            
        final_normal = n_ls
        centroid = np.mean(final_inliers, axis=0)
        d_final = np.dot(centroid, final_normal)
        
    elif fit_method == 'svd':
        if verbose: print("Performing final SVD fit.")
        center, e0, e1, e2 = svd_fit(final_inliers, verbose=False)
        
        # SVD normal is e2 (smallest eigenvector)
        final_normal = e2.flatten()
        
        # Ensure it points up
        if final_normal[2] < 0:
            final_normal = -final_normal
            
        # d = dot(center, normal)
        d_final = np.dot(center.flatten(), final_normal)
        
    else:
        raise ValueError(f"Unknown fit method: {fit_method}")
    
    # 5. Construct Final Transform (base_link -> base_footprint)
    # We want T s.t. P_footprint = T * P_baselink
    # P_footprint should have z=0 for floor inliers.
    # So the origin of footprint is on the floor.
    # The Z-axis of footprint is `final_normal`.
    # The X-axis of footprint: "same directions as the base_link with a bias toward the X-axis"
    # Projected X of base_link onto plane?
    
    # Z_f = final_normal
    # X_bl = [1, 0, 0]
    # Projection of X_bl onto plane defined by Z_f:
    # X_proj = X_bl - dot(X_bl, Z_f) * Z_f
    # X_f = X_proj / norm(X_proj)
    # Y_f = cross(Z_f, X_f)
    
    Z_f = final_normal
    X_bl = np.array([1.0, 0.0, 0.0])
    X_proj = X_bl - np.dot(X_bl, Z_f) * Z_f
    if np.linalg.norm(X_proj) < 1e-3:
        # Singularity: Normal is X-axis? Unlikely for floor.
        # Fallback to Y
        Y_bl = np.array([0.0, 1.0, 0.0])
        X_proj = Y_bl - np.dot(Y_bl, Z_f) * Z_f
        
    X_f = X_proj / np.linalg.norm(X_proj)
    Y_f = np.cross(Z_f, X_f)
    
    # Rotation Matrix (rows are axes of new frame expressed in old frame? No, columns)
    # R_bl_to_fp (Rotates vectors from footprint to base_link? or represents footprint axes in base_link?)
    # Columns of R are the basis vectors of footprint expressed in base_link.
    # R = [X_f, Y_f, Z_f]
    
    R = np.column_stack((X_f, Y_f, Z_f))
    
    # Rotation from base_link TO footprint is R.T
    
    # Translation?
    # Origin of footprint expressed in base_link:
    # P_org = d_final * Z_f (as derived before: closest point on plane to origin along normal)
    # Wait, "directly below the base_link origin using the new Z-axis"
    # Yes, that is the point on the plane intersected by the line passing through origin with direction Z_f.
    
    t = d_final * Z_f
    
    # T_base_footprint_from_base_link
    # P_fp = R.T * (P_bl - t)
    #      = R.T * P_bl - R.T * t
    
    # 4x4
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    
    # 6. Calculate RMSE
    # dist = dot(p - t, final_normal)
    final_dist = np.dot(final_inliers - t, final_normal)
    rmse = np.sqrt(np.mean(final_dist**2))
    
    return T, [final_normal[0], final_normal[1], final_normal[2], d_final], rmse

class FitPlane():
    def __init__(self):
        self.d = None
        self.n = None
        # defines the direction from points to the camera
        self.towards_camera = np.reshape(np.array([0.0, 0.0, -1.0]), (3,1))

    def set_plane(self, n, d):
        self.n = n
        self.d = d
        self.update()
        
    def update(self):
        return

    def get_plane_normal(self):
        return -self.n

    def get_plane_coordinate_system(self):
        z_p = -self.n
        # two options to avoid selecting poor choice that is almost
        # parallel to z_p
        x_approx = np.reshape(np.array([1.0, 0.0, 0.0]), (3,1))
        x_approx_1 = x_approx - (np.matmul(z_p.transpose(), x_approx) * z_p)
        x_approx = np.reshape(np.array([0.0, 1.0, 0.0]), (3,1))
        x_approx_2 = x_approx - (np.matmul(z_p.transpose(), x_approx) * z_p)
        x_approx_1_mag = np.linalg.norm(x_approx_1)
        x_approx_2_mag = np.linalg.norm(x_approx_2)
        if x_approx_1_mag > x_approx_2_mag: 
            x_p = x_approx_1 / x_approx_1_mag
        else:
            x_p = x_approx_2 / x_approx_2_mag
        y_p = np.reshape(np.cross(z_p.flatten(), x_p.flatten()), (3,1))

        p_origin = self.d * self.n
        return x_p, y_p, z_p, p_origin
        
        
    def get_points_on_plane(self, plane_origin=None, side_length=1.0, sample_spacing=0.01):
        x_p, y_p, z_p, p_origin = self.get_plane_coordinate_system()
        h = side_length/2.0
        if plane_origin is None: 
            plane_list = [np.reshape((x_p * alpha) + (y_p * beta) + p_origin, (3,))
                          for alpha in np.arange(-h, h, sample_spacing)
                          for beta in np.arange(-h, h, sample_spacing)]
        else:
            plane_origin = np.reshape(plane_origin, (3, 1))
            plane_list = [np.reshape((x_p * alpha) + (y_p * beta) + plane_origin, (3,))
                          for alpha in np.arange(-h, h, sample_spacing)
                          for beta in np.arange(-h, h, sample_spacing)]

        plane_array = np.array(plane_list)
        return plane_array

        
    def abs_dist(self, points_array):
        out = np.abs(np.matmul(self.n.transpose(), points_array.transpose()) - self.d).flatten()
        return out
        
    def height(self, points_array):
        # positive is closer to the camera (e.g., above floor)
        # negative is farther from the camera (e.g., below floor)?
        out = - (np.matmul(self.n.transpose(), points_array.transpose()) - self.d).flatten()
        return out

    def get_points_nearby(self, points_array, dist_threshold_mm):
        # return points that are within a distance from the current plane
        if (self.n is not None) and (self.d is not None): 
            dist = np.abs(np.matmul(self.n.transpose(), points_array.transpose()) - self.d).flatten()
            # only points < dist_threshold meters away from the plane are
            # considered in the fit dist_threshold = 0.2 #1.0 #0.5 #0.2

            dist_threshold_m = dist_threshold_mm / 1000.0
            thresh_test = np.abs(dist) < dist_threshold_m
            points = points_array[thresh_test, :]
        else:
            points = points_array
        return points
        
    
    def fit_svd(self, points_array,
                dist_threshold_mm=200.0,
                prefilter_points=False,
                verbose=True):
        # relevant numpy documentation for SVD:
        #
        # "When a is a 2D array, it is factorized as u @ np.diag(s) @ vh"
        #
        #" The rows of vh are the eigenvectors of A^H A and the
        # columns of u are the eigenvectors of A A^H. In both cases
        # the corresponding (possibly non-zero) eigenvalues are given
        # by s**2. "

        if prefilter_points:
            # only fit to points near the current plane
            points = self.get_points_nearby(points_array, dist_threshold_mm)
        else:
            points = points_array

        center, e0, e1, e2 = svd_fit(points, verbose)
        
        # find the smallest eigenvector, which corresponds to the
        # normal of the plane
        n = e2
        
        # ensure that the direction of the normal matches our convention
        approximate_up = self.towards_camera
        if np.matmul(n.transpose(), approximate_up) > 0.0:
            n = -n
        if verbose: 
            print( 'SVD fit' ) 
            print( 'n =', n )
            print( 'np.linalg.norm(n) =', np.linalg.norm(n) )

        #center = np.reshape(center, (3,1))
        d = np.matmul(n.transpose(), center)
        if verbose: 
            print( 'd =', d )

        self.d = d
        self.n = n
        if verbose: 
            print( 'self.d =', self.d )
            print( 'self.n =', self.n )
        self.update()
        
         
    def fit_ransac(self, points_array,
                   dist_threshold=0.2,
                   ransac_inlier_threshold_m=0.04,
                   use_density_normalization=False,
                   number_of_iterations=100,
                   prefilter_points=False,
                   verbose=True):
        # Initial RANSAC algorithm based on pseudocode on Wikipedia
        # https://en.wikipedia.org/wiki/Random_sample_consensus

        if prefilter_points:
            # only fit to points near the current plane
            dist_threshold_mm = dist_threshold * 1000.0
            points = self.get_points_nearby(points_array, dist_threshold_mm)
        else:
            points = points_array
            
        num_points = points.shape[0]
        indices = np.arange(num_points)

        ransac_threshold_m = ransac_inlier_threshold_m

        min_num_inliers = 100

        approximate_up = self.towards_camera
        
        # should be well above the maximum achievable error, since
        # error is average distance in meters

        best_model_inlier_selector = None
        best_model_inlier_count = 0
        
        for i in range(number_of_iterations):
            if verbose:
                print( 'RANSAC iteration', i )
            candidate_inliers = points[np.random.choice(indices, 3), :]
            c0, c1, c2 = candidate_inliers
            # fit plane to candidate inliers
            n = np.cross(c1 - c0, c2 - c0)
            if np.dot(n, approximate_up) > 0.0:
                n = -n
            n = np.reshape(n / np.linalg.norm(n), (3,1))
            c0 = np.reshape(c0, (3,1))
            d = np.matmul(n.transpose(), c0)

            dist = np.abs(np.matmul(n.transpose(), points.transpose()) - d).flatten()
            select_model_inliers = dist < ransac_threshold_m
            if use_density_normalization:
                inliers = points[select_model_inliers]
                # square grid with this many bins to a side, small
                # values (e.g., 10 and 20) can result in the fit being
                # biased towards edges of the planar region
                num_bins = 100 # num_bins x num_bins = total bins
                density_image, mm_per_pix, x_indices, y_indices = create_density_image(inliers, self, image_width_pix=num_bins, view_width_m=5.0, return_indices=True)
                density_image = np.reciprocal(density_image, where=density_image!=0.0)
                number_model_inliers = np.int(np.round(np.sum(density_image[y_indices, x_indices])))
            else:
                number_model_inliers = np.count_nonzero(select_model_inliers)
            if number_model_inliers > min_num_inliers:
                if verbose:
                    print( 'model found with %d inliers' % number_model_inliers )
                if number_model_inliers > best_model_inlier_count:
                    if verbose:
                        print( 'model has more inliers than the previous best model, so updating' )
                    best_model_n = n
                    best_model_d = d
                    best_model_inlier_count = number_model_inliers
                    best_model_inlier_selector = select_model_inliers
                    best_model_inliers = None
                    best_model_error = None
                elif number_model_inliers == best_model_inlier_count:
                    if verbose:
                        print( 'model has the same number of inliers as the previous best model, so comparing' )
                    model_inliers = points[select_model_inliers]
                    # error is the average distance of points from the plane
                    # sum_i | n^T p_i - d |
                    # should be able to make this faster by selecting from the already computed distances
                    new_error = np.average(np.abs(np.matmul(n.transpose(), model_inliers.transpose()) - d))
                    if best_model_inliers is None: 
                        best_model_inliers = points[best_model_inlier_selector]
                    if best_model_error is None:
                        # should be able to make this faster by
                        # selecting from the already computed
                        # distances
                        best_model_error = np.average(np.abs(np.matmul(best_model_n.transpose(), best_model_inliers.transpose()) - best_model_d))
                    if new_error < best_model_error:
                        if verbose:
                            print( 'model has a lower error than the previous model, so updating' )
                        best_model_n = n
                        best_model_d = d
                        best_model_inlier_count = number_model_inliers
                        best_model_inlier_selector = select_model_inliers
                        best_model_inliers = model_inliers
                        best_model_error = new_error
        if best_model_inlier_count > 0:
            if verbose:
                print( 'RANSAC FINISHED' ) 
                print( 'new model found by RANSAC:' )
            self.d = best_model_d
            self.n = best_model_n
            if verbose:
                print( 'self.d =', self.d )
                print( 'self.n =', self.n )
            self.update()
        else:
            print( 'RANSAC FAILED TO FIND A MODEL' )
