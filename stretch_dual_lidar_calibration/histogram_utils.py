import numpy as np
import cv2
import math
import operator as op

def create_floor_histogram(points, cuboid, pix_m):
    """
    Creates a 2D histogram of points projected onto the floor plane (XY).
    
    This function is adapted from airy_helpers.py:create_floor_histogram.
    It expects points in a world frame where the Z axis is up.
    The histogram bins are computed based on the provided cuboid bounds.

    Args:
        points (np.ndarray): (N, 2) or (N, 3) array of point coordinates. Only X and Y are used.
        cuboid (dict): Dictionary defining the ROI with keys 'x_min', 'x_max', 'y_min', 'y_max'.
        pix_m (float): Resolution of each pixel in meters.

    Returns:
        tuple: (histogram, x_coord, y_coord, zero_pix, select_valid)
            - histogram (np.ndarray): The 2D histogram array.
            - x_coord (np.ndarray): The computed x pixel indices for valid points.
            - y_coord (np.ndarray): The computed y pixel indices for valid points.
            - zero_pix (np.ndarray): The pixel coordinate corresponding to world (0,0).
            - select_valid (np.ndarray): Boolean mask of which input points fell within the cuboid.
    """
    im_height_m = cuboid['x_max'] - cuboid['x_min']
    im_width_m = cuboid['y_max'] - cuboid['y_min']
    im_height = math.ceil(im_height_m / pix_m)
    im_width = math.ceil(im_width_m / pix_m)

    x = (im_height - 1) - np.floor((points[:, 0] - cuboid['x_min']) / pix_m).astype(np.int32)
    y = (im_width - 1) - np.floor((points[:, 1] - cuboid['y_min']) / pix_m).astype(np.int32)

    zero_x = (im_height - 1) - ((0.0 - cuboid['x_min']) / pix_m)
    zero_y = (im_width - 1) - ((0.0 - cuboid['y_min']) / pix_m)
    zero_pix = np.array([zero_x, zero_y])
    
    select_valid = (((x >= 0) & (x < im_height)) & ((y >= 0) & (y < im_width)))

    x_indices = x[select_valid]
    y_indices = y[select_valid]

    flat_indices = (x_indices * im_width) + y_indices    
    bin_count = np.bincount(flat_indices, minlength=im_height * im_width)
    histogram = bin_count.reshape((im_height, im_width))
    
    x_coord = x
    y_coord = y

    return histogram, x_coord, y_coord, zero_pix, select_valid

def floor_histogram_bins_to_points(valid_indices, cuboid, pix_m):
    """
    Converts floor histogram bin indices (row, col) back to world coordinates (X, Y).
    
    This performs the inverse of the mapping used in create_floor_histogram.

    Args:
        valid_indices (np.ndarray): (M, 2) array of valid bin indices. Each row is [row_idx, col_idx]
                                    corresponding to [x_image_coord, y_image_coord].
        cuboid (dict): Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'.
        pix_m (float): Pixel resolution in meters.

    Returns:
        np.ndarray: (M, 2) array of (X, Y) world coordinates.
    """
    im_height_m = cuboid['x_max'] - cuboid['x_min']
    im_width_m = cuboid['y_max'] - cuboid['y_min']
    im_height = math.ceil(im_height_m / pix_m)
    im_width = math.ceil(im_width_m / pix_m)

    x_idx = valid_indices[:, 0]
    y_idx = valid_indices[:, 1]
    
    world_x = ((im_height - 1 - x_idx) * pix_m) + cuboid['x_min']
    world_y = ((im_width - 1 - y_idx) * pix_m) + cuboid['y_min']
    
    return np.stack((world_x, world_y), axis=1).astype(np.float32)

def world_to_floor_histogram_pixel(world_point, cuboid, pix_m):
    """
    Converts a single world (X, Y) coordinate to image coordinate (row, col) for the floor histogram.
    
    Args:
        world_point (tuple or list): (X, Y) coordinates.
        cuboid (dict): Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'.
        pix_m (float): Pixel resolution in meters.
        
    Returns:
        tuple: (row, col) image coordinates as integers.
    """
    im_height_m = cuboid['x_max'] - cuboid['x_min']
    im_width_m = cuboid['y_max'] - cuboid['y_min']
    im_height = math.ceil(im_height_m / pix_m)
    im_width = math.ceil(im_width_m / pix_m)
    
    x_px = (im_height - 1) - ((world_point[0] - cuboid['x_min']) / pix_m) 
    y_px = (im_width - 1) - ((world_point[1] - cuboid['y_min']) / pix_m)
    
    return int(x_px), int(y_px)

def histogram_to_grayscale_image(histogram):
    """
    Normalizes a 2D histogram to a grayscale image (0-255).
    
    Args:
        histogram (np.ndarray): The 2D histogram array.
        
    Returns:
        np.ndarray: A 3-channel grayscale image of type uint8.
    """
    mx = np.max(histogram)
    if mx > 0.0:
        hist_img_float = (histogram.astype(np.float32) / float(mx)) * 255.0
        hist_img_gray = hist_img_float.astype(np.uint8)
    else:
        hist_img_gray = np.zeros(histogram.shape, dtype=np.uint8)
    
    return np.stack((hist_img_gray,) * 3, axis=-1)

def visualize_histogram_with_circle(image, center_px, radius_px):
    """
    Creates a copy of the image and draws a circle and its center on it.
    
    Args:
        image (np.ndarray): A 3-channel image array.
        center_px (tuple): (row, col) of the circle's center in pixel coordinates.
                           Note that cv2.circle expects (x, y) which is (col, row).
        radius_px (float or int): Radius of the circle in pixels.
        
    Returns:
        np.ndarray: A new image with the circle drawn.
    """
    vis_image = image.copy()
    
    # cv2.circle expects center as (x, y) which is (col, row)
    center_cv = (center_px[1], center_px[0])
    radius_cv = int(radius_px)
    
    # Blue circle, Red center
    cv2.circle(vis_image, center_cv, radius_cv, (255, 0, 0), 1)
    cv2.circle(vis_image, center_cv, 2, (0, 0, 255), -1)
    
    return vis_image

def connected_components(binary_image, connectivity=4, area_thresh=10):
    """
    Label connected components in a generic binary image. Adapts tracking pattern from airy_helpers.py.
    
    Args:
        binary_image (np.ndarray): 2D binary image (CV_8U)
        connectivity (int): 4 or 8 connected regions
        area_thresh (int): Minimum pixel area for a component
        
    Returns:
        tuple: (label_image, components)
            - label_image (np.ndarray): Integer map with labeled components.
            - components (list): List of component properties, skipping the background component.
    """
    output = cv2.connectedComponentsWithStats(binary_image, connectivity, cv2.CV_32S)
    num_labels = output[0]
    label_image = output[1]
    stats = output[2]
    centroids = output[3]
    components = []
    for i, c in enumerate(stats):
        area = c[cv2.CC_STAT_AREA]
        if area > area_thresh:
            s = {'label': i,
                 'left': c[cv2.CC_STAT_LEFT],
                 'top': c[cv2.CC_STAT_TOP],
                 'width': c[cv2.CC_STAT_WIDTH],
                 'height': c[cv2.CC_STAT_HEIGHT],
                 'area': area,
                 'centroid': centroids[i]}
            components.append(s)
            
    # sort by area descending and remove background component (which is the largest)
    if len(components) > 0:
        components = sorted(components, key=op.itemgetter('area'), reverse=True)[1:]
        
    return label_image, components
