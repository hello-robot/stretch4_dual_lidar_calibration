import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

def generate_ellipse_points(cx, cy, ax, ay, angle_deg, num_points=60):
    t = np.linspace(0, 2*np.pi, num_points)
    x = (ax/2.0) * np.cos(t)
    y = (ay/2.0) * np.sin(t)
    theta = -np.deg2rad(angle_deg)
    cos_a = np.cos(theta)
    sin_a = np.sin(theta)
    rot_x = x * cos_a - y * sin_a
    rot_y = x * sin_a + y * cos_a
    return np.vstack([rot_x + cx, rot_y + cy]).T

def generate_tapered_capsule_points(x1, y1, r1, x2, y2, r2, num_points=60):
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    diff = r1 - r2
    if d <= abs(diff) + 1e-4:
        cx, cy, r = (x1, y1, r1) if r1 >= r2 else (x2, y2, r2)
        angles = np.linspace(0, 2*np.pi, num_points)
        return np.vstack([cx + r * np.cos(angles), cy + r * np.sin(angles)]).T
        
    alpha = np.arctan2(y2 - y1, x2 - x1)
    phi = np.arccos(np.clip((r1 - r2) / d, -1.0, 1.0))
    angle1 = np.linspace(alpha + phi, alpha - phi + 2*np.pi, num_points // 2)
    pts1 = np.vstack([x1 + r1 * np.cos(angle1), y1 + r1 * np.sin(angle1)]).T
    angle2 = np.linspace(alpha - phi, alpha + phi, num_points // 2)
    pts2 = np.vstack([x2 + r2 * np.cos(angle2), y2 + r2 * np.sin(angle2)]).T
    return np.vstack([pts1, pts2, pts1[0:1]])

def generate_circle_points(cx, cy, r, num_points=60):
    angles = np.linspace(0, 2*np.pi, num_points)
    return np.vstack([cx + r * np.cos(angles), cy + r * np.sin(angles)]).T

def get_shape_pts(shape_data):
    if shape_data['type'] == 'ellipse':
        return generate_ellipse_points(shape_data['center'][0], shape_data['center'][1], shape_data['axes'][0], shape_data['axes'][1], shape_data['angle'])
    elif shape_data['type'] == 'tapered_capsule':
        return generate_tapered_capsule_points(shape_data['x1'], shape_data['y1'], shape_data['r1'], shape_data['x2'], shape_data['y2'], shape_data['r2'])
    elif shape_data['type'] == 'circle':
        return generate_circle_points(shape_data['center'][0], shape_data['center'][1], shape_data['r'])
    return None

def create_shape_marker_array(header, inner_shape, outer_shape, min_shape, ns="shapes"):
    ma = MarkerArray()
    shapes = [
        (inner_shape, 0, 1.0, 0.0, 0.0, 0.6, 0.01), # Red Inner
        (outer_shape, 1, 1.0, 0.0, 0.0, 0.6, 0.01), # Red Outer
        (min_shape,   2, 0.0, 0.0, 1.0, 1.0, 0.01)  # Blue Min
    ]
    
    for shape_data, mask_id, color_r, color_g, color_b, a, w in shapes:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = mask_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = w  # line width
        marker.color.a = a
        marker.color.r = color_r
        marker.color.g = color_g
        marker.color.b = color_b
        
        pts = get_shape_pts(shape_data)
        if pts is None:
            continue
            
        for pt in pts:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = 0.0
            marker.points.append(p)
            
        ma.markers.append(marker)
        
    # --- FILL FOR MIN SHAPE ---
    pts_min = get_shape_pts(min_shape)
    if pts_min is not None and len(pts_min) > 2:
        marker = Marker()
        marker.header = header
        marker.ns = ns + "_fill"
        marker.id = 3
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        # Ogre multiplies per-vertex colors by the root color field. We MUST reset this to opaque white.
        marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0
        
        # Providing per-vertex colors forces an unlit material (no specularity)
        c_rgba = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.08)
        
        centroid = np.mean(pts_min[:-1], axis=0) # last point is duplicated
        c_pt = Point(x=float(centroid[0]), y=float(centroid[1]), z=0.0)
        
        for i in range(len(pts_min)-1):
            pa = c_pt
            pb = Point(x=float(pts_min[i][0]), y=float(pts_min[i][1]), z=0.0)
            pc = Point(x=float(pts_min[i+1][0]), y=float(pts_min[i+1][1]), z=0.0)
            # Front Face
            marker.points.extend([pa, pb, pc])
            marker.colors.extend([c_rgba, c_rgba, c_rgba])
            # Back Face
            marker.points.extend([pa, pc, pb])
            marker.colors.extend([c_rgba, c_rgba, c_rgba])
        ma.markers.append(marker)

    # --- FILL FOR MARGIN (INNER to OUTER) ---
    pts_in = get_shape_pts(inner_shape)
    pts_out = get_shape_pts(outer_shape)
    if pts_in is not None and pts_out is not None and len(pts_in) == len(pts_out) and len(pts_in) > 2:
        marker = Marker()
        marker.header = header
        marker.ns = ns + "_fill"
        marker.id = 4
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0
        c_rgba = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.08)
        
        for i in range(len(pts_in)-1):
            p_in1 = Point(x=float(pts_in[i][0]), y=float(pts_in[i][1]), z=0.0)
            p_in2 = Point(x=float(pts_in[i+1][0]), y=float(pts_in[i+1][1]), z=0.0)
            p_out1 = Point(x=float(pts_out[i][0]), y=float(pts_out[i][1]), z=0.0)
            p_out2 = Point(x=float(pts_out[i+1][0]), y=float(pts_out[i+1][1]), z=0.0)
            
            # Triangle 1 (Front & Back)
            marker.points.extend([p_in1, p_out1, p_out2])
            marker.colors.extend([c_rgba, c_rgba, c_rgba])
            marker.points.extend([p_in1, p_out2, p_out1])
            marker.colors.extend([c_rgba, c_rgba, c_rgba])
            
            # Triangle 2 (Front & Back)
            marker.points.extend([p_in1, p_out2, p_in2])
            marker.colors.extend([c_rgba, c_rgba, c_rgba])
            marker.points.extend([p_in1, p_in2, p_out2])
            marker.colors.extend([c_rgba, c_rgba, c_rgba])
            
        ma.markers.append(marker)
        
    return ma
