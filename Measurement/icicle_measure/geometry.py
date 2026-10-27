"""Pure geometry helpers for the 3D depth point cloud.

Coordinate contract (single source of truth):
  * Every depth sample is deprojected into the camera frame
    (X right, Y down, Z away from the camera) with the pinhole model.
  * The endpoint origin is one explicit camera-frame reference point
    obtained by deprojecting the endpoint pixel at the endpoint reference
    depth.  The same reference is subtracted from every sample, so points
    and origin always live in one coordinate system.
  * The resulting offsets are metres.  They are scaled by a single
    isotropic factor for display, so the rendered cloud keeps its true
    shape and all inter-point distances stay proportional to reality.
  * Cluster category only selects the colour; it never moves a point.
"""
import numpy as np


def deproject(pixel_x, pixel_y, depth_m, intrinsics):
    """Back-project pixel(s) + depth to camera-frame XYZ in metres."""
    x = (np.asarray(pixel_x, dtype=float) - intrinsics.cx) * np.asarray(depth_m, dtype=float) / intrinsics.fx
    y = (np.asarray(pixel_y, dtype=float) - intrinsics.cy) * np.asarray(depth_m, dtype=float) / intrinsics.fy
    return np.column_stack((x, y, np.asarray(depth_m, dtype=float)))


def endpoint_reference(pixel, depth_m, intrinsics):
    """The one explicit endpoint reference point, in camera-frame metres."""
    u, v = float(pixel[0]), float(pixel[1])
    return np.asarray(((u - intrinsics.cx) * depth_m / intrinsics.fx,
                        (v - intrinsics.cy) * depth_m / intrinsics.fy,
                        float(depth_m)), dtype=float)


def relative_to_origin(points_xyz, origin_xyz):
    """Offsets of the samples from the endpoint reference, in metres."""
    return np.asarray(points_xyz, dtype=float) - np.asarray(origin_xyz, dtype=float)[None, :]


def isotropic_scale(relative_m, radius):
    """One factor per axis set: the largest axis spans `radius` display units."""
    if len(relative_m) == 0:
        return 1.0
    span = max(float(np.ptp(relative_m[:, 0])),
               float(np.ptp(relative_m[:, 1])),
               float(np.ptp(relative_m[:, 2])), 1e-12)
    return float(radius) / span


def build_cloud(pixel_x, pixel_y, depth_m, intrinsics, endpoint_pixel,
                reference_depth_m, radius):
    """Return the metric cloud: camera XYZ, endpoint reference, offsets, scale.

    Returns a dict with
      points_xyz   camera-frame points (m)
      origin_xyz   endpoint reference point (m)
      relative_m   points - origin (m)
      scale        isotropic display factor (display units per metre)
    """
    points_xyz = deproject(pixel_x, pixel_y, depth_m, intrinsics)
    origin_xyz = endpoint_reference(endpoint_pixel, reference_depth_m, intrinsics)
    relative_m = relative_to_origin(points_xyz, origin_xyz)
    scale = isotropic_scale(relative_m, radius)
    return dict(points_xyz=points_xyz, origin_xyz=origin_xyz,
                relative_m=relative_m, scale=scale)


def rotation_matrix(quaternion):
    """wxyz quaternion -> 3x3 world-to-view matrix (rows: right, up, eye)."""
    w, x, y, z = quaternion
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def project(vertices, quaternion, scale, center_x, center_y, mode='perspective',
            distance=0.0):
    """Project display-space vertices to screen pixels.

    mode='orthographic'  : x = cx + X*scale, y = cy - Y*scale
    mode='perspective'   : a real pinhole camera placed `distance` display
                           units behind the view origin along the eye axis.
    Returns (screen[N,2], view_depth[N]).
    """
    rotation = rotation_matrix(quaternion)
    rotated = np.asarray(vertices, dtype=float) @ rotation.T
    if mode == 'perspective' and distance > 0:
        depth = np.maximum(distance - rotated[:, 2], 1e-3)
        focal = scale * distance
        screen = np.column_stack((center_x + focal * rotated[:, 0] / depth,
                                  center_y - focal * rotated[:, 1] / depth))
    else:
        screen = np.column_stack((center_x + rotated[:, 0] * scale,
                                  center_y - rotated[:, 1] * scale))
    return screen, rotated[:, 2]


def pairwise_distances(vertices):
    """Full pairwise distance matrix of a point set."""
    v = np.asarray(vertices, dtype=float)
    diff = v[:, None, :] - v[None, :, :]
    return np.linalg.norm(diff, axis=-1)
