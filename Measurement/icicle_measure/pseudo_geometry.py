"""Explicit scene constraints. These construct hypotheses, never measured geometry."""
import cv2
import numpy as np


def apply_background_planes(depth, valid, excluded, planes):
    """Apply user-confirmed polygon planes; inverse Z is affine in image pixels.

    Depth anchors are top-left, top-right, bottom-left of each polygon's bounds.
    Instance masks are excluded, so background edits cannot move ice surfaces.
    Later polygons take precedence, recorded by their order in the project.
    """
    result = depth.copy()
    edited = np.zeros(valid.shape, bool)
    reports = []
    for plane in planes:
        points = np.asarray(plane['polygon'], float)
        anchors = np.asarray(plane['depths_m'], float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or anchors.shape != (3,)
                or not np.isfinite(anchors).all() or (anchors <= 0).any()):
            raise ValueError('背景平面需要至少三个边界点及三个正的深度锚值')
        low, high = points.min(axis=0), points.max(axis=0)
        if (high - low < 1).any() or cv2.contourArea(points.astype(np.float32)) < 4:
            raise ValueError('背景平面区域过小或退化')
        region = np.zeros(valid.shape, np.uint8)
        cv2.fillPoly(region, [np.rint(points).astype(np.int32)], 1)
        for protected in plane.get('protected_polygons', []):
            polygon = np.asarray(protected, float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
                raise ValueError('受保护背景轮廓无效')
            cv2.fillPoly(region, [np.rint(polygon).astype(np.int32)], 0)
        region = region.astype(bool) & valid & ~excluded & (depth > 0)
        yy, xx = np.nonzero(region)
        inverse = (1 / anchors[0] + (xx - low[0]) / (high[0] - low[0]) * (1 / anchors[1] - 1 / anchors[0])
                   + (yy - low[1]) / (high[1] - low[1]) * (1 / anchors[2] - 1 / anchors[0]))
        if (inverse <= 0).any():
            raise ValueError('背景平面透视外推产生非正深度，请减小倾斜')
        result[yy, xx] = 1 / inverse
        edited |= region
        reports.append({'pixels': int(region.sum()), 'source': 'manual_polygon_inverse_depth_plane',
                        'polygon': points.tolist(), 'depths_m': anchors.tolist(),
                        'protected_polygons': plane.get('protected_polygons', [])})
    return result, edited, reports


def boundary_evidence(rgb, mask):
    """A weak ranking cue only: image edges do not prove transparent-object masks."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    scale = max(float(np.percentile(gradient, 90)), 1.)
    evidence = float(np.mean(np.minimum(gradient[boundary] / scale, 1))) if boundary.any() else 0.
    return evidence


def matches_prompts(mask, coords, labels):
    for point, label in zip(coords, labels):
        x, y = np.rint(point).astype(int)
        x, y = np.clip(x, 0, mask.shape[1]-1), np.clip(y, 0, mask.shape[0]-1)
        if bool(mask[y, x]) != bool(label):
            return False
    return True
