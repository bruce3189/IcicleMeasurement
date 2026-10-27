# -*- coding: utf-8 -*-
"""Verification for the repaired 3D point cloud geometry.

Checks (as required):
  1. edge sample: point offsets from the origin match the deprojection result
  2. changing a point's cluster category never moves the point
  3. rotating the view never changes any 3D distance
  4. an unverified candidate endpoint stays unverified; rejection is not relaxed
"""
import os
import sys
import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icicle_measure.simulator import generate
from icicle_measure.core import endpoint_estimate, measure_frame
from icicle_measure.models import Config
from icicle_measure.core import Strategies
from icicle_measure.geometry import deproject, endpoint_reference

FAILURES = []
CHECKS = []


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print('[%s] %s%s' % (status, name, ('  -> ' + detail) if detail else ''))
    CHECKS.append(name)
    if not condition:
        FAILURES.append(name)


def main():
    from PyQt5 import QtWidgets
    from icicle_measure.gui import DepthTerrainCanvas

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    sim = generate('edge', seed=42)
    frame, detections, intrinsics = sim.frame, sim.detections, sim.intrinsics
    detection = detections[0]
    config = Config()
    endpoint = endpoint_estimate(frame, detection, 0, intrinsics, config, Strategies(), lambda: False)

    canvas = DepthTerrainCanvas()
    canvas.set_data(endpoint, frame.depth_m, detection, intrinsics)

    # ---- 4a. endpoint state is untouched ----------------------------------
    check('端点未通过验证时 status 仍为 rejected', endpoint.status == 'rejected', endpoint.status)
    check('端点 depth_m 仍为 None（未验证）', endpoint.depth_m is None)
    check('GUI 标记为 fallback 原点（未验证）', canvas.origin_depth_fallback is True)
    check('原点 Z 取自目标簇候选而非原始样本', canvas.origin_depth_source == 'target_candidate',
          '%s=%.6f' % (canvas.origin_depth_source, canvas.z_origin_depth))

    # ---- 1. offsets match deprojection -----------------------------------
    u, v = detection.keypoints[0]
    radius = canvas.radius
    origin_xyz = endpoint_reference((u, v), canvas.z_origin_depth, intrinsics)
    check('原点参考坐标与端点像素反投影一致',
          np.allclose(origin_xyz, canvas.origin_xyz_m, atol=1e-12),
          'origin=%s' % np.round(origin_xyz, 6))

    # the sample that sits on the endpoint pixel
    d = np.hypot(canvas.samples[:, 0] % frame.depth_m.shape[1] - u,
                 canvas.samples[:, 0] // frame.depth_m.shape[1] - v) if False else None
    px, py = [], []
    for pixel_id, category, depth_m in canvas.samples:
        px.append(pixel_id % frame.depth_m.shape[1])
        py.append(pixel_id // frame.depth_m.shape[1])
    px, py = np.asarray(px), np.asarray(py)
    dist_px = np.hypot(px - u, py - v)
    idx = int(np.argmin(dist_px))
    depth_at_endpoint = canvas.samples[idx][2]
    expected = deproject(px[idx], py[idx], depth_at_endpoint, intrinsics) - origin_xyz
    actual = canvas.relative_m[idx]
    check('端点像素处的相对坐标等于反投影差',
          np.allclose(expected, actual, atol=1e-12),
          'expected=%s actual=%s (m)' % (np.round(expected, 6), np.round(actual, 6)))
    check('端点像素原始深度未被强行吸附到原点',
          abs(depth_at_endpoint - canvas.z_origin_depth) > 1e-6,
          'sample=%.6f origin=%.6f' % (depth_at_endpoint, canvas.z_origin_depth))
    check('端点处 Z 偏移为毫米级而非被放大到 -1.0 级别',
          abs(actual[2]) < 0.02 and abs(actual[2]) > 1e-5,
          'Z=%+.6f m' % actual[2])
    visible = canvas.visible_mask if canvas.visible_mask is not None else np.ones(len(canvas.vertices), bool)
    check('可见点的显示坐标幅度受控（局部视图内 Z 不超过半径）',
          np.all(np.abs(canvas.vertices[visible][:, 2]) <= canvas.radius * 1.001),
          'max|Z(visible)|=%.3f radius=%d' % (float(np.max(np.abs(canvas.vertices[visible][:, 2]))), canvas.radius))
    check('端点像素处显示 Z 与真实偏移成正比（不再被单独放大）',
          abs(canvas.vertices[idx, 2] - actual[2] * canvas.cloud_scale) < 1e-9,
          'display Z=%.4f  metric=%.6f m  scale=%.3f' % (
              canvas.vertices[idx, 2], actual[2], canvas.cloud_scale))

    # isotropic scaling: display distance == metric distance * one factor
    if len(canvas.vertices) > 1:
        i, j = 0, 1
        display = float(np.linalg.norm(canvas.vertices[i] - canvas.vertices[j]))
        metric = float(np.linalg.norm(canvas.relative_m[i] - canvas.relative_m[j]))
        check('XYZ 使用同一比例缩放（显示距离 = 米制距离 × 单一系数）',
              abs(display - metric * canvas.cloud_scale) < 1e-9,
              'display=%.6f metric=%.6f scale=%.4f' % (display, metric, canvas.cloud_scale))
        check('各轴比例一致（X/Y/Z 同一系数）',
              abs((canvas.vertices[i, 0] - canvas.vertices[j, 0])
                  - (canvas.relative_m[i, 0] - canvas.relative_m[j, 0]) * canvas.cloud_scale) < 1e-9)

    # ---- 2. category does not move points --------------------------------
    vertices_before = canvas.vertices.copy()
    relative_before = canvas.relative_m.copy()
    canvas.vertex_categories = np.zeros_like(canvas.vertex_categories)
    canvas.update()
    check('改类别不改变顶点坐标', np.array_equal(canvas.vertices, vertices_before))
    check('改类别不改变米制相对坐标', np.array_equal(canvas.relative_m, relative_before))

    # ---- 3. rotation preserves 3D distances ------------------------------
    def distance_matrix():
        v = canvas.vertices
        diff = v[:, None, :] - v[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    before = distance_matrix()
    for axis in ('x', 'y', 'z'):
        canvas.rotate_axis(axis, 37)
    for eye in ((0., 0., -1.), (1., 0., 0.), (0., 1., 0.)):
        canvas.set_view_from_direction(eye)
    after = distance_matrix()
    check('旋转后点间三维距离不变', np.allclose(before, after, atol=1e-12),
          'max delta=%.3e' % float(np.max(np.abs(before - after))))
    check('旋转不改变顶点本身', np.array_equal(canvas.vertices, vertices_before))

    # perspective projection is a real camera: nearer points are magnified
    canvas.set_projection_mode('perspective')
    screen_p, depth_p = canvas._project(canvas.display_vertices(), 10.0, 200.0, 150.0)
    canvas.set_projection_mode('orthographic')
    screen_o, _ = canvas._project(canvas.display_vertices(), 10.0, 200.0, 150.0)
    check('透视与正交给出不同画面（透视已生效）', not np.allclose(screen_p, screen_o))
    origin_screen, _ = canvas._project(np.asarray([[0., 0., 0.]]), 10.0, 200.0, 150.0)
    check('透视投影把原点留在画面中心',
          np.allclose(origin_screen, [[200.0, 150.0]], atol=1e-6),
          'origin screen=%s' % np.round(origin_screen, 3))

    # exaggerated mode must be a separate, labelled display mode
    canvas.set_depth_display_mode('exaggerated')
    exaggerated = canvas.display_vertices()
    canvas.set_depth_display_mode('true')
    true_vertices = canvas.display_vertices()
    check('夸张模式只放大 Z', np.allclose(exaggerated[:, :2], true_vertices[:, :2])
          and np.allclose(exaggerated[:, 2], true_vertices[:, 2] * canvas.EXAGGERATION_FACTOR))
    check('夸张模式不改动底层米制坐标', np.array_equal(canvas.relative_m, relative_before))

    # ---- 4b. measurement rejection is not relaxed -------------------------
    result = measure_frame(frame, detections, intrinsics, config, Strategies())
    obj = result.objects[0]
    check('测距结果仍为 rejected（未放宽）', obj.status == 'rejected', obj.status)
    check('拒绝原因保留', bool(obj.reason), obj.reason)
    check('被拒端点不产生距离值', not getattr(obj, 'distance_m', None))
    check('端点元素仍标记未通过',
          all(ep.status != 'accepted' for ep in obj.endpoints),
          str([ep.status for ep in obj.endpoints]))

    print('\n%s  (%d 项检查，%d 项失败)' % (
        '全部通过' if not FAILURES else '存在失败', len(CHECKS), len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
