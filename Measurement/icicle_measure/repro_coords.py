# -*- coding: utf-8 -*-
"""对比 DepthTerrainCanvas 的修复前 / 修复后坐标计算（不依赖 Qt）。

修复前的逻辑（本文件 legacy_* 函数）：
  * X/Y 用 (u-cx)*d/fx 反投影后减去端点参考，再乘 xy_display_scale；
  * Z 却用「深度簇基线压缩 + 残差放大」的显示量 display_z，
    其比例与 X/Y 的比例互不相同，且与真实 Z 差无关；
  * 端点原点直接落在 (0,0,0)，端点像素本身没有对应坐标。

修复后的逻辑（icicle_measure.geometry.build_cloud）：
  * 所有点统一反投影到相机 XYZ；
  * 统一减去同一个端点参考点（端点像素在参考深度下的反投影）；
  * XYZ 共用同一个等比系数；
  * 类别只影响颜色，不参与任何坐标计算。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icicle_measure.core import Strategies, endpoint_estimate
from icicle_measure.geometry import build_cloud
from icicle_measure.models import Config
from icicle_measure.simulator import generate


def legacy_coords(sim, detection_index=0, keypoint_index=0):
    """复刻修复前 gui.py 的几何部分。"""
    frame, detections, intrinsics = sim.frame, sim.detections, sim.intrinsics
    detection = detections[detection_index]
    endpoint = endpoint_estimate(frame, detection, keypoint_index, intrinsics,
                                  Config(), Strategies(), lambda: False)
    depth = frame.depth_m
    record = endpoint.history[-1]
    width, height = depth.shape[1], depth.shape[0]
    u, v = detection.keypoints[keypoint_index]
    radius = max(int(record.radius), 1)
    x0, x1 = max(0, int(np.floor(u - radius))), min(width, int(np.ceil(u + radius)) + 1)
    y0, y1 = max(0, int(np.floor(v - radius))), min(height, int(np.ceil(v + radius)) + 1)
    grid_depth = np.full((y1 - y0, x1 - x0), np.nan, dtype=float)
    grid_category = np.full(grid_depth.shape, -1, dtype=np.int8)
    ids = np.asarray(record.valid_ids, dtype=np.int64)
    px, py = ids % width, ids // width
    inside = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
    ids, px, py = ids[inside], px[inside], py[inside]
    if len(ids):
        grid_depth[py - y0, px - x0] = depth.ravel()[ids]
        cat = np.zeros(len(ids), dtype=np.int8)
        cat[np.isin(ids, np.asarray(record.near_ids, dtype=np.int64))] = 1
        cat[np.isin(ids, np.asarray(record.far_ids, dtype=np.int64))] = 2
        cat[np.isin(ids, np.asarray(record.target_ids, dtype=np.int64))] = 3
        grid_category[py - y0, px - x0] = cat
    gy, gx = np.where(np.isfinite(grid_depth))
    measured = grid_depth[gy, gx]
    categories = grid_category[gy, gx]
    if endpoint.depth_m is not None:
        z_origin_depth = float(endpoint.depth_m)
    elif record.center_m is not None and record.target_ids:
        z_origin_depth = float(record.center_m)
    else:
        z_origin_depth = float(np.median(measured))
    depth_groups = np.where(np.isin(categories, (1, 3)), 1, categories)
    group_medians = {int(g): float(np.median(measured[depth_groups == g]))
                     for g in np.unique(depth_groups)}
    baseline_values = np.asarray([group_medians[int(g)] for g in depth_groups])
    baseline_span = max(float(baseline_values.max() - baseline_values.min()), .001)
    residuals = measured - baseline_values
    residual_span = max(float(np.quantile(residuals, .95) - np.quantile(residuals, .05)), .001)
    baseline_scale = radius * .45 / baseline_span
    relief_scale = radius * .24 / residual_span
    display_z = ((baseline_values - z_origin_depth) * baseline_scale + residuals * relief_scale)
    pixel_x, pixel_y = gx + x0, gy + y0
    origin_x = (float(u) - intrinsics.cx) * z_origin_depth / intrinsics.fx
    origin_y = (float(v) - intrinsics.cy) * z_origin_depth / intrinsics.fy
    camera_x = (pixel_x - intrinsics.cx) * measured / intrinsics.fx - origin_x
    camera_y = (pixel_y - intrinsics.cy) * measured / intrinsics.fy - origin_y
    xy_span = max(float(np.ptp(camera_x)), float(np.ptp(camera_y)), 1e-6)
    xy_display_scale = radius / xy_span
    vertices = np.column_stack((camera_x * xy_display_scale, camera_y * xy_display_scale, display_z))
    return dict(endpoint=endpoint, record=record, vertices=vertices,
                pixel_x=pixel_x, pixel_y=pixel_y, measured=measured,
                categories=categories, u=u, v=v, radius=radius,
                z_origin_depth=z_origin_depth, intrinsics=intrinsics,
                xy_display_scale=xy_display_scale,
                baseline_scale=baseline_scale, relief_scale=relief_scale)


def fixed_coords(sim, detection_index=0, keypoint_index=0):
    """复刻修复后 gui.py 的几何部分（geometry.build_cloud）。"""
    frame, detections, intrinsics = sim.frame, sim.detections, sim.intrinsics
    detection = detections[detection_index]
    endpoint = endpoint_estimate(frame, detection, keypoint_index, intrinsics,
                                  Config(), Strategies(), lambda: False)
    record = endpoint.history[-1]
    depth = frame.depth_m
    width, height = depth.shape[1], depth.shape[0]
    u, v = detection.keypoints[keypoint_index]
    radius = max(int(record.radius), 1)
    x0, x1 = max(0, int(np.floor(u - radius))), min(width, int(np.ceil(u + radius)) + 1)
    y0, y1 = max(0, int(np.floor(v - radius))), min(height, int(np.ceil(v + radius)) + 1)
    grid_depth = np.full((y1 - y0, x1 - x0), np.nan, dtype=float)
    grid_category = np.full(grid_depth.shape, -1, dtype=np.int8)
    ids = np.asarray(record.valid_ids, dtype=np.int64)
    px, py = ids % width, ids // width
    inside = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
    ids, px, py = ids[inside], px[inside], py[inside]
    if len(ids):
        grid_depth[py - y0, px - x0] = depth.ravel()[ids]
        cat = np.zeros(len(ids), dtype=np.int8)
        cat[np.isin(ids, np.asarray(record.near_ids, dtype=np.int64))] = 1
        cat[np.isin(ids, np.asarray(record.far_ids, dtype=np.int64))] = 2
        cat[np.isin(ids, np.asarray(record.target_ids, dtype=np.int64))] = 3
        grid_category[py - y0, px - x0] = cat
    gy, gx = np.where(np.isfinite(grid_depth))
    measured = grid_depth[gy, gx]
    categories = grid_category[gy, gx]
    if endpoint.depth_m is not None:
        z_origin_depth = float(endpoint.depth_m)
    elif record.center_m is not None and record.target_ids:
        z_origin_depth = float(record.center_m)
    else:
        z_origin_depth = float(np.median(measured))
    pixel_x, pixel_y = gx + x0, gy + y0
    cloud = build_cloud(pixel_x, pixel_y, measured, intrinsics, (u, v),
                        z_origin_depth, radius)
    return dict(endpoint=endpoint, record=record, cloud=cloud,
                pixel_x=pixel_x, pixel_y=pixel_y, measured=measured,
                categories=categories, u=u, v=v, radius=radius,
                z_origin_depth=z_origin_depth, intrinsics=intrinsics)


if __name__ == '__main__':
    sim = generate('edge', seed=42)
    old = legacy_coords(sim)
    new = fixed_coords(sim)
    ep, rec = old['endpoint'], old['record']
    print('端点状态:', ep.status)
    print('端点像素 (u,v) =', (old['u'], old['v']), '| endpoint.depth_m =', ep.depth_m)
    print('原点在 Z 上取自:', 'target_candidate' if ep.depth_m is None else 'accepted',
          '=', old['z_origin_depth'])
    print('radius =', old['radius'], '| 有效像素', len(rec.valid_ids),
          '| 目标簇', len(rec.target_ids))

    d = np.hypot(old['pixel_x'] - old['u'], old['pixel_y'] - old['v'])
    idx = int(np.argmin(d))
    print('\n端点像素最近点: 像素(%d,%d) 深度=%.5f m' % (
        old['pixel_x'][idx], old['pixel_y'][idx], old['measured'][idx]))

    print('\n--- 修复前（显示坐标） ---')
    print('  顶点 = (%.4f, %.4f, %.4f)' % tuple(old['vertices'][idx]))
    print('  X/Y 比例 = %.3f /单位米 ; Z 比例 = 基线 %.3f + 残差 %.3f（与 X/Y 不一致）' % (
        old['xy_display_scale'], old['baseline_scale'], old['relief_scale']))

    rel = new['cloud']['relative_m'][idx]
    print('\n--- 修复后（米制相对端点） ---')
    print('  相对坐标 = (%.6f, %.6f, %.6f) m   |Δ|=%.6f m' % (
        rel[0], rel[1], rel[2], float(np.linalg.norm(rel))))
    print('  等比系数 = %.4f（X/Y/Z 共用）' % new['cloud']['scale'])
    print('  显示顶点 = (%.4f, %.4f, %.4f)' % tuple(new['cloud']['relative_m'][idx] * new['cloud']['scale']))
    print('  真实距离/显示距离 = %.6f m' % float(np.linalg.norm(rel)))

    # 类别不影响坐标：build_cloud 的输入里根本没有类别，改类别只改颜色。
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(new['categories'])
    cloud2 = build_cloud(new['pixel_x'], new['pixel_y'], new['measured'],
                         new['intrinsics'], (new['u'], new['v']),
                         new['z_origin_depth'], new['radius'])
    print('\n类别只着色：build_cloud 不接收类别参数，改类别后坐标是否完全一致 ->',
          bool(np.array_equal(cloud2['relative_m'], new['cloud']['relative_m'])))
    print('  类别打乱后坐标最大差异 = %.3e m' % float(np.abs(
        cloud2['relative_m'] - new['cloud']['relative_m']).max()))
    print('  类别分布:', np.bincount(new['categories']))
