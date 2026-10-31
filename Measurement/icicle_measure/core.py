"""Deterministic estimator. No Qt, camera SDK, or neural-network dependency.

Strategy call signatures are documented by the default functions below.
Pixel membership comparisons use only valid pixels shared by adjacent windows.
"""
from dataclasses import dataclass, asdict
from typing import Callable, Optional
import math
import cv2
import numpy as np
from .models import (Config, Frame, Detection, Intrinsics, WindowRecord,
                     EndpointResult, ObjectResult, MeasurementResult)
from .evidence import (line_support, anchor_evidence, subgroup_evidence, plane_evidence,
                       growth_evidence, coverage_evidence)


def deproject(pixel, depth_m, intrinsics):
    u, v = pixel
    return ((u - intrinsics.cx) * depth_m / intrinsics.fx,
            (v - intrinsics.cy) * depth_m / intrinsics.fy, depth_m)


def otsu_split(depths, config):
    """Return 0=discard, 1=near, 2=far labels for a vector of valid metres.

    Exact 1D between-class variance maximization over sorted sample boundaries,
    rather than histogram quantization. A split alone is never an acceptance.
    """
    labels = np.zeros(len(depths), np.uint8)
    if len(depths) < 2:
        return labels
    order = np.argsort(depths)
    z = depths[order].astype(float)
    boundaries = np.flatnonzero(np.diff(z) > 1e-12)
    if not len(boundaries):
        return labels
    sums = np.cumsum(z)
    n1 = boundaries + 1
    n2 = len(z) - n1
    mu1 = sums[boundaries] / n1
    mu2 = (sums[-1] - sums[boundaries]) / n2
    cut = boundaries[np.argmax(n1 * n2 * (mu1 - mu2) ** 2)]
    labels[order[:cut + 1]] = 1
    labels[order[cut + 1:]] = 2
    for k in (1, 2):
        indices = np.flatnonzero(labels == k)
        sample = depths[indices]
        median = np.median(sample)
        sigma = max(1.4826 * np.median(np.abs(sample - median)), config.mad_floor_m)
        labels[indices[np.abs(sample - median) > config.mad_multiplier * sigma]] = 0
    return labels


def select_target(ids, depths, labels, shape, detection, config, endpoint_pixel=None):
    """Choose a spatially supported near component; return mask and reason.

    With an endpoint, only components supported by its local anchor may win.
    Anchor radius/count are ranked before total support. Six-argument callers
    remain compatible; endpoint_estimate also audits their selected component.
    """
    h, w = shape
    near = labels == 1
    if not near.any():
        return np.zeros(len(ids), bool), '没有近簇样本'
    ys, xs = np.divmod(ids, w)
    ox, oy = int(xs.min()), int(ys.min())
    local_w, local_h = int(xs.max()) - ox + 1, int(ys.max()) - oy + 1
    local_ids = (ys - oy) * local_w + xs - ox
    mask = np.zeros((local_h, local_w), np.uint8)
    mask.flat[local_ids[near]] = 1
    count, components = cv2.connectedComponents(mask, connectivity=8)
    support = line_support(ids, w, detection, config) & near
    component_ids = components.flat[local_ids]
    support_mask = np.zeros_like(mask)
    support_mask.flat[local_ids[support]] = 1
    support_n, support_components, stats, _ = cv2.connectedComponentsWithStats(support_mask, connectivity=8)
    supported_components = set()
    for support_id in range(1, support_n):
        if stats[support_id, cv2.CC_STAT_AREA] >= config.component_min:
            support_pixel = np.flatnonzero(support_components.ravel() == support_id)[0]
            supported_components.add(int(components.ravel()[support_pixel]))
    candidates = []
    for cid in range(1, count):
        member = component_ids == cid
        size = int(np.count_nonzero(member & support))
        if cid in supported_components:
            anchor = (anchor_evidence(ids[member], w, endpoint_pixel, detection, config)
                      if endpoint_pixel is not None else {'passed': True, 'count': size, 'radius_px': 0})
            if anchor['passed']:
                candidates.append((size, float(np.mean(depths[member])), member, anchor))
    if not candidates:
        reason = ('端点锚区缺少可关联的目标空间支持' if supported_components and endpoint_pixel is not None
                  else '近簇在框内连线附近缺少连续空间支持')
        return np.zeros(len(ids), bool), reason
    candidates.sort(key=lambda item: (item[3]['radius_px'], -item[3]['count'], -item[0]))
    best = candidates[0]
    for other in candidates[1:]:
        if (other[3]['count'] >= best[3]['count'] * config.ambiguity_size_ratio
                and abs(other[1] - best[1]) >= config.ambiguity_depth_m):
            return np.zeros(len(ids), bool), '目标归属不明确：多个空间支持相当且深度不同的近簇分量'
    return best[2], ''


def confidence_scores(separation, min_count, delta, tolerance, jaccard, config):
    """Return named scores. The caller takes min, not a weighted average."""
    return {
        'separation': min(1.0, separation / config.separation_full),
        'samples': min(1.0, min_count / config.full_samples),
        'center': math.exp(-0.5 * (delta / tolerance) ** 2) if delta is not None else 0.0,
        'membership': min(1.0, jaccard / config.membership_full) if jaccard is not None else 0.0,
    }


def arithmetic_depth(depths):
    return float(np.mean(depths))


@dataclass
class Strategies:
    clusterer: Callable = otsu_split
    selector: Callable = select_target
    scorer: Callable = confidence_scores
    aggregator: Callable = arithmetic_depth


def shared_membership(previous, current):
    common = set(previous.valid_ids) & set(current.valid_ids)
    old = set(previous.target_ids) & common
    new = set(current.target_ids) & common
    union = old | new
    return len(old & new) / len(union) if union else 0.0


def endpoint_estimate(frame, detection, index, intrinsics, config, strategies, cancelled):
    if config.endpoint_depth_mode == 'near_cluster_simple':
        from .near_cluster import endpoint_estimate as near_estimate
        return near_estimate(frame, detection, index, intrinsics, config, strategies, cancelled)
    point = detection.keypoints[index]
    result = EndpointResult(('tip', 'base')[index], tuple(point))
    h, w = frame.depth_m.shape
    previous = None
    streak = 0
    previous_bounds = None
    reached_box_limit = False
    bx1, by1, bx2, by2 = detection.bbox
    short_side = max(1.0, min(float(bx2 - bx1), float(by2 - by1)))
    # The square search window has 2r+1 pixels per side. Cap r so its
    # widest side never exceeds the detector box's shorter side.
    max_radius = max(0, math.floor((short_side - 1.0) / 2.0))
    for radius in config.radii:
        if cancelled():
            result.status, result.reason = 'cancelled', '用户取消'
            return result
        radius = min(radius, max_radius)
        u, v = point
        pad = config.bbox_padding
        bounds = (max(0, math.ceil(bx1 - pad), math.floor(u - radius)),
                  max(0, math.ceil(by1 - pad), math.floor(v - radius)),
                  min(w, math.floor(bx2 + pad) + 1, math.floor(u + radius) + 1),
                  min(h, math.floor(by2 + pad) + 1, math.floor(v + radius) + 1))
        # A clipped window that no longer grows must not manufacture stability.
        if bounds == previous_bounds:
            if radius >= max_radius:
                reached_box_limit = True
                break
            continue
        previous_bounds = bounds
        record = WindowRecord(radius, bounds)
        result.history.append(record)
        x1, y1, x2, y2 = bounds
        if x1 >= x2 or y1 >= y2:
            record.reasons = ['搜索窗口与目标区域无交集']
            previous, streak = None, 0
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2]
        z = frame.depth_m[y1:y2, x1:x2].astype(float).ravel()
        valid = np.isfinite(z) & (z >= frame.min_depth_m) & (z <= frame.max_depth_m) & (z > 0)
        ids = (yy * w + xx).ravel()[valid]
        z = z[valid]
        record.valid_ids = ids.tolist()
        labels = strategies.clusterer(z, config)
        near, far = labels == 1, labels == 2
        record.near_ids, record.far_ids = ids[near].tolist(), ids[far].tolist()
        record.near_count, record.far_count = int(near.sum()), int(far.sum())
        if not near.any() or not far.any():
            record.reasons = ['有效深度不足或无法形成两个深度簇']
            previous, streak = None, 0
            continue
        zn, zf = z[near], z[far]
        separation = float(abs(zf.mean() - zn.mean()) / max(np.sqrt(zn.var() + zf.var()), 1e-9))
        gap = float(np.percentile(zf, 10) - np.percentile(zn, 90))
        record.separation, record.gap_m = separation, gap
        if strategies.selector is select_target:
            target, spatial_reason = select_target(ids, z, labels, (h, w), detection, config, endpoint_pixel=point)
        else:
            # Six-argument replacement selectors stay compatible. Evidence
            # gates below still apply to whichever samples they return.
            target, spatial_reason = strategies.selector(ids, z, labels, (h, w), detection, config)
        record.target_ids = ids[target].tolist()
        record.target_count = int(target.sum())
        if target.any():
            record.center_m = strategies.aggregator(z[target])
        record.anchor = anchor_evidence(ids[target], w, point, detection, config)
        record.subgroups = subgroup_evidence(z[target], config, otsu_split)
        record.plane = plane_evidence(ids[target], z[target], w, point, record.center_m, config)
        record.growth = growth_evidence(previous, ids, target, far, z, config)
        record.coverage = coverage_evidence(ids, w, near, far, target, config)
        delta = abs(record.center_m - previous.center_m) if (
            previous is not None and previous.center_m is not None and record.center_m is not None) else None
        membership = shared_membership(previous, record) if previous is not None else None
        tolerance = max(config.center_abs_m, config.center_rel * (record.center_m or 0))
        record.center_change_m, record.membership = delta, membership
        min_count = min(record.near_count, record.far_count, record.target_count)
        record.scores = strategies.scorer(separation, min_count, delta, tolerance, membership, config)
        record.score = float(min(record.scores.values()))
        record.checks = {
            'separation': separation >= config.separation_min,
            'gap': gap >= config.gap_min_m,
            'samples': min_count >= config.min_samples,
            'spatial': not spatial_reason and record.target_count >= config.component_min,
            'center': delta is not None and delta <= tolerance,
            'membership': membership is not None and membership >= config.membership_min,
            'confidence': record.score >= config.confidence_min,
            'anchor': record.anchor['passed'],
            'subgroups': bool(target.any() and not record.subgroups['ambiguous']),
            'representative': record.plane['representative'],
            'growth': record.growth['size_passed'],
            'ring': record.growth['consistency_passed'],
            'coverage': record.coverage['passed'],
        }
        if config.endpoint_depth_mode == 'near_cluster_mean':
            # A neighbourhood mean is deliberately an approximation. Local
            # plane extrapolation and ring differences remain diagnostics,
            # not extra rejection criteria for that requested measurement.
            for key in ('representative', 'ring'):
                record.advisories[key] = record.checks.pop(key)
            if not record.plane['representative']:
                record.warnings.append('均值代表性提示：' + record.plane['reason'])
            if previous is not None and not record.growth['consistency_passed']:
                record.warnings.append('新增区域提示：与保留目标存在深度差或缺少比较样本')
        reasons = {'separation': '两簇分离度不足', 'gap': '两簇分位深度间隔不足',
                   'samples': '去噪后簇样本量不足', 'spatial': spatial_reason or '缺少目标空间支持',
                   'center': '中心深度不稳定或尚无前一窗口',
                   'membership': '共有像素成员不稳定或尚无前一窗口', 'confidence': '综合评分未达阈值',
                   'anchor': '端点锚区缺少目标支持或目标无法关联',
                   'subgroups': ('目标内部存在多个可信深度群，不能合并取均值' if record.subgroups['ambiguous'] else '缺少可检查的目标样本'),
                   'representative': record.plane['reason'],
                   'growth': '新增目标或远簇样本不足或尚无前一窗口',
                   'ring': '新增区域与已有目标深度不一致或缺少可比较样本',
                   'coverage': '目标或远近簇空间覆盖不足'}
        record.reasons = [reasons[key] for key, passed in record.checks.items() if not passed]
        streak = streak + 1 if all(record.checks.values()) else 0
        record.consecutive_passes = streak
        if streak >= config.consecutive:
            result.status = 'accepted'
            result.reason = ('通过扩窗证据判定（近簇均值近似）' if config.endpoint_depth_mode == 'near_cluster_mean'
                             else '通过扩窗证据判定（严格局部一致）')
            result.depth_m = record.center_m
            result.xyz_m = deproject(point, result.depth_m, intrinsics)
            result.confidence = record.score
            return result
        if all(record.checks.values()):
            record.reasons.append('本窗未通过，等待下一窗口比较')
        if radius >= max_radius:
            reached_box_limit = True
            record.reasons.append('已达到检测框短边搜索上限')
            break
        previous = record
    result.confidence = result.history[-1].score if result.history else 0.0
    last_reasons = result.history[-1].reasons if result.history else ['无有效窗口']
    prefix = '检测框短边已到搜索上限仍未通过' if reached_box_limit else '达到搜索边界仍未通过'
    result.reason = prefix + '：' + '；'.join(last_reasons)
    return result


def detection_problem(detection, shape):
    h, w = shape
    if len(detection.bbox) != 4 or not np.all(np.isfinite(detection.bbox)):
        return '检测框坐标无效'
    x1, y1, x2, y2 = detection.bbox
    if x2 <= x1 or y2 <= y1 or x2 < 0 or y2 < 0 or x1 >= w or y1 >= h:
        return '检测框尺寸无效或完全超出图像'
    if len(detection.keypoints) != 2 or any(p is None for p in detection.keypoints):
        return '缺少完整两个关键点'
    for p in detection.keypoints:
        if len(p) != 2 or not np.all(np.isfinite(p)) or not (0 <= p[0] < w and 0 <= p[1] < h):
            return '关键点无效或超出图像'
    if np.linalg.norm(np.asarray(detection.keypoints[0]) - detection.keypoints[1]) < 1e-6:
        return '两个关键点重合'
    return ''


def measure_frame(frame: Frame, detections, intrinsics: Intrinsics, config: Optional[Config] = None,
                  strategies: Optional[Strategies] = None, cancel: Optional[Callable] = None,
                  progress: Optional[Callable] = None):
    """Measure all boxes. cancel()->bool; progress(done,total) are optional.

    A cancelled frame is explicitly flagged and includes no distance for an
    interrupted object. Already completed objects remain available to callers.
    """
    config, strategies = config or Config(), strategies or Strategies()
    cancelled = cancel or (lambda: False)
    frame.validate()
    intrinsics.validate()
    config.validate()
    detections = list(detections)
    if len({d.id for d in detections}) != len(detections):
        raise ValueError('目标 ID 必须唯一')
    result = MeasurementResult([], config.active_dict(), asdict(intrinsics), frame.rgb.shape[1::-1],
                               source=dict(frame.metadata))
    if frame.metadata.get('is_synthetic') is True:
        result.assumptions = ('模型伪深度 / 人工合成，光轴 Z；尺度为人工设定，针孔几何是假设，原照片畸变未经验证。'
                              '距离为基于伪深度的估计；邻域目标均值赋给原关键点射线，结果不验证真实精度。')
    for i, detection in enumerate(detections):
        if cancelled():
            result.cancelled = True
            break
        problem = detection_problem(detection, frame.depth_m.shape)
        if problem:
            obj = ObjectResult(detection.id, 'skipped', problem)
        else:
            endpoints = []
            for index in range(2):
                ep = endpoint_estimate(frame, detection, index, intrinsics, config, strategies, cancelled)
                endpoints.append(ep)
                if ep.status == 'cancelled':
                    break
            if any(ep.status == 'cancelled' for ep in endpoints):
                obj = ObjectResult(detection.id, 'cancelled', '用户取消', endpoints)
                result.cancelled = True
            elif all(ep.status == 'accepted' for ep in endpoints):
                length = float(np.linalg.norm(np.asarray(endpoints[0].xyz_m) - endpoints[1].xyz_m))
                obj = ObjectResult(detection.id, 'accepted', '两个端点均通过', endpoints, length)
            else:
                reason = '；'.join(ep.name + ': ' + ep.reason for ep in endpoints if ep.status != 'accepted')
                obj = ObjectResult(detection.id, 'rejected', reason, endpoints)
        result.objects.append(obj)
        if progress:
            progress(i + 1, len(detections))
        if result.cancelled:
            break
    return result
