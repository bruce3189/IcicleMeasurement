"""Endpoint evidence gates. Diagnostic models never supply output depth."""
import math
import numpy as np


def line_support(ids, width, detection, config):
    ys, xs = np.divmod(np.asarray(ids, dtype=np.int64), width)
    a, b = np.asarray(detection.keypoints, dtype=float)
    ab = b - a
    pixels = np.column_stack((xs, ys))
    t = np.clip(((pixels - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    distance = np.linalg.norm(pixels - (a + t[:, None] * ab), axis=1)
    x1, y1, x2, y2 = detection.bbox
    band = max(config.band_min_px, (x2 - x1) * config.band_width_ratio)
    return ((distance <= band) & (xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2))


def anchor_evidence(ids, width, point, detection, config):
    ids = np.asarray(ids, dtype=np.int64)
    ids = ids[line_support(ids, width, detection, config)]
    ys, xs = np.divmod(ids, width)
    distance = np.hypot(xs - point[0], ys - point[1])
    trials = []
    radii = list(config.anchor_radii)
    if config.endpoint_depth_mode == 'near_cluster_mean' and len(distance) >= config.anchor_min_samples:
        # Samples already come from the current clipped search window. Expand
        # association only as far as the nearest sufficient spatial support;
        # never fetch pixels outside that window or move the original keypoint.
        needed_radius = int(math.ceil(float(np.partition(distance, config.anchor_min_samples - 1)[config.anchor_min_samples - 1])))
        if needed_radius > radii[-1]:
            radii.append(needed_radius)
    for radius in radii:
        count = int(np.count_nonzero(distance <= radius))
        trials.append({'radius_px': radius, 'count': count})
        if count >= config.anchor_min_samples:
            return {'passed': True, 'radius_px': radius, 'count': count, 'trials': trials}
    return {'passed': False, 'radius_px': radii[-1],
            'count': trials[-1]['count'], 'trials': trials}


def subgroup_evidence(depths, config, splitter):
    labels = splitter(depths, config)
    groups = [depths[labels == k] for k in (1, 2)]
    counts = [len(z) for z in groups]
    result = {'ambiguous': False, 'counts': counts,
              'fractions': [n / max(len(depths), 1) for n in counts],
              'centers_m': [], 'center_gap_m': None, 'separation': None, 'gap_m': None}
    if not all(counts):
        return result
    near, far = groups
    centers = [float(z.mean()) for z in groups]
    delta = abs(centers[1] - centers[0])
    separation = delta / max(float(np.sqrt(near.var() + far.var())), 1e-9)
    gap = float(np.percentile(far, 10) - np.percentile(near, 90))
    result.update(centers_m=centers, center_gap_m=delta, separation=separation, gap_m=gap)
    result['ambiguous'] = bool(min(counts) >= config.subgroup_min_samples
        and min(result['fractions']) >= config.subgroup_min_fraction
        and delta >= config.subgroup_center_gap_m
        and separation >= config.subgroup_separation_min and gap >= config.subgroup_gap_min_m)
    return result


def plane_evidence(ids, depths, width, point, mean_depth, config):
    result = {'reliable': False, 'representative': False, 'predicted_depth_m': None,
              'bias_m': None, 'uncertainty_m': None, 'inlier_count': 0,
              'condition': None, 'reason': ''}
    if len(ids) < config.plane_min_samples or mean_depth is None:
        result['reason'] = '局部平面样本不足'
        return result
    ys, xs = np.divmod(np.asarray(ids, dtype=np.int64), width)
    xy = np.column_stack((xs - point[0], ys - point[1])).astype(float)
    scale = np.maximum(np.ptp(xy, axis=0), 1.)
    design = np.column_stack((np.ones(len(ids)), xy / scale))
    keep = np.ones(len(ids), bool)
    # Deterministic MAD trimming; retained samples for OUTPUT remain unchanged.
    for _ in range(4):
        matrix = design[keep]
        if len(matrix) < config.plane_min_samples or np.linalg.matrix_rank(matrix) < 3:
            result['reason'] = '局部平面缺少二维空间支持'
            return result
        condition = float(np.linalg.cond(matrix))
        result['condition'] = condition
        if not np.isfinite(condition) or condition > config.plane_max_condition:
            result['reason'] = '局部平面拟合病态'
            return result
        beta = np.linalg.lstsq(matrix, depths[keep], rcond=None)[0]
        residual = depths - design @ beta
        median = float(np.median(residual[keep]))
        sigma = max(1.4826 * float(np.median(np.abs(residual[keep] - median))), config.mad_floor_m)
        next_keep = np.abs(residual - median) <= config.mad_multiplier * sigma
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    matrix = design[keep]
    result['inlier_count'] = int(keep.sum())
    if (keep.sum() < config.plane_min_samples or keep.mean() < config.plane_min_inlier_fraction
            or np.linalg.matrix_rank(matrix) < 3):
        result['reason'] = '局部平面内点不足或缺少二维支持'
        return result
    condition = float(np.linalg.cond(matrix))
    result['condition'] = condition
    if condition > config.plane_max_condition:
        result['reason'] = '局部平面拟合病态'
        return result
    beta = np.linalg.lstsq(matrix, depths[keep], rcond=None)[0]
    residual = depths[keep] - matrix @ beta
    # Numerical model diagnostic only; spatially correlated depth pixels mean
    # this formal standard error is NOT a calibrated probability interval.
    variance = float(residual @ residual) / max(int(keep.sum()) - 3, 1)
    uncertainty = 2 * math.sqrt(max(0., variance * float(np.linalg.pinv(matrix.T @ matrix)[0, 0])))
    prediction = float(beta[0])
    bias = abs(float(mean_depth) - prediction)
    reliable = bool(np.isfinite(prediction) and prediction > 0 and np.isfinite(uncertainty)
                    and uncertainty <= config.plane_bias_max_m)
    result.update(reliable=reliable, predicted_depth_m=prediction, bias_m=bias, uncertainty_m=uncertainty,
                  representative=bool(reliable and bias + uncertainty <= config.plane_bias_max_m))
    result['reason'] = ('' if result['representative'] else '局部平面拟合不确定度过大' if not reliable
                        else '邻域算术均值不能代表原端点深度（存在梯度偏差）')
    return result


def growth_evidence(previous, ids, target, far, depths, config):
    result = {'new_target': 0, 'new_far': 0, 'required_target': config.growth_min_samples,
              'required_far': config.growth_min_samples, 'new_target_mean_m': None,
              'retained_target_mean_m': None, 'mean_difference_m': None,
              'size_passed': False, 'consistency_passed': False}
    if previous is None:
        return result
    new_region = ~np.isin(ids, np.asarray(previous.valid_ids, dtype=np.int64))
    added_target, added_far = target & new_region, far & new_region
    retained = target & np.isin(ids, np.asarray(previous.target_ids, dtype=np.int64))
    need_target = max(config.growth_min_samples, math.ceil(previous.target_count * config.growth_min_fraction))
    need_far = max(config.growth_min_samples, math.ceil(previous.far_count * config.growth_min_fraction))
    n_target, n_far = int(added_target.sum()), int(added_far.sum())
    result.update(new_target=n_target, new_far=n_far, required_target=need_target, required_far=need_far,
                  size_passed=bool(n_target >= need_target and n_far >= need_far))
    if added_target.any() and retained.any():
        new_mean, old_mean = float(depths[added_target].mean()), float(depths[retained].mean())
        difference = abs(new_mean - old_mean)
        result.update(new_target_mean_m=new_mean, retained_target_mean_m=old_mean,
                      mean_difference_m=difference, consistency_passed=bool(difference <= config.growth_mean_max_m))
    return result


def coverage_evidence(ids, width, near, far, target, config):
    ys, xs = np.divmod(ids, width)
    cells = np.column_stack((xs // config.coverage_cell_px, ys // config.coverage_cell_px))
    counts = {name: int(len(np.unique(cells[mask], axis=0)))
              for name, mask in (('near', near), ('far', far), ('target', target))}
    counts['passed'] = min(counts.values()) >= config.coverage_min_cells
    return counts
