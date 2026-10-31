"""Near-first endpoint estimator: sample count and separation, no rule score."""
import numpy as np
from .models import EndpointResult, WindowRecord


def separation(near, far):
    # Numerical zero guard only; this is not a physical noise/accuracy setting.
    return float(abs(near.mean() - far.mean()) /
                 max(float(np.sqrt(near.var() + far.var())), 1e-9))


def endpoint_estimate(frame, detection, index, intrinsics, config, strategies, cancelled):
    # The same numeric path supports first-pass measurement and full diagnostics.
    from .core import deproject
    from .window_analysis import WindowAnalysisConfig, analyze_endpoint
    cfg = WindowAnalysisConfig(**{k: getattr(config, k) for k in
                                WindowAnalysisConfig.__dataclass_fields__})
    data = analyze_endpoint(frame, detection, index, cfg, cancelled,
                            stop_on_first_candidate=True, clusterer=strategies.clusterer)
    point = detection.keypoints[index]
    result = EndpointResult(('tip', 'base')[index], tuple(point), confidence=None)
    for item in data['windows']:
        record = WindowRecord(item['radius_px'], tuple(item['bounds']), score=None)
        record.valid_ids = item['valid_ids']
        record.near_ids = record.target_ids = item['target_ids']
        record.far_ids = item['reference_ids']
        stats = item['selected']
        if stats:
            record.near_count = record.target_count = stats['near_n']
            record.far_count = stats['far_n']
            record.center_m = stats['near_mean_m']
            record.separation = stats['S']
        record.center_change_m = item['mean_change_m']
        record.checks = item['checks']
        record.reasons = item['reasons']
        record.subgroups = {k: item[k] for k in
                           ('basic', 'selected', 'internal_split', 'identity_overlap_n',
                            'valid_n', 'invalid_n', 'mad_removed_n',
                            'mad_removed_initial_n', 'mad_removed_internal_n', 'unselected_n')}
        result.history.append(record)
    if data['status'] == 'cancelled':
        result.status, result.reason = 'cancelled', '用户取消'
    elif result.history and all(result.history[-1].checks.values()):
        result.status = 'accepted'
        result.reason = '近簇数量、远簇数量及分离度同时达标，首个达标窗立即接受'
        result.depth_m = result.history[-1].center_m
        result.xyz_m = deproject(point, result.depth_m, intrinsics)
    else:
        reasons = result.history[-1].reasons if result.history else ['无有效窗口']
        result.reason = '达到短边上限仍未达标：' + '；'.join(reasons)
    return result
