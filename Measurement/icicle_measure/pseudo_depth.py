"""Deterministic scene construction, Z16 export and editable pseudo RGB-D projects.

Model inference lives in pseudo_models and its optional Python environment.
This module only needs the workbench's existing numpy/OpenCV/scipy runtime.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt
from .models import Detection, Frame, Intrinsics, DEPTH_MIN_M, DEPTH_MAX_M
from .pseudo_geometry import apply_background_planes
from .io import read_image, write_image, write_json, read_json, load_detections

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'work' / 'pseudo_models'
MODEL_PYTHON = ROOT / 'work' / 'pseudo_env' / 'Scripts' / 'python.exe'
SAMPLE_STEMS = ('001_00045', '002_00002', '002_00058', '002_00041', '004_00064', '005_00036')
DEFAULT_RGB_DIR = ROOT / 'assets' / 'rgb'
DEFAULT_LABEL_DIR = ROOT.parent / 'final_results' / 'predictions' / 'labels'


@dataclass
class PseudoConfig:
    generation_version: int = 2
    width: int = 1280
    height: int = 720
    depth_scale_m_per_unit: float = .001
    near_m: float = .5
    far_m: float = 4.
    object_near_m: float = .5
    object_far_m: float = 2.
    fallback_m: float = 1.25
    length_min_m: float = .17
    length_max_m: float = .25
    length_nominal_min_m: float = .18
    length_nominal_max_m: float = .22
    hole_fraction: float = .10
    return_fraction: float = .05
    noise_sigma_m_at_1m: float = .001
    edge_fraction: float = .20
    seed: int = 42
    model_long_edge: int = 1036
    color_min_m: float = .5
    color_max_m: float = 4.
    intrinsics: Optional[Dict[str, float]] = None
    intrinsics_source: str = 'assumed_virtual_pinhole'

    def validate(self):
        if self.generation_version not in (1, 2):
            raise ValueError('不支持的表面生成版本')
        if type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) < 32:
            raise ValueError('输出尺寸须为至少 32 像素的整数')
        numbers = (self.depth_scale_m_per_unit, self.near_m, self.far_m, self.object_near_m,
                   self.object_far_m, self.fallback_m, self.length_min_m, self.length_max_m,
                   self.noise_sigma_m_at_1m, self.color_min_m, self.color_max_m,
                   self.length_nominal_min_m, self.length_nominal_max_m)
        if not np.isfinite(numbers).all() or min(numbers[:8]) <= 0 or self.noise_sigma_m_at_1m < 0:
            raise ValueError('深度、比例和先验须为有限有效值')
        if not (self.near_m < self.far_m and self.object_near_m <= self.fallback_m <= self.object_far_m
                and self.length_min_m <= self.length_nominal_min_m < self.length_nominal_max_m <= self.length_max_m
                and self.color_min_m < self.color_max_m):
            raise ValueError('范围顺序或假设基线无效')
        if self.far_m / self.depth_scale_m_per_unit > 65535:
            raise ValueError('深度范围超过 Z16 表示范围')
        for value in (self.hole_fraction, self.return_fraction, self.edge_fraction):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('缺陷比例须在 0..1')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('随机种子须为非负整数')
        if type(self.model_long_edge) is not int or self.model_long_edge < 224:
            raise ValueError('模型长边至少 224 像素')
        if self.intrinsics is not None:
            Intrinsics(**self.intrinsics).validate()
            if not self.intrinsics_source or self.intrinsics_source == 'assumed_virtual_pinhole':
                raise ValueError('手动输入内参时必须声明来源')


def letterbox(rgb, detections, width=1280, height=720):
    h, w = rgb.shape[:2]
    scale = min(width / w, height / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (width - rw) // 2, (height - rh) // 2
    sx, sy = rw / w, rh / h
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[top:top + rh, left:left + rw] = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    valid = np.zeros((height, width), bool)
    valid[top:top + rh, left:left + rw] = True
    # OpenCV pixel-centre convention, including half-pixel resizing offset.
    tx, ty = left + (sx - 1) / 2, top + (sy - 1) / 2
    transformed = []
    for d in detections:
        box = tuple(float(v) for v in (d.bbox[0] * sx + tx, d.bbox[1] * sy + ty,
                                      d.bbox[2] * sx + tx, d.bbox[3] * sy + ty))
        points = [None if p is None else (p[0] * sx + tx, p[1] * sy + ty) for p in d.keypoints]
        transformed.append(Detection(d.id, box, points))
    return canvas, transformed, valid, {'source_size': [w, h], 'output_size': [width, height],
        'resized_size': [rw, rh], 'offset': [left, top], 'scale_xy': [sx, sy],
        'pixel_center_translation': [tx, ty], 'convention': 'opencv_pixel_centers'}


def transformed_intrinsics(original, transform):
    sx, sy = transform['scale_xy']
    tx, ty = transform['pixel_center_translation']
    return Intrinsics(original.fx * sx, original.fy * sy, original.cx * sx + tx, original.cy * sy + ty)


def virtual_intrinsics(config, transform):
    if config.intrinsics:
        return transformed_intrinsics(Intrinsics(**config.intrinsics), transform)
    # Nominal virtual COLOR FOV; not a D435i factory calibration or a phone calibration.
    return Intrinsics(config.width / (2 * np.tan(np.deg2rad(69 / 2))),
                      config.height / (2 * np.tan(np.deg2rad(42 / 2))),
                      (config.width - 1) / 2, (config.height - 1) / 2)


def to_z16(depth_m, scale=.001):
    z = np.asarray(depth_m, dtype=np.float64)
    valid = np.isfinite(z) & (z > 0)
    raw = np.zeros(z.shape, np.uint16)
    values = np.rint(z[valid] / scale)
    if np.any(values > 65535) or np.any(values < 1):
        raise ValueError('深度不能按当前比例表示为非零 Z16；请调整比例或深度')
    raw[valid] = values.astype(np.uint16)
    return raw


def fixed_depth_rgb(depth, low=.5, high=4.):
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.clip((np.nan_to_num(depth) - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def depth_colorbar(low=.5, high=4., width=460):
    image = np.full((44, width, 3), (20, 32, 47), np.uint8)
    left, right = 26, width - 30
    image[3:19, left:right] = fixed_depth_rgb(np.linspace(low, high, right - left)[None], low, high)
    for value in np.linspace(low, high, 5):
        x = round(left + (value - low) / (high - low) * (right - left - 1))
        cv2.line(image, (x, 19), (x, 23), (220, 230, 240), 1)
        cv2.putText(image, '%.2f' % value, (max(0, x - 16), 38), cv2.FONT_HERSHEY_SIMPLEX, .35,
                    (220, 230, 240), 1, cv2.LINE_AA)
    return image


def _map_relative(q, valid, config):
    values = q[valid & np.isfinite(q)]
    if not len(values):
        raise ValueError('深度模型没有返回有效的相对深度')
    lo, hi = np.percentile(values, [2, 98])
    if hi - lo <= 1e-8:
        raise ValueError('深度模型输出近乎常数，不能生成有图像依据的背景层次')
    t = np.clip((q - lo) / (hi - lo), 0, 1)
    # Larger DA-V2 output means nearer. Map inverse depth monotonically.
    z = 1 / (1 / config.far_m + t * (1 / config.near_m - 1 / config.far_m))
    z[~valid | ~np.isfinite(q)] = 0
    return z.astype(np.float32), {'kind': 'assumed_affine_inverse_depth', 'q_p02': float(lo),
        'q_p98': float(hi), 'near_m': config.near_m, 'far_m': config.far_m,
        'scale_source': 'manually_assumed_not_calibrated', 'normalization_scope': 'per_image_recorded'}


@dataclass
class PseudoScene:
    rgb: np.ndarray
    relative: np.ndarray
    masks: np.ndarray
    valid: np.ndarray
    detections: List[Detection]
    intrinsics: Intrinsics
    transform: Dict[str, Any]
    model_info: Dict[str, Any]
    source_rgb: str = ''
    source_labels: str = ''
    original_masks: Optional[np.ndarray] = None
    background_offset: Optional[np.ndarray] = None
    edits: Dict[str, Any] = field(default_factory=lambda: {'objects': {}, 'actions': []})


@dataclass
class PseudoResult:
    scene: PseudoScene
    config: PseudoConfig
    frame: Frame
    clean_frame: Frame
    background_m: np.ndarray
    latent_m: np.ndarray
    clean_raw: np.ndarray
    observed_raw: np.ndarray
    instance_map: np.ndarray
    source_map: np.ndarray
    uncertainty: np.ndarray
    report: Dict[str, Any]


def scene_from_inference(rgb_path, labels_path, inference_path, config=None):
    config = config or PseudoConfig()
    config.validate()
    original = cv2.cvtColor(read_image(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    detections = load_detections(labels_path, original.shape[1], original.shape[0])
    rgb, detections, valid, transform = letterbox(original, detections, config.width, config.height)
    with np.load(inference_path, allow_pickle=False) as data:
        relative = data['relative'].copy()
        masks = data['masks'].astype(bool)
    if relative.shape != valid.shape or masks.shape != (len(detections), *valid.shape):
        raise ValueError('模型缓存与输入图片尺寸/实例数量不符')
    info = read_json(Path(inference_path).with_suffix('.json'))
    info['cache_file'] = str(inference_path)
    scene = PseudoScene(rgb, relative, masks & valid, valid, detections, virtual_intrinsics(config, transform),
                        transform, info, str(rgb_path), str(labels_path))
    scene.original_masks = scene.masks.copy()
    scene.background_offset = np.zeros(valid.shape, np.float32)
    return scene


def _fill_series(values, good):
    indexes = np.arange(len(values))
    return np.interp(indexes, indexes[good], values[good])


def _surface(mask, background, intrinsics, config, edit):
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None, {'status': 'needs_mask_review', 'reason': '掩膜有效像素不足'}
    xy = np.column_stack((xs, ys)).astype(float)
    center = xy.mean(axis=0)
    eigenvalues, axes = np.linalg.eigh(np.cov(xy.T))
    if eigenvalues[-1] < 2 or eigenvalues[-1] / max(eigenvalues[0], 1e-6) < 1.5:
        return None, {'status': 'needs_mask_review', 'reason': '掩膜没有可辨认的细长主方向'}
    axis = axes[:, -1]
    if axis[1] < 0:
        axis = -axis
    normal = np.array([-axis[1], axis[0]])
    longitudinal, transverse = (xy - center) @ axis, (xy - center) @ normal
    start, stop = float(longitudinal.min()), float(longitudinal.max())
    count = max(3, min(256, int(stop - start) + 1))
    t = (longitudinal - start) / max(stop - start, 1e-6)
    bins = np.minimum((t * (count - 1)).astype(int), count - 1)
    mid, radius, profile = np.zeros(count), np.zeros(count), np.zeros(count)
    good = np.zeros(count, bool)
    for b in range(count):
        members = bins == b
        if not members.any():
            continue
        vals = transverse[members]
        mid[b] = np.median(vals)
        radius[b] = max(.5, (np.percentile(vals, 98) - np.percentile(vals, 2)) / 2)
        profile[b] = np.median(background[ys[members], xs[members]])
        good[b] = True
    mid = gaussian_filter1d(_fill_series(mid, good), 2.)
    radius = gaussian_filter1d(_fill_series(radius, good), 2.)
    profile = gaussian_filter1d(_fill_series(profile, good), max(2., count / 20))
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~mask & (background > 0)
    target_z = float(np.median(background[mask]))
    surrounding_z = float(np.median(background[ring])) if ring.any() else None
    supported = surrounding_z is not None and surrounding_z - target_z >= .020
    saturated = bool(np.mean(background[mask] <= config.near_m + config.depth_scale_m_per_unit) > .10)
    if config.generation_version >= 2 and saturated:
        supported = False
    base = float(np.clip(target_z, config.object_near_m, config.object_far_m)) if supported else config.fallback_m
    manual = edit.get('base_m') is not None
    if manual:
        base = float(edit['base_m'])
    trend = np.clip(profile - np.median(profile), -.1 * base, .1 * base)
    trend_source = 'legacy_model_profile'
    residual_m = None
    if config.generation_version >= 2:
        # Only retain a robust linear trend; reflected background bumps are not ice geometry.
        coordinate = np.linspace(-.5, .5, count)
        design = np.column_stack((coordinate, np.ones(count)))
        weights = np.ones(count)
        for _ in range(5):
            fit = np.linalg.lstsq(design * np.sqrt(weights[:, None]), profile * np.sqrt(weights), rcond=None)[0]
            residual = profile - design @ fit
            mad = max(float(np.median(np.abs(residual - np.median(residual)))) * 1.4826, .002)
            weights = np.minimum(1., 1.5 * mad / np.maximum(np.abs(residual), 1e-9))
        residual_m = float(np.median(np.abs(residual)))
        reliable = supported and residual_m <= .015 and abs(fit[0]) <= .10
        trend = coordinate * fit[0] if reliable else np.zeros(count)
        trend_source = 'robust_model_linear_trend' if reliable else 'assumed_zero_axial_trend'
    slope = float(edit.get('slope_m', 0.))
    center_z = base + trend[bins] + slope * (t - .5)
    f_normal = 1 / np.sqrt((normal[0] / intrinsics.fx) ** 2 + (normal[1] / intrinsics.fy) ** 2)
    radius_m = radius[bins] * center_z / f_normal
    cross = np.clip((transverse - mid[bins]) / radius[bins], -1, 1)
    z = center_z - radius_m * np.sqrt(np.maximum(0, 1 - cross ** 2))
    if not np.isfinite(z).all() or np.min(z) <= 0:
        raise ValueError('人工参数使目标表面深度无效')
    endpoints = []
    for tt, b in [(0., 0), (1., count - 1)]:
        pixel = center + axis * (start + tt * (stop - start)) + normal * mid[b]
        zz = base + trend[b] + slope * (tt - .5) - radius[b] * base / f_normal
        endpoints.append(np.array([(pixel[0] - intrinsics.cx) * zz / intrinsics.fx,
                                   (pixel[1] - intrinsics.cy) * zz / intrinsics.fy, zz]))
    length = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    report = {'status': 'manual_surface' if manual or slope != 0 else 'assumed_surface', 'model_foreground_support': supported,
              'base_m': base, 'base_source': 'manual' if manual else 'model_candidate' if supported else 'assumed_distance_prior',
              'surrounding_candidate_m': surrounding_z, 'model_candidate_m': target_z,
              'mask_pixels': int(mask.sum()), 'silhouette_chord_m': length,
              'length_prior_satisfied': bool(config.length_min_m <= length <= config.length_max_m),
              'length_nominal_satisfied': bool(config.length_nominal_min_m <= length <= config.length_nominal_max_m),
              'length_prior_used_to_fit': False, 'slope_m': slope,
              'trend_source': trend_source, 'trend_residual_m': residual_m, 'model_depth_saturated': saturated,
              'axis_direction_px': axis.tolist(), 'warnings': []}
    if not supported:
        report['warnings'].append('模型无法区分目标与背景；使用显式假想表面')
    if config.generation_version >= 2 and saturated:
        report['warnings'].append('模型候选在映射近端饱和；不用截断值充当目标距离')
    if trend_source == 'assumed_zero_axial_trend':
        report['warnings'].append('模型轴向变化不可靠，采用零倾斜假设；可人工设置首尾深度差')
    if not report['length_prior_satisfied']:
        report['warnings'].append('假想轮廓弦长不在 %.0f–%.0f cm 极限参考范围；没有据此修改深度' % (
            config.length_min_m * 100, config.length_max_m * 100))
    elif not report['length_nominal_satisfied']:
        report['warnings'].append('假想轮廓弦长在极限参考范围内，但超出常见 %.0f–%.0f cm；仅提示' % (
            config.length_nominal_min_m * 100, config.length_nominal_max_m * 100))
    if surrounding_z is not None and np.median(z) >= surrounding_z:
        report['warnings'].append('假想目标未位于周围背景之前；测距场景假设可能不成立')
    return (ys, xs, z), report


def _patch_subset(mask, fraction, rng, sigma=2.):
    out = np.zeros(mask.shape, bool)
    n = int(round(int(mask.sum()) * fraction))
    if n <= 0:
        return out
    field = gaussian_filter(rng.standard_normal(mask.shape), sigma)
    ids = np.flatnonzero(mask)
    selected = ids[np.argpartition(field.ravel()[ids], max(0, len(ids) - n))[len(ids) - n:]]
    out.ravel()[selected] = True
    return out


def compose_scene(scene, config=None, cancel=lambda: False):
    config = config or PseudoConfig()
    config.validate()
    if scene.rgb.shape[:2] != (config.height, config.width):
        raise ValueError('尺寸修改需要重新生成模型输入')
    background, mapping = _map_relative(scene.relative, scene.valid, config)
    model_valid = scene.valid & np.isfinite(scene.relative)
    # Manual background edits never alter target surface parameters.
    union = scene.masks.any(axis=0) if len(scene.masks) else np.zeros(scene.valid.shape, bool)
    surface_candidates = background.copy()
    background, plane_pixels, plane_reports = apply_background_planes(
        background, model_valid, union, scene.edits.get('background_planes', []))
    if scene.background_offset is not None:
        editable = scene.valid & ~union & (background > 0)
        background[editable] += scene.background_offset[editable]
        if np.any(background[editable] <= 0):
            raise ValueError('背景修正使深度小于或等于零')
    latent = background.copy()
    instance_map = np.zeros(scene.valid.shape, np.uint16)
    source_map = model_valid.astype(np.uint8)  # 0 invalid, 1 model, 2 assumed, 3 manual
    source_map[plane_pixels] = 3
    if scene.background_offset is not None:
        source_map[(scene.background_offset != 0) & scene.valid & ~union] = 3
    uncertainty = np.zeros(scene.valid.shape, np.uint8)
    uncertainty[~model_valid] = 1
    overlap = scene.masks.sum(axis=0) > 1 if len(scene.masks) else np.zeros(scene.valid.shape, bool)
    reports = []
    for i, detection in enumerate(scene.detections):
        if cancel():
            raise InterruptedError('用户取消伪深度合成')
        mask = scene.masks[i] & model_valid & ~overlap
        edit = scene.edits.get('objects', {}).get(detection.id, {})
        surface, info = _surface(mask, surface_candidates, scene.intrinsics, config, edit)
        segmentation = scene.model_info.get('segmentation', [])
        if i < len(segmentation) and segmentation[i].get('needs_review'):
            info.setdefault('warnings', []).append(segmentation[i]['reason'])
            uncertainty[mask] |= 4
        if surface is not None and plane_reports:
            ring = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~union & model_valid
            if ring.any():
                near_background = float(np.median(background[ring]))
                info['edited_surrounding_m'] = near_background
                if float(np.median(surface[2])) >= near_background:
                    info['warnings'].append('修正背景后目标未在背景前方；保留冲突，不自动移动表面')
        info['id'] = detection.id
        if surface is not None:
            ys, xs, z = surface
            latent[ys, xs] = z
            instance_map[ys, xs] = i + 1
            changed_mask = scene.original_masks is not None and not np.array_equal(scene.masks[i], scene.original_masks[i])
            manually_changed = edit.get('base_m') is not None or edit.get('slope_m', 0) != 0 or changed_mask
            source_map[ys, xs] = 3 if manually_changed else 2
            uncertainty[ys, xs] |= 2  # Synthesized surface has no measured truth.
        else:
            uncertainty[scene.masks[i]] |= 4
            x1, y1, x2, y2 = detection.bbox
            uncertainty[max(0, int(y1)):min(config.height, int(np.ceil(y2)) + 1),
                        max(0, int(x1)):min(config.width, int(np.ceil(x2)) + 1)] |= 4
        if overlap.any() and np.any(scene.masks[i] & overlap):
            info.setdefault('warnings', []).append('与其他实例重叠；重叠像素无效，需人工修正')
        reports.append(info)
    latent[overlap | ~model_valid] = 0
    source_map[overlap] = 0
    uncertainty[overlap] |= 8
    objects = instance_map > 0
    clean_raw = to_z16(latent, config.depth_scale_m_per_unit)
    clean_m = clean_raw.astype(np.float32) * config.depth_scale_m_per_unit
    observed = latent.copy()
    rng = np.random.default_rng(config.seed)
    active = observed > 0
    noise = gaussian_filter(rng.standard_normal(observed.shape), .65)
    deviation = max(float(noise[active].std()), 1e-12) if active.any() else 1.
    observed[active] += (noise[active] / deviation * config.noise_sigma_m_at_1m * observed[active] ** 2)
    holes = _patch_subset(objects, config.hole_fraction, rng)
    returns = _patch_subset(objects & ~holes, config.return_fraction, rng)
    # A target's nearest non-target background is a separate hypothesized layer.
    bg_support = scene.valid & ~union & (background > 0)
    nearest_background = background.copy()
    if bg_support.any():
        iy, ix = distance_transform_edt(~bg_support, return_distances=False, return_indices=True)
        nearest_background = background[iy, ix]
    else:
        returns[:] = False
    observed[returns] = nearest_background[returns]
    edge = objects & ~cv2.erode(objects.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    wrong_edge = _patch_subset(edge & ~holes & ~returns, config.edge_fraction, rng, 1.) if bg_support.any() else np.zeros_like(edge)
    observed[wrong_edge] = nearest_background[wrong_edge]
    observed[holes] = 0
    observed[~model_valid | overlap] = 0
    observed[(observed <= 0) | ~np.isfinite(observed)] = 0
    uncertainty[holes] |= 16
    uncertainty[returns] |= 32
    uncertainty[wrong_edge] |= 64
    observed_raw = to_z16(observed, config.depth_scale_m_per_unit)
    observed_m = observed_raw.astype(np.float32) * config.depth_scale_m_per_unit
    max_depth = DEPTH_MAX_M
    provenance = {'input_kind': 'pseudo_rgbd', 'emulated_model': 'D435i_aligned_color_Z16',
        'is_synthetic': True, 'measurement_label': '基于伪深度的估计距离',
        'scale_source': 'manually_assumed_not_calibrated', 'intrinsics_source': config.intrinsics_source,
        'projection_assumption': 'virtual_pinhole_source_distortion_unverified',
        'geometry_mode': 'assumed_pinhole', 'undistorted': False, 'aligned_to_rgb': True,
        'depth_unit': 'raw', 'depth_scale_m_per_unit': config.depth_scale_m_per_unit,
        'source_rgb': scene.source_rgb, 'source_labels': scene.source_labels,
        'mapping': mapping, 'transform': scene.transform, 'models': scene.model_info,
        'generation_version': config.generation_version, 'background_planes': plane_reports,
        'prior_distance_m': [config.object_near_m, config.object_far_m],
        'prior_length_m': [config.length_min_m, config.length_max_m],
        'nominal_prior_length_m': [config.length_nominal_min_m, config.length_nominal_max_m],
        'sensor_errors': {'kind': 'engineering_assumptions_not_device_calibrated', 'seed': config.seed,
            'hole_fraction': config.hole_fraction, 'return_fraction': config.return_fraction,
            'noise_sigma_m_at_1m': config.noise_sigma_m_at_1m, 'edge_fraction': config.edge_fraction},
        'edits': scene.edits,
        'display_depth_range_m': [config.color_min_m, config.color_max_m],
        'source_map_legend': {'0': 'invalid', '1': 'model_background', '2': 'assumed_surface', '3': 'manual_surface'},
        'uncertainty_bits': {'1': 'padding_or_nonfinite_model', '2': 'unmeasured_assumed_surface', '4': 'mask_review', '8': 'overlap',
                             '16': 'hole', '32': 'background_return', '64': 'boundary_error'}}
    frame = Frame(scene.rgb, observed_m, DEPTH_MIN_M, max_depth, dict(provenance, depth_variant='observed'))
    clean_frame = Frame(scene.rgb, clean_m, DEPTH_MIN_M, max_depth, dict(provenance, depth_variant='complete'))
    report = {'objects': reports, 'mapping': mapping, 'background_planes': plane_reports, 'target_pixels': int(objects.sum()),
              'hole_pixels': int(holes.sum()), 'return_pixels': int(returns.sum()),
              'boundary_error_pixels': int(wrong_edge.sum()), 'overlap_pixels': int(overlap.sum()),
              'measurement_truth_available': False}
    frame.validate(); clean_frame.validate()
    return PseudoResult(scene, config, frame, clean_frame, background, latent, clean_raw,
                        observed_raw, instance_map, source_map, uncertainty, report)


def export_project(directory, result):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    s, cfg = result.scene, result.config
    for original, name in [(s.source_rgb, 'original_rgb'), (s.source_labels, 'original_prediction')]:
        if original and Path(original).is_file():
            target = root / (name + Path(original).suffix)
            if Path(original).resolve() != target.resolve():
                shutil.copyfile(original, target)
    write_image(root / 'rgb.png', s.rgb)
    write_image(root / 'depth_colorbar_m.png', depth_colorbar(cfg.color_min_m, cfg.color_max_m))
    write_json(root / 'detections.json', {'detections': [asdict(d) for d in s.detections]})
    for name, raw, frame in [('complete', result.clean_raw, result.clean_frame),
                              ('observed', result.observed_raw, result.frame)]:
        cv2.imencode('.png', raw)[1].tofile(str(root / ('depth_' + name + '_z16.png')))
        np.save(str(root / ('depth_' + name + '_m.npy')), frame.depth_m)
        write_image(root / ('preview_' + name + '.png'), fixed_depth_rgb(frame.depth_m, cfg.color_min_m, cfg.color_max_m))
        manifest = {'rgb': 'rgb.png', 'depth': 'depth_' + name + '_z16.png', 'detections': 'detections.json',
            'depth_unit': 'raw', 'depth_kind': 'z', 'depth_scale_m_per_unit': cfg.depth_scale_m_per_unit,
            'aligned_to_rgb': True, 'undistorted': False, 'geometry_mode': 'assumed_pinhole',
            'valid_range_m': [frame.min_depth_m, frame.max_depth_m],
            'intrinsics': asdict(s.intrinsics), 'intrinsics_image_size': [cfg.width, cfg.height], 'source': frame.metadata}
        write_json(root / ('input_' + name + '.json'), manifest)
        if name == 'observed':
            write_json(root / 'input.json', manifest)
    np.save(str(root / 'model_relative.npy'), s.relative)
    np.save(str(root / 'background_m.npy'), result.background_m)
    np.save(str(root / 'latent_surface_m.npy'), result.latent_m)
    for name, array in [('valid_region', s.valid.astype(np.uint8) * 255), ('instances', result.instance_map),
                         ('valid_depth_complete', (result.clean_raw > 0).astype(np.uint8) * 255),
                         ('valid_depth_observed', (result.observed_raw > 0).astype(np.uint8) * 255),
                         ('source_map', result.source_map), ('uncertainty', result.uncertainty)]:
        cv2.imencode('.png', array)[1].tofile(str(root / (name + '.png')))
    np.savez_compressed(str(root / 'editable_layers.npz'), relative=s.relative, masks=s.masks,
        original_masks=s.original_masks if s.original_masks is not None else s.masks,
        valid=s.valid, background_offset=s.background_offset if s.background_offset is not None else np.zeros(s.valid.shape, np.float32))
    project = {'version': 1, 'config': asdict(cfg), 'transform': s.transform, 'models': s.model_info,
        'intrinsics': asdict(s.intrinsics), 'source_rgb': s.source_rgb, 'source_labels': s.source_labels,
        'edits': s.edits, 'report': result.report, 'measurement_label': '基于伪深度的估计距离',
        'depth_format': 'aligned Z16, zero=invalid, raw*scale=Z metres',
        'display_colorbar_m': [cfg.color_min_m, cfg.color_max_m]}
    write_json(root / 'pseudo_project.json', project)
    write_json(root / 'generation_report.json', result.report)
    return root / 'input.json'


def load_project(path):
    path = Path(path)
    root = path.parent if path.is_file() else path
    meta = read_json(root / 'pseudo_project.json')
    if meta.get('version') != 1:
        raise ValueError('不支持该伪深度项目版本')
    saved_config = dict(meta['config'])
    saved_config.setdefault('generation_version', 1)  # Existing projects retain their original geometry.
    cfg = PseudoConfig(**saved_config); cfg.validate()
    rgb = cv2.cvtColor(read_image(root / 'rgb.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ds = load_detections(root / 'detections.json', rgb.shape[1], rgb.shape[0])
    with np.load(str(root / 'editable_layers.npz'), allow_pickle=False) as layers:
        s = PseudoScene(rgb, layers['relative'].copy(), layers['masks'].astype(bool), layers['valid'].astype(bool),
            ds, Intrinsics(**meta['intrinsics']), meta['transform'], meta['models'],
            meta['source_rgb'], meta['source_labels'], layers['original_masks'].astype(bool),
            layers['background_offset'].copy(), meta['edits'])
    return compose_scene(s, cfg)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
