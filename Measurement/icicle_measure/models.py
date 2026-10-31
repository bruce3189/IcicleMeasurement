from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
import math
import numpy as np


DEPTH_MIN_M = 0.001
DEPTH_MAX_M = 5.0


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self):
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError('相机内参必须为有限数值')
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError('fx 和 fy 必须大于零')


@dataclass
class Frame:
    rgb: np.ndarray
    depth_m: np.ndarray
    min_depth_m: float = DEPTH_MIN_M
    max_depth_m: float = DEPTH_MAX_M
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        if not isinstance(self.rgb, np.ndarray) or not isinstance(self.depth_m, np.ndarray):
            raise ValueError('RGB 和深度必须为 NumPy 数组')
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ValueError('RGB 必须为 H×W×3 uint8 图像')
        if self.depth_m.ndim != 2 or self.depth_m.shape != self.rgb.shape[:2]:
            raise ValueError('RGB 与深度尺寸不一致；必须预先对齐')
        if not self.depth_m.size:
            raise ValueError('图像尺寸不得为空')
        if not (np.issubdtype(self.depth_m.dtype, np.integer) or
                np.issubdtype(self.depth_m.dtype, np.floating)):
            raise ValueError('深度必须为数值数组')
        if not (all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (self.min_depth_m, self.max_depth_m))
                and 0 < self.min_depth_m < self.max_depth_m):
            raise ValueError('有效深度范围必须为有限正数且 min < max')


@dataclass
class Detection:
    id: str
    bbox: Tuple[float, float, float, float]
    keypoints: List[Optional[Tuple[float, float]]]


@dataclass
class Config:
    radii: Tuple[int, ...] = (4, 5, 8, 12, 18, 25, 35, 45, 60)
    # Active simple mode uses strict lower bounds; legacy fields remain below.
    near_min_exclusive: int = 26
    far_min_exclusive: int = 46
    # Accepted on loading old parameter files only; inactive in the current simple mode.
    far_to_near_min_exclusive: float = 0.5
    separation_min_exclusive: float = 3.0
    bbox_padding: int = 10
    mad_multiplier: float = 3.5
    mad_floor_m: float = 0.001
    separation_min: float = 3.0
    separation_full: float = 4.0
    gap_min_m: float = 0.020
    min_samples: int = 64
    full_samples: int = 100
    center_abs_m: float = 0.005
    center_rel: float = 0.005
    membership_min: float = 0.85
    membership_full: float = 0.95
    confidence_min: float = 0.8
    # A single window that satisfies every hard check is enough to accept.
    # Keeping the field (default 1) preserves the acceptance gate and the
    # GUI/test hooks; raise it to require several consecutive passes again.
    consecutive: int = 1
    band_min_px: float = 3.0
    band_width_ratio: float = 0.15
    component_min: int = 6
    ambiguity_size_ratio: float = 0.5
    ambiguity_depth_m: float = 0.020
    anchor_radii: Tuple[int, ...] = (3, 5)
    anchor_min_samples: int = 6
    subgroup_min_fraction: float = 0.20
    subgroup_min_samples: int = 12
    subgroup_center_gap_m: float = 0.020
    subgroup_separation_min: float = 2.0
    subgroup_gap_min_m: float = 0.005
    plane_min_samples: int = 12
    plane_min_inlier_fraction: float = 0.60
    plane_max_condition: float = 10000.0
    plane_bias_max_m: float = 0.005
    growth_min_samples: int = 12
    growth_min_fraction: float = 0.10
    growth_mean_max_m: float = 0.005
    coverage_cell_px: int = 2
    coverage_min_cells: int = 12
    endpoint_depth_mode: str = 'near_cluster_simple'

    def active_dict(self):
        values = asdict(self)
        if self.endpoint_depth_mode == 'near_cluster_simple':
            names = ('radii', 'mad_multiplier', 'mad_floor_m', 'near_min_exclusive',
                     'far_min_exclusive',
                     'separation_min_exclusive', 'endpoint_depth_mode')
            return {k: values[k] for k in names}
        return values

    def validate(self):
        if self.endpoint_depth_mode == 'near_cluster_simple':
            from .window_analysis import WindowAnalysisConfig
            WindowAnalysisConfig(**{k: getattr(self, k) for k in
                                  WindowAnalysisConfig.__dataclass_fields__}).validate()
            return
        if self.endpoint_depth_mode not in ('near_cluster_mean', 'strict_local'):
            raise ValueError('endpoint_depth_mode 须为 near_cluster_mean 或 strict_local')
        if not self.radii or any(type(r) is not int or r <= 0 for r in self.radii):
            raise ValueError('搜索半径必须是正整数列表')
        if list(self.radii) != sorted(set(self.radii)) or len(self.radii) < self.consecutive + 1:
            raise ValueError('搜索半径必须递增且足够完成连续稳定性比较')
        integer_fields = ('bbox_padding', 'min_samples', 'full_samples', 'consecutive', 'component_min')
        integer_fields += ('anchor_min_samples', 'subgroup_min_samples', 'plane_min_samples',
                           'growth_min_samples', 'coverage_cell_px', 'coverage_min_cells')
        if (not self.anchor_radii or any(type(r) is not int or r <= 0 for r in self.anchor_radii)
                or list(self.anchor_radii) != sorted(set(self.anchor_radii))):
            raise ValueError('锚区半径必须为递增正整数列表')
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == 'bbox_padding' else 1):
                raise ValueError(name + ' 必须为有效整数')
        for name, value in asdict(self).items():
            if name in integer_fields or name in ('radii', 'anchor_radii', 'endpoint_depth_mode'):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' 必须为有限正数')
        for name in ('membership_min', 'membership_full', 'confidence_min', 'ambiguity_size_ratio'):
            if getattr(self, name) > 1:
                raise ValueError(name + ' 不得大于 1')
        for name in ('subgroup_min_fraction', 'plane_min_inlier_fraction', 'growth_min_fraction'):
            if getattr(self, name) > 1:
                raise ValueError(name + ' 不得大于 1')
        if self.subgroup_min_fraction > .5 or self.plane_min_samples < 3:
            raise ValueError('子群最小占比不得超过 0.5，拟合最少点数不得小于 3')
        if self.full_samples < self.min_samples or self.separation_full < self.separation_min:
            raise ValueError('满分阈值不得小于硬阈值')
        if self.membership_full < self.membership_min:
            raise ValueError('成员稳定满分阈值不得小于硬阈值')

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        if 'radii' in values:
            values['radii'] = tuple(values['radii'])
        if 'anchor_radii' in values:
            values['anchor_radii'] = tuple(values['anchor_radii'])
        obj = cls(**values)
        obj.validate()
        return obj


@dataclass
class WindowRecord:
    radius: int
    bounds: Tuple[int, int, int, int]
    near_count: int = 0
    far_count: int = 0
    target_count: int = 0
    center_m: Optional[float] = None
    separation: Optional[float] = None
    gap_m: Optional[float] = None
    center_change_m: Optional[float] = None
    membership: Optional[float] = None
    score: Optional[float] = None
    scores: Dict[str, float] = field(default_factory=dict)
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    # Pixel ids are y * image_width + x. Retained for audit and GUI visualization.
    near_ids: List[int] = field(default_factory=list)
    far_ids: List[int] = field(default_factory=list)
    target_ids: List[int] = field(default_factory=list)
    valid_ids: List[int] = field(default_factory=list)
    anchor: Dict[str, Any] = field(default_factory=dict)
    subgroups: Dict[str, Any] = field(default_factory=dict)
    plane: Dict[str, Any] = field(default_factory=dict)
    growth: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    consecutive_passes: int = 0
    advisories: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class EndpointResult:
    name: str
    pixel: Optional[Tuple[float, float]]
    status: str = 'rejected'
    reason: str = ''
    depth_m: Optional[float] = None
    xyz_m: Optional[Tuple[float, float, float]] = None
    confidence: Optional[float] = None
    history: List[WindowRecord] = field(default_factory=list)


@dataclass
class ObjectResult:
    id: str
    status: str
    reason: str
    endpoints: List[EndpointResult] = field(default_factory=list)
    distance_m: Optional[float] = None


@dataclass
class MeasurementResult:
    objects: List[ObjectResult]
    config: Dict[str, Any]
    intrinsics: Dict[str, float]
    image_size: Tuple[int, int]
    cancelled: bool = False
    assumptions: str = '输入深度为对齐、去畸变后的光轴 Z；冰凌位于背景前方；邻域目标深度代表原端点射线深度。评分不是概率。'
    source: Dict[str, Any] = field(default_factory=dict)
