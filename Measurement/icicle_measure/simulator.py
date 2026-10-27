"""Reproducible fixtures with metric ground truth (not sensor realism)."""
from dataclasses import dataclass
import numpy as np
from .models import Frame, Detection, Intrinsics
from .core import deproject

SCENARIOS = {
    'clean': '无噪声双深度端点', 'noisy': '高斯噪声',
    'holes': '端点空洞', 'background_endpoint': '端点误读背景',
    'outliers': '噪声与离群点', 'sparse': '样本不足',
    'overlap': '两簇重叠', 'background': '只有背景',
    'empty': '全部无效', 'drift': '扩窗中心漂移',
    'membership': '扩窗成员变化', 'interference': '邻近物体干扰',
    'multi': '多目标与缺失关键点', 'edge': '图像边缘',
    'tilted': '弯曲冰凌（关键点连线偏离主轴）',
    'adjacent': '两根紧挨冰凌（窗口内双近簇）',
    'slope_bias': '端点斜面（均值偏离端点）',
    'touching_surfaces': '相接双表面（混合均值）',
}


@dataclass
class Simulation:
    frame: Frame
    detections: list
    intrinsics: Intrinsics
    truth: dict
    scenario: str
    seed: int


def generate(scenario='noisy', seed=42, noise_m=0.003):
    if scenario not in SCENARIOS:
        raise ValueError('未知模拟场景: ' + scenario)
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    intrinsics = Intrinsics(320., 320., 159.5, 119.5)
    depth = np.full((h, w), 2.4, dtype=np.float64)
    rgb = np.full((h, w, 3), (38, 49, 59), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    center = 5 if scenario == 'edge' else (92 if scenario == 'multi' else 160)
    detections, truth = [], {}

    def add_object(cx, identifier, z_tip, z_base):
        objmask = (abs(xx - cx) <= 5) & (yy >= 40) & (yy <= 200)
        # Each endpoint has a locally flat patch. The middle smoothly connects
        # the two depths, so endpoint Z differs without biasing local means.
        blend = np.clip((yy - 80) / 80, 0, 1)
        z = z_base + blend * (z_tip - z_base)
        depth[objmask] = z[objmask]
        rgb[objmask] = (167, 222, 237)
        points = [(float(cx), 200.), (float(cx), 40.)]
        box = (float(max(0, cx - 12)), 40., float(cx + 12), 200.)
        detections.append(Detection(identifier, box, points))
        xyz = [deproject(p, z0, intrinsics) for p, z0 in zip(points, (z_tip, z_base))]
        truth[identifier] = {'depth_m': [z_tip, z_base], 'xyz_m': xyz,
                             'distance_m': float(np.linalg.norm(np.subtract(*xyz)))}
        return objmask

    mask = add_object(center, 'ice-1', 1.5, 1.3)
    if scenario == 'multi':
        mask |= add_object(232, 'ice-2', 1.6, 1.4)
        detections.append(Detection('incomplete', (140., 50., 180., 180.), [(160., 180.), None]))
    if scenario != 'clean':
        depth += rng.normal(0, noise_m, depth.shape)
    if scenario in ('holes', 'background_endpoint'):
        for px, py in detections[0].keypoints:
            hole = (xx - px) ** 2 + (yy - py) ** 2 <= 4
            depth[hole] = 0 if scenario == 'holes' else 2.4
    if scenario == 'outliers':
        outliers = rng.random(depth.shape) < 0.025
        depth[outliers] = rng.choice([0.2, 4.8, 0], outliers.sum())
    if scenario == 'sparse':
        depth[:] = 0
        depth[200, center], depth[40, center] = 1.5, 1.3
        depth[201, center + 1], depth[39, center + 1] = 2.4, 2.4
    if scenario == 'overlap':
        depth[:] = 1.5 + rng.normal(0, .025, depth.shape)
        depth[mask] -= .008
    if scenario == 'background':
        depth[:] = 2.4 + rng.normal(0, noise_m, depth.shape)
    if scenario == 'empty':
        depth[:] = 0
        depth[0, 0], depth[0, 1] = np.nan, np.inf
    if scenario == 'drift':
        for px, py in detections[0].keypoints:
            local = mask & (abs(yy - py) <= 36)
            depth[local] += 0.018 * abs(yy[local] - py)
    if scenario == 'membership':
        # Near-center surface A, larger surrounding surface B: expanding the
        # window changes the Otsu split and the foreground component membership.
        for px, py in detections[0].keypoints:
            r = np.maximum(abs(xx - px), abs(yy - py))
            depth[(r <= 3) & mask] = 1.5
            depth[(r > 3) & (r <= 35) & mask] = 0.8 + 0.025 * r[(r > 3) & (r <= 35) & mask]
    if scenario == 'interference':
        # Two disconnected equally supported near strips at distinct Z.
        depth[mask] = 2.4
        left = (xx >= center - 3) & (xx <= center - 1) & (yy >= 40) & (yy <= 200)
        right = (xx >= center + 1) & (xx <= center + 3) & (yy >= 40) & (yy <= 200)
        depth[left], depth[right] = 1.3, 1.38
    if scenario == 'tilted':
        # Straight icicle tilted 8° from vertical.  A naive detector places
        # keypoints vertically at (cx, 200) and (cx, 40); the real tip/base
        # are at (cx+11, 200) and (cx-11, 40).  The keypoint line is a vertical
        # chord that misses the endpoints by 11 px (> band 6 px), so spatial
        # support only covers the icicle middle.  The full component (all
        # near pixels) is still selected as target — this tests whether the
        # depth estimate stays correct when the band is misaligned, and
        # whether the XYZ / distance stays correct when the keypoint pixel
        # is off the true endpoint.
        depth[:] = 2.4
        angle = np.deg2rad(8.0)
        dx = (yy - 120) * np.tan(angle)
        x_center = center + dx
        objmask = (abs(xx - x_center) <= 5) & (yy >= 40) & (yy <= 200)
        blend = np.clip((yy - 80) / 80, 0, 1)
        z = 1.3 + blend * (1.5 - 1.3)
        depth[objmask] = z[objmask]
        rgb[objmask] = (167, 222, 237)
        depth += rng.normal(0, noise_m, depth.shape)  # re-add noise after reset
        tip_x = float(center + (200 - 120) * np.tan(angle))
        base_x = float(center + (40 - 120) * np.tan(angle))
        truth['ice-1'] = {'depth_m': [1.5, 1.3],
                          'xyz_m': [deproject((tip_x, 200.), 1.5, intrinsics),
                                    deproject((base_x, 40.), 1.3, intrinsics)],
                          'distance_m': float(np.linalg.norm(np.subtract(
                              deproject((tip_x, 200.), 1.5, intrinsics),
                              deproject((base_x, 40.), 1.3, intrinsics))))}
        detections[0] = Detection('ice-1',
                                  (float(max(0, center - 20)), 40.,
                                   float(center + 20), 200.),
                                  [(float(center), 200.), (float(center), 40.)])
    if scenario == 'adjacent':
        # Two icicles 11 px apart (1 px gap → 8-connected into one
        # connected component).  ice-2 is 50 mm deeper than ice-1.  Because
        # they merge into one component, select_target cannot separate them;
        # the target includes both icicles and the arithmetic mean is biased.
        # Noise is kept so the MAD filter does not outright discard ice-2.
        cx2 = center + 11
        om2 = (abs(xx - cx2) <= 5) & (yy >= 40) & (yy <= 200)
        blend2 = np.clip((yy - 80) / 80, 0, 1)
        z2 = 1.35 + blend2 * (1.55 - 1.35)
        depth[om2] = z2[om2]
        rgb[om2] = (167, 222, 237)
        detections[0] = Detection('ice-1',
                                  (float(max(0, center - 12)), 40.,
                                   float(center + 24), 200.),
                                  [(float(center), 200.), (float(center), 40.)])
        truth['ice-1'] = {'depth_m': [1.5, 1.3],
                          'xyz_m': [deproject((float(center), 200.), 1.5, intrinsics),
                                    deproject((float(center), 40.), 1.3, intrinsics)],
                          'distance_m': float(np.linalg.norm(np.subtract(
                              deproject((float(center), 200.), 1.5, intrinsics),
                              deproject((float(center), 40.), 1.3, intrinsics))))}
    if scenario == 'slope_bias':
        depth[mask] = 1.3 + (yy[mask] - 40) * .002
        xyz = [deproject(p, z, intrinsics) for p, z in zip(detections[0].keypoints, (1.62, 1.3))]
        truth['ice-1'] = {'depth_m': [1.62, 1.3], 'xyz_m': xyz,
                          'distance_m': float(np.linalg.norm(np.subtract(*xyz)))}
    if scenario == 'touching_surfaces':
        # Touching in image space, 40 mm apart in depth. Ten-mm noise prevents
        # the global near-cluster MAD filter from removing the second surface.
        other_rng = np.random.default_rng(seed)
        strip = (xx >= center - 5) & (xx <= center + 6) & (yy >= 40) & (yy <= 200)
        base = 1.3 + np.clip((yy - 80) / 80, 0, 1) * .2
        surfaces = base + (xx > center) * .04 + other_rng.normal(0, .01, depth.shape)
        depth[strip] = surfaces[strip]
        rgb[strip] = (167, 222, 237)
    return Simulation(Frame(rgb, depth, metadata={'input_kind': 'simulated', 'scenario': scenario,
                                                'seed': seed, 'depth_unit': 'm'}),
                      detections, intrinsics, truth, scenario, seed)
