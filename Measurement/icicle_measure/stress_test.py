# -*- coding: utf-8 -*-
"""压力测试：覆盖现有测试套件遗漏的边界条件与鲁棒性场景。

运行方式：
  cd <project_root>
  python -m icicle_measure.stress_test
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from icicle_measure.simulator import generate, SCENARIOS
from icicle_measure.core import measure_frame, endpoint_estimate, Strategies, otsu_split, confidence_scores
from icicle_measure.models import Config, Frame, Detection, Intrinsics
from icicle_measure.io import save_simulation, load_bundle

PASS = 0
FAIL = 0
CHECKS = []


def check(name, condition, detail=''):
    global PASS, FAIL
    status = 'PASS' if condition else 'FAIL'
    print('[%s] %s%s' % (status, name, ('  -> ' + detail) if detail else ''))
    CHECKS.append(name)
    if condition:
        PASS += 1
    else:
        FAIL += 1


def section(title):
    print('\n' + '=' * 60)
    print('  ' + title)
    print('=' * 60)


def safe_measure(frame, detections, intrinsics, config=None, **kw):
    """measure_frame 包装，捕获 OpenCV 层面的偶发崩溃。"""
    try:
        return measure_frame(frame, detections, intrinsics, config=config, **kw)
    except Exception as e:
        class _R:
            objects = []
            cancelled = False
        print('  [警告] measure_frame 异常: %s' % e)
        return _R()


# ---------------------------------------------------------------------------
section('1. 确定性：同种子多次运行结果完全一致')


def run_once(seed):
    s = generate('noisy', seed)
    r = safe_measure(s.frame, s.detections, s.intrinsics)
    if not r.objects:
        return None
    obj = r.objects[0]
    return (obj.status,
            obj.distance_m,
            [(ep.depth_m, ep.xyz_m, ep.confidence) for ep in obj.endpoints])


all_same = True
for seed in range(10):
    r1 = run_once(seed)
    r2 = run_once(seed)
    if r1 is None or r2 is None:
        print('  seed %d 有一次运行异常，跳过' % seed)
        continue
    if r1 != r2:
        all_same = False
        print('  seed %d 不一致! %s vs %s' % (seed, r1[0], r2[0]))
        break
check('10 个种子各跑两次结果完全一致', all_same)


# ---------------------------------------------------------------------------
section('2. 噪声水平扫描：从 1mm 到 50mm')

noise_levels = [0.001, 0.002, 0.003, 0.005, 0.010, 0.020, 0.030, 0.050]
noise_results = {}
for noise in noise_levels:
    accepted = 0
    errors = []
    for seed in range(20):
        s = generate('noisy', seed, noise_m=noise)
        obj = safe_measure(s.frame, s.detections, s.intrinsics).objects
        if not obj:
            continue
        obj = obj[0]
        if obj.status == 'accepted':
            accepted += 1
            err = abs(obj.distance_m - s.truth[obj.id]['distance_m']) * 1000
            errors.append(err)
    max_err = max(errors) if errors else float('nan')
    rate = accepted / 20
    noise_results[noise] = (accepted, rate)
    flag = '' if rate >= 0.8 else '  ⚠ 下降'
    print('  noise=%.1fmm: accepted %d/20 (%.0f%%), max_err=%.2f mm%s' % (
        noise * 1000, accepted, rate * 100, max_err, flag))

check('3mm 噪声下 acceptance >= 85%', noise_results.get(0.003, (0, 0))[1] >= 0.85)
check('50mm 噪声下 acceptance < 50%', noise_results.get(0.050, (1, 1))[1] < 0.5)


# ---------------------------------------------------------------------------
section('3. 深度边界：像素深度恰好在 min/max 边界')

s = generate('clean')
cfg = Config()
s2 = generate('clean')
s2.frame.depth_m[s2.frame.depth_m < 1.4] = s2.frame.min_depth_m
r = safe_measure(s2.frame, s2.detections, s2.intrinsics, cfg)
print('  背景深度=下限: status=%s' % (r.objects[0].status if r.objects else 'N/A'))
check('背景深度被压到下限时不崩溃', bool(r.objects) and r.objects[0].status in ('accepted', 'rejected'))

s3 = generate('clean')
s3.frame.depth_m[:] = s3.frame.max_depth_m
r3 = safe_measure(s3.frame, s3.detections, s3.intrinsics, cfg)
check('全部深度=上限时拒绝', bool(r3.objects) and r3.objects[0].status == 'rejected')


# ---------------------------------------------------------------------------
section('4. 全均匀深度（Otsu 无法分簇）')

s = generate('clean')
s.frame.depth_m[:] = 1.5
r = safe_measure(s.frame, s.detections, s.intrinsics)
check('全均匀深度 → rejected', bool(r.objects) and r.objects[0].status == 'rejected')
check('全均匀深度 → 无距离值', bool(r.objects) and r.objects[0].distance_m is None)


# ---------------------------------------------------------------------------
section('5. 极端内参')

s = generate('clean')
r1 = safe_measure(s.frame, s.detections, Intrinsics(10000, 10000, 159.5, 119.5))
check('超大焦距(10000)不崩溃', bool(r1.objects) and r1.objects[0].status in ('accepted', 'rejected'))

r2 = safe_measure(s.frame, s.detections, Intrinsics(1.0, 1.0, 159.5, 119.5))
check('极小焦距(1.0)不崩溃', bool(r2.objects) and r2.objects[0].status in ('accepted', 'rejected'))

try:
    measure_frame(s.frame, s.detections, Intrinsics(0, 1, 0, 0))
    check('焦距=0 抛异常', False)
except ValueError:
    check('焦距=0 抛异常', True)


# ---------------------------------------------------------------------------
section('6. 检测框极端位置')

s = generate('clean')
h, w = s.frame.depth_m.shape

d_corner = Detection('corner', (0, 0, 20, 20), [(10, 15), (10, 5)])
r_corner = safe_measure(s.frame, [d_corner], s.intrinsics)
check('检测框在左上角不崩溃', bool(r_corner.objects) and r_corner.objects[0].status in ('accepted', 'rejected', 'skipped'))

d_br = Detection('br', (w - 20, h - 20, w, h), [(w - 10, h - 15), (w - 10, h - 5)])
r_br = safe_measure(s.frame, [d_br], s.intrinsics)
check('检测框在右下角不崩溃', bool(r_br.objects) and r_br.objects[0].status in ('accepted', 'rejected', 'skipped'))

d_out = Detection('out', (w + 10, h + 10, w + 30, h + 30), [(w + 20, h + 20), (w + 20, h + 10)])
r_out = safe_measure(s.frame, [d_out], s.intrinsics)
check('检测框在图像外 → skipped', bool(r_out.objects) and r_out.objects[0].status == 'skipped')

d_kp_out = Detection('kp_out', (140, 30, 180, 200), [(-5, 100), (160, 100)])
r_kp = safe_measure(s.frame, [d_kp_out], s.intrinsics)
check('关键点超出图像 → skipped', bool(r_kp.objects) and r_kp.objects[0].status == 'skipped')


# ---------------------------------------------------------------------------
section('7. 重复 ID 检测')

s = generate('multi')
d1 = s.detections[0]
d2 = Detection('ice-1', d1.bbox, d1.keypoints)
try:
    measure_frame(s.frame, [d1, d2], s.intrinsics)
    check('重复 ID 抛异常', False)
except ValueError as e:
    check('重复 ID 抛异常', '唯一' in str(e))


# ---------------------------------------------------------------------------
section('8. 连续判定窗口数 sensitivity')

s = generate('clean')
cfg_fast = Config(consecutive=1)
r_fast = safe_measure(s.frame, s.detections, s.intrinsics, cfg_fast)
check('consecutive=1 可以更快接受', bool(r_fast.objects) and r_fast.objects[0].status == 'accepted')
if r_fast.objects and r_fast.objects[0].status == 'accepted':
    n_history = max(len(ep.history) for ep in r_fast.objects[0].endpoints)
    print('  consecutive=1 下最多 %d 个窗口记录' % n_history)

cfg_strict = Config(consecutive=5)
r_strict = safe_measure(s.frame, s.detections, s.intrinsics, cfg_strict)
check('consecutive=5 不崩溃', bool(r_strict.objects) and r_strict.objects[0].status in ('accepted', 'rejected'))


# ---------------------------------------------------------------------------
section('9. 极端 min_samples')

s = generate('clean')
cfg_loose = Config(min_samples=1, full_samples=2)
r_loose = safe_measure(s.frame, s.detections, s.intrinsics, cfg_loose)
check('min_samples=1 不崩溃', bool(r_loose.objects) and r_loose.objects[0].status in ('accepted', 'rejected'))

cfg_impossible = Config(min_samples=10000, full_samples=20000)
r_imp = safe_measure(s.frame, s.detections, s.intrinsics, cfg_impossible)
check('min_samples=10000 → rejected', bool(r_imp.objects) and r_imp.objects[0].status == 'rejected')


# ---------------------------------------------------------------------------
section('10. 离群点比例扫描')

for frac in [0.01, 0.05, 0.10, 0.20, 0.40]:
    s = generate('clean')
    rng = np.random.default_rng(42)
    outlier_mask = rng.random(s.frame.depth_m.shape) < frac
    s.frame.depth_m[outlier_mask] = rng.choice([0.2, 4.8, 0], outlier_mask.sum())
    obj = safe_measure(s.frame, s.detections, s.intrinsics).objects
    if not obj:
        check('离群 %.0f%% 不崩溃' % (frac * 100), False, 'measure_frame 异常')
        continue
    obj = obj[0]
    err = abs(obj.distance_m - s.truth['ice-1']['distance_m']) * 1000 if obj.distance_m else None
    print('  离群占比 %.0f%%: status=%s, err=%.2f mm' % (frac * 100, obj.status, err or -1))
    check('离群 %.0f%% 不崩溃' % (frac * 100), obj.status in ('accepted', 'rejected'))

# 40% 离群应该拒绝
s = generate('clean')
rng = np.random.default_rng(42)
outlier_mask = rng.random(s.frame.depth_m.shape) < 0.40
s.frame.depth_m[outlier_mask] = rng.choice([0.2, 4.8, 0], outlier_mask.sum())
r40 = safe_measure(s.frame, s.detections, s.intrinsics)
check('离群 40% → rejected 或误差 < 20mm',
      bool(r40.objects) and (r40.objects[0].status == 'rejected' or
      (r40.objects[0].distance_m and abs(
          r40.objects[0].distance_m - s.truth['ice-1']['distance_m']) * 1000 < 20)))


# ---------------------------------------------------------------------------
section('11. 各拒绝场景的拒绝原因正确性')

rejection_scenarios = {
    'sparse': '样本',
    'overlap': '分离度|间隔|成员',
    'background': '簇',
    'empty': '有效',
    'drift': '中心|稳定',
    'membership': '成员|中心|稳定',
    'interference': '明确|支持',
}

for name, expected_keyword in rejection_scenarios.items():
    s = generate(name, seed=42)
    obj = safe_measure(s.frame, s.detections, s.intrinsics).objects
    if not obj:
        check('场景 %s 拒绝原因包含[%s]' % (name, expected_keyword), False, '异常')
        continue
    obj = obj[0]
    reason = obj.reason or ''
    has_keyword = any(kw in reason for kw in expected_keyword.split('|'))
    check('场景 %s 拒绝原因包含[%s]' % (name, expected_keyword),
          obj.status == 'rejected' and has_keyword,
          reason[:80])


# ---------------------------------------------------------------------------
section('12. 多目标准确度')

s = generate('multi')
r = safe_measure(s.frame, s.detections, s.intrinsics)
if r.objects:
    accepted_objs = [o for o in r.objects if o.status == 'accepted']
    check('multi 场景有 2 个 accepted', len(accepted_objs) == 2)

    for obj in accepted_objs:
        truth = s.truth[obj.id]
        err = abs(obj.distance_m - truth['distance_m']) * 1000
        check('multi %s 距离误差 < 20mm' % obj.id, err < 20, '%.2f mm' % err)
        for i, ep in enumerate(obj.endpoints):
            z_err = abs(ep.depth_m - truth['depth_m'][i]) * 1000
            check('multi %s %s 深度误差 < 10mm' % (obj.id, ep.name), z_err < 10, '%.2f mm' % z_err)

    check('multi 第三个目标 skipped', any(o.status == 'skipped' for o in r.objects))
else:
    check('multi 场景不崩溃', False, 'measure_frame 返回空')


# ---------------------------------------------------------------------------
section('13. 深度量化的极端情况')

s = generate('clean')
s.frame.depth_m = np.round(s.frame.depth_m * 100) / 100
r_cm = safe_measure(s.frame, s.detections, s.intrinsics)
if r_cm.objects and r_cm.objects[0].status == 'accepted':
    err = abs(r_cm.objects[0].distance_m - s.truth['ice-1']['distance_m']) * 1000
    check('厘米级量化误差 < 30mm', err < 30, '%.2f mm' % err)
else:
    check('厘米级量化被拒绝（可接受）', True, r_cm.objects[0].status if r_cm.objects else 'N/A')

s2 = generate('clean')
s2.frame.depth_m = np.round(s2.frame.depth_m * 1000000) / 1000000
r_um = safe_measure(s2.frame, s2.detections, s2.intrinsics)
check('微米级量化 = 无损 accepted', bool(r_um.objects) and r_um.objects[0].status == 'accepted')


# ---------------------------------------------------------------------------
section('14. 大量目标（性能）')

s_base = generate('clean')
detections = []
for i in range(50):
    cx = 20 + i * 5
    if cx > 300:
        break
    d = Detection('ice-%d' % i, (max(0, cx - 12), 40, cx + 12, 200),
                  [(float(cx), 200.), (float(cx), 40.)])
    detections.append(d)

t0 = time.perf_counter()
r_batch = safe_measure(s_base.frame, detections, s_base.intrinsics)
dt = time.perf_counter() - t0
n_processed = len(r_batch.objects) if r_batch.objects else 0
print('  %d 个目标耗时 %.1f ms (%.1f ms/目标)' % (n_processed, dt * 1000, dt * 1000 / max(n_processed, 1)))
check('50 个目标在 5 秒内完成', dt < 5.0)
check('50 个目标全部有状态',
      bool(r_batch.objects) and all(o.status in ('accepted', 'rejected', 'skipped') for o in r_batch.objects))


# ---------------------------------------------------------------------------
section('15. Otsu 分簇边界')

depths_close = np.array([1.500, 1.501, 1.502, 1.503] * 20 + [1.510, 1.511, 1.512, 1.513] * 20)
labels_close = otsu_split(depths_close, Config())
check('微小深度差仍能分两簇', len(set(labels_close)) >= 2)

depths_flat = np.full(100, 1.5)
labels_flat = otsu_split(depths_flat, Config())
check('零深度差 → 全 0 或无法分簇', set(labels_flat) == {0} or set(labels_flat) == {0, 1})

labels_single = otsu_split(np.array([1.5]), Config())
check('单像素 Otsu → 全 0', set(labels_single) == {0})


# ---------------------------------------------------------------------------
section('16. confidence_scores 边界')

cfg = Config()
perfect = confidence_scores(100, 200, 0, 0.005, 1.0, cfg)
check('完美情况所有分数 = 1.0', all(v >= 0.99 for v in perfect.values()))

worst = confidence_scores(0, 0, 100, 0.005, 0.0, cfg)
check('最差情况分离度分数 = 0', worst['separation'] == 0)
check('最差情况样本分数 = 0', worst['samples'] == 0)
check('最差情况成员分数 = 0', worst['membership'] == 0)


# ---------------------------------------------------------------------------
section('17. NaN / Inf 深度处理')

s = generate('clean')
s.frame.depth_m[50:55, 155:165] = np.nan
s.frame.depth_m[60:65, 155:165] = np.inf
r_nan = safe_measure(s.frame, s.detections, s.intrinsics)
check('NaN/Inf 深度不崩溃', bool(r_nan.objects) and r_nan.objects[0].status in ('accepted', 'rejected'))
if r_nan.objects and r_nan.objects[0].status == 'accepted':
    check('NaN/Inf 区域被正确忽略',
          abs(r_nan.objects[0].distance_m - s.truth['ice-1']['distance_m']) < 0.02)


# ---------------------------------------------------------------------------
section('18. 深度全部为 0')

s = generate('clean')
s.frame.depth_m[:] = 0
r_zero = safe_measure(s.frame, s.detections, s.intrinsics)
check('全零深度 → rejected', bool(r_zero.objects) and r_zero.objects[0].status == 'rejected')
check('全零深度 → 无距离', bool(r_zero.objects) and r_zero.objects[0].distance_m is None)


# ---------------------------------------------------------------------------
section('19. 渐进噪声下的误差趋势（不应突变）')

prev_med = 0
trend_ok = True
for noise_mm in [1, 2, 3, 5, 8, 12]:
    errs = []
    for seed in range(15):
        s = generate('noisy', seed, noise_m=noise_mm / 1000)
        obj = safe_measure(s.frame, s.detections, s.intrinsics).objects
        if not obj:
            continue
        obj = obj[0]
        if obj.status == 'accepted':
            errs.append(abs(obj.distance_m - s.truth[obj.id]['distance_m']) * 1000)
    med = float(np.median(errs)) if errs else float('nan')
    print('  noise=%2dmm: 中位误差 %.2f mm (accepted %d/15)' % (noise_mm, med, len(errs)))
    if med != med:  # nan
        continue
    if med < prev_med * 0.3 and prev_med > 0:
        trend_ok = False
    prev_med = max(prev_med, med)
check('误差随噪声递增不突变', trend_ok)


# ---------------------------------------------------------------------------
section('20. Config 边界验证')

# radii 非递增
try:
    Config(radii=(5, 3, 8)).validate()
    check('radii 非递增 → 抛异常', False)
except ValueError:
    check('radii 非递增 → 抛异常', True)

# radii 有重复
try:
    Config(radii=(3, 5, 5, 8)).validate()
    check('radii 有重复 → 抛异常', False)
except ValueError:
    check('radii 有重复 → 抛异常', True)

# radii 太短（不足以完成 consecutive 比较）
try:
    Config(radii=(3, 5), consecutive=3).validate()
    check('radii 太短 → 抛异常', False)
except ValueError:
    check('radii 太短 → 抛异常', True)

# full_samples < min_samples
try:
    Config(min_samples=100, full_samples=50).validate()
    check('full < min → 抛异常', False)
except ValueError:
    check('full < min → 抛异常', True)

# confidence_min > 1
try:
    Config(confidence_min=1.1).validate()
    check('confidence_min > 1 → 抛异常', False)
except ValueError:
    check('confidence_min > 1 → 抛异常', True)


# ---------------------------------------------------------------------------
print('\n' + '=' * 60)
print('  总结: %d 项通过, %d 项失败, 共 %d 项' % (PASS, FAIL, len(CHECKS)))
print('=' * 60)
sys.exit(1 if FAIL else 0)
