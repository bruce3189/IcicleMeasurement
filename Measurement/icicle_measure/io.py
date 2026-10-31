"""File adapters. Bundles explicitly declare depth units, alignment and intrinsics."""
import csv
import json
import math
from pathlib import Path
from dataclasses import asdict
import cv2
import numpy as np
from .models import Frame, Detection, Intrinsics, Config, DEPTH_MIN_M, DEPTH_MAX_M
from .render import overlay


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    path = Path(path)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), flags)
    if image is None:
        raise ValueError('无法读取图像: ' + str(path))
    return image


def write_image(path, rgb):
    path = Path(path)
    ok, data = cv2.imencode(path.suffix, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError('无法编码图像: ' + str(path))
    data.tofile(str(path))


def load_detections(path, width, height, keypoint_min=0.0):
    """YOLO rows: cls cx cy bw bh tip_x tip_y tip_conf base_x base_y base_conf [box_conf].

    Confidence/visibility <= keypoint_min marks a missing keypoint. JSON points
    are pixel coordinates, nullable or {x,y,confidence}; no confidence is required.
    """
    path = Path(path)
    detections = []
    if path.suffix.lower() == '.json':
        data = read_json(path)
        rows = data['detections'] if isinstance(data, dict) else data
        for i, row in enumerate(rows):
            points = []
            for p in row.get('keypoints', []):
                if p is None:
                    points.append(None)
                elif isinstance(p, dict):
                    points.append((float(p['x']), float(p['y'])) if p.get('confidence', 1) > keypoint_min else None)
                else:
                    if len(p) not in (2, 3):
                        raise ValueError('JSON 关键点必须为 [x,y] 或 [x,y,confidence]')
                    points.append(tuple(map(float, p[:2])) if len(p) == 2 or p[2] > keypoint_min else None)
            detections.append(Detection(str(row.get('id', i + 1)), tuple(map(float, row['bbox'])), points))
        return detections
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        vals = list(map(float, line.split()))
        if len(vals) not in (11, 12) or not np.all(np.isfinite(vals)):
            raise ValueError('YOLO 第 %d 行应为 11 或 12 个有限数字' % line_no)
        _, cx, cy, bw, bh = vals[:5]
        if not all(0 <= v <= 1 for v in vals[1:5] + vals[5:7] + vals[8:10]) or bw <= 0 or bh <= 0:
            raise ValueError('YOLO 坐标必须归一化到 0..1，框尺寸必须大于零')
        box = ((cx - bw / 2) * width, (cy - bh / 2) * height,
               (cx + bw / 2) * width, (cy + bh / 2) * height)
        points = [(vals[k] * width, vals[k + 1] * height) if vals[k + 2] > keypoint_min else None for k in (5, 8)]
        detections.append(Detection(str(line_no), box, points))
    return detections


def load_files(rgb_path, depth_path, detection_path, metadata_path):
    meta = read_json(metadata_path)
    if not isinstance(meta, dict) or not isinstance(meta.get('source', {}), dict):
        raise ValueError('输入元数据和 source 须为 JSON 对象')
    assumed_pseudo = (meta.get('geometry_mode') == 'assumed_pinhole'
                      and meta.get('source', {}).get('input_kind') == 'pseudo_rgbd'
                      and meta.get('source', {}).get('is_synthetic') is True)
    if meta.get('aligned_to_rgb') is not True or (meta.get('undistorted') is not True and not assumed_pseudo):
        raise ValueError('元数据须声明 aligned_to_rgb=true 和 undistorted=true；本模块不代替配准与去畸变')
    if meta.get('depth_kind', 'z') != 'z':
        raise ValueError('当前仅支持光轴 Z 深度，不支持径向距离')
    unit = meta.get('depth_unit')
    explicit_scale = meta.get('depth_scale_m_per_unit')
    if explicit_scale is not None:
        if unit not in (None, 'raw'):
            raise ValueError('指定 depth_scale_m_per_unit 时 depth_unit 须为 raw 或省略，避免重复换算')
        if isinstance(explicit_scale, bool) or not isinstance(explicit_scale, (int, float)) or not math.isfinite(explicit_scale) or explicit_scale <= 0:
            raise ValueError('depth_scale_m_per_unit 必须为有限正数（米/原始单位）')
        scale = float(explicit_scale)
        unit = 'raw'
    elif unit in ('m', 'mm'):
        scale = .001 if unit == 'mm' else 1.0
    else:
        raise ValueError('depth_unit 须指定 m/mm；原始深度须指定 depth_scale_m_per_unit')
    rgb = cv2.cvtColor(read_image(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    depth_path = Path(depth_path)
    if depth_path.suffix.lower() not in ('.npy', '.png', '.tif', '.tiff'):
        raise ValueError('深度仅支持无损 NPY、PNG、TIFF，不能使用 JPEG 或伪彩色预览')
    depth = np.load(str(depth_path), allow_pickle=False) if depth_path.suffix.lower() == '.npy' else read_image(depth_path)
    if depth.ndim != 2 or depth.dtype.kind not in 'uif' or depth.dtype.itemsize < 2:
        raise ValueError('深度须为单通道 16 位以上整数或浮点数组；8 位/彩色预览图不可测距')
    depth = depth.astype(np.float64) * scale
    valid_range = meta.get('valid_range_m', [DEPTH_MIN_M, DEPTH_MAX_M])
    if not isinstance(valid_range, (list, tuple)) or len(valid_range) != 2:
        raise ValueError('valid_range_m 必须为 [最小米值, 最大米值]')
    frame = Frame(rgb, depth, *valid_range)
    frame.validate()
    expected_size = meta.get('intrinsics_image_size')
    if expected_size is not None and list(expected_size) != [rgb.shape[1], rgb.shape[0]]:
        raise ValueError('内参标定尺寸与当前图像不符；请提供该分辨率的内参')
    if not isinstance(meta.get('intrinsics'), dict) or set(meta['intrinsics']) != {'fx', 'fy', 'cx', 'cy'}:
        raise ValueError('intrinsics 须包含 fx、fy、cx、cy 四个数值')
    intrinsics = Intrinsics(**meta['intrinsics'])
    intrinsics.validate()
    frame.metadata = dict(meta.get('source', {}), input_kind=meta.get('source', {}).get('input_kind', 'offline'),
                          depth_unit=unit, depth_scale_m_per_unit=scale,
                          aligned_to_rgb=True, undistorted=meta.get('undistorted') is True,
                          geometry_mode=meta.get('geometry_mode', 'rectified_pinhole'),
                          rgb_path=str(rgb_path), depth_path=str(depth_path))
    detections = load_detections(detection_path, rgb.shape[1], rgb.shape[0], meta.get('keypoint_min', 0.0))
    return frame, detections, intrinsics


def depth_quality(frame):
    """Describe observed valid coverage without changing or filling any sample."""
    z = frame.depth_m
    valid = np.isfinite(z) & (z > 0) & (z >= frame.min_depth_m) & (z <= frame.max_depth_m)
    values = z[valid]
    return {'valid_pixels': int(valid.sum()), 'total_pixels': int(z.size),
            'valid_fraction': float(valid.mean()),
            'p01_m': float(np.percentile(values, 1)) if values.size else None,
            'p99_m': float(np.percentile(values, 99)) if values.size else None}


def load_bundle(manifest_path):
    path = Path(manifest_path)
    meta = read_json(path)
    return load_files(*(path.parent / meta[name] for name in ('rgb', 'depth', 'detections')), path)


def save_simulation(directory, simulation):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_image(root / 'rgb.png', simulation.frame.rgb)
    np.save(str(root / 'depth.npy'), simulation.frame.depth_m)
    write_json(root / 'detections.json', {'detections': [asdict(d) for d in simulation.detections]})
    write_json(root / 'input.json', {
        'rgb': 'rgb.png', 'depth': 'depth.npy', 'detections': 'detections.json',
        'depth_unit': 'm', 'depth_kind': 'z', 'aligned_to_rgb': True, 'undistorted': True,
        'valid_range_m': [simulation.frame.min_depth_m, simulation.frame.max_depth_m],
        'intrinsics': asdict(simulation.intrinsics), 'scenario': simulation.scenario, 'seed': simulation.seed,
        'source': simulation.frame.metadata,
    })
    write_json(root / 'truth.json', simulation.truth)
    return root / 'input.json'


def save_input_bundle(directory, frame, detections, intrinsics):
    """Save the exact metric frame used for inference, including provenance."""
    frame.validate()
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_image(root / 'rgb.png', frame.rgb)
    np.save(str(root / 'depth.npy'), frame.depth_m)
    write_json(root / 'detections.json', {'detections': [asdict(d) for d in detections]})
    write_json(root / 'input.json', {
        'rgb': 'rgb.png', 'depth': 'depth.npy', 'detections': 'detections.json',
        'depth_unit': 'm', 'depth_kind': 'z', 'aligned_to_rgb': True,
        'undistorted': frame.metadata.get('undistorted', True),
        'geometry_mode': frame.metadata.get('geometry_mode', 'rectified_pinhole'),
        'valid_range_m': [frame.min_depth_m, frame.max_depth_m],
        'intrinsics': asdict(intrinsics), 'intrinsics_image_size': [frame.rgb.shape[1], frame.rgb.shape[0]],
        'source': frame.metadata})
    return root / 'input.json'


def export_results(directory, frame, detections, result):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    simple = result.config.get('endpoint_depth_mode') == 'near_cluster_simple'
    if simple:
        # Keep the runtime/legacy schema intact; omit inactive fields only in new exports.
        inactive = ('gap_m', 'membership', 'score', 'scores', 'anchor', 'plane',
                    'growth', 'coverage', 'consecutive_passes', 'advisories')
        for obj in payload['objects']:
            for ep in obj['endpoints']:
                ep.pop('confidence', None)
                for record in ep['history']:
                    for key in inactive:
                        record.pop(key, None)
    write_json(root / 'measurement.json', payload)
    write_json(root / 'config.json', result.config)
    fields = ['id', 'status', 'reason', 'distance_m', 'endpoint', 'endpoint_status',
              'endpoint_reason', 'u_px', 'v_px', 'depth_m', 'x_m', 'y_m', 'z_m', 'confidence', 'radius_px',
              'input_kind', 'measurement_label', 'scale_source', 'intrinsics_source', 'depth_variant']
    if simple:
        fields.remove('confidence')
    with (root / 'measurement.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for obj in result.objects:
            common = {'id': obj.id, 'status': obj.status, 'reason': obj.reason, 'distance_m': obj.distance_m}
            common.update({key: frame.metadata.get(key, '') for key in fields[-5:]})
            if not obj.endpoints:
                writer.writerow(common)
            for ep in obj.endpoints:
                xyz = ep.xyz_m or (None, None, None)
                pixel = ep.pixel or (None, None)
                row = dict(common, endpoint=ep.name, endpoint_status=ep.status,
                                     endpoint_reason=ep.reason, u_px=pixel[0], v_px=pixel[1], depth_m=ep.depth_m,
                                     x_m=xyz[0], y_m=xyz[1], z_m=xyz[2], confidence=ep.confidence,
                                     radius_px=ep.history[-1].radius if ep.history else None)
                if simple:
                    row.pop('confidence')
                writer.writerow(row)
    write_image(root / 'overlay.png', overlay(frame, detections, result))
    return root
