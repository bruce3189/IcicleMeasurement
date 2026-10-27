import cv2
import numpy as np


def verified_distance(obj):
    """Only the core's accepted pair supplies an exportable length."""
    return bool(obj is not None and obj.status == 'accepted' and obj.distance_m is not None
                and np.isfinite(obj.distance_m) and obj.distance_m >= 0
                and len(obj.endpoints) == 2 and all(ep.status == 'accepted' and ep.xyz_m is not None
                and np.isfinite(ep.xyz_m).all() for ep in obj.endpoints))


def distance_summary(obj, pseudo=False, variant='', pending='待计算'):
    prefix = '基于伪深度的估计欧氏距离' if pseudo else '端点估计欧氏距离'
    source = {'complete': '完整表面', 'observed': '观测深度'}.get(variant, '')
    prefix += (' · ' + source) if source else ''
    if obj is None:
        return prefix + '：' + pending
    if verified_distance(obj):
        return prefix + ' · #' + obj.id + '：%.2f mm' % (obj.distance_m * 1000)
    missing = ['尖端' if i == 0 else '根部' for i in range(2)
               if i >= len(obj.endpoints) or obj.endpoints[i].status != 'accepted' or obj.endpoints[i].xyz_m is None]
    return prefix + ' · #' + obj.id + '：不可测（' + (' / '.join(missing) + '未通过' if missing else obj.reason or '目标未通过') + '）'


def draw_selected_distance(image, detection, obj=None, pending='pending'):
    """Small, clipped label beside the original endpoint segment."""
    h, w = image.shape[:2]
    points = [p for p in detection.keypoints if p is not None and len(p) == 2
              and np.isfinite(p).all() and 0 <= p[0] < w and 0 <= p[1] < h]
    if len(points) == 2:
        a, b = [tuple(int(round(x)) for x in p) for p in points]
        cv2.line(image, a, b, (255, 215, 80), 1, cv2.LINE_AA)
        anchor = ((a[0]+b[0])//2+6, (a[1]+b[1])//2)
    else:
        anchor = (int(detection.bbox[0])+4, int(detection.bbox[1])+18)
    text = ('L = %.2f mm' % (obj.distance_m*1000) if verified_distance(obj)
            else 'L: unavailable' if obj is not None else 'L: ' + pending)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, .38
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x = int(np.clip(anchor[0], 2, max(2, w-tw-4)))
    y = int(np.clip(anchor[1], th+3, max(th+3, h-baseline-3)))
    cv2.rectangle(image, (x-2,y-th-2), (min(w-1,x+tw+2),min(h-1,y+baseline+2)), (22,35,45), -1)
    cv2.putText(image, text, (x,y), font, scale, (255,225,130), 1, cv2.LINE_AA)
    return image


def depth_rgb(frame):
    z = frame.depth_m
    valid = np.isfinite(z) & (z > 0) & (z >= frame.min_depth_m) & (z <= frame.max_depth_m)
    gray = np.zeros(z.shape, np.uint8)
    if valid.any():
        lo, hi = frame.metadata.get('display_depth_range_m', np.percentile(z[valid], [1, 99]))
        gray[valid] = np.clip((z[valid] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def overlay(frame, detections, result=None, depth=False, boxes=True, points=True, distances=True,
            selected=None, clusters=True, distance_scope='all'):
    image = depth_rgb(frame) if depth else frame.rgb.copy()
    mapping = {obj.id: obj for obj in result.objects} if result else {}
    if selected and result:
        obj_id, index = selected
        obj = mapping.get(obj_id)
        if obj and index < len(obj.endpoints) and obj.endpoints[index].history:
            record = obj.endpoints[index].history[-1]
            if clusters:
                pixels = image.reshape(-1, 3)
                for ids, color in ((record.far_ids, (70, 110, 245)),
                                   (record.near_ids, (250, 182, 55)),
                                   (record.target_ids, (45, 230, 125))):
                    if ids:
                        pixels[ids] = (0.45 * pixels[ids] + 0.55 * np.array(color)).astype(np.uint8)
            x1, y1, x2, y2 = record.bounds
            cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (255, 215, 80), 1)
    for idx, detection in enumerate(detections):
        obj = mapping.get(detection.id)
        color = (55, 220, 145) if obj and obj.status == 'accepted' else (255, 165, 70)
        if len(detection.bbox) != 4 or not np.all(np.isfinite(detection.bbox)):
            continue
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(round(np.clip(v, -2 * max(h, w), 2 * max(h, w)))) for v in detection.bbox]
        if boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        indexed_points = [(j, tuple(int(round(v)) for v in p)) for j, p in enumerate(detection.keypoints)
                          if p is not None and len(p) == 2 and np.all(np.isfinite(p))
                          and 0 <= p[0] < w and 0 <= p[1] < h]
        valid_points = [p for _, p in indexed_points]
        if points:
            if len(valid_points) == 2:
                cv2.line(image, valid_points[0], valid_points[1], (245, 205, 60), 1, cv2.LINE_AA)
            for j, p in indexed_points:
                cv2.circle(image, p, 2, (255, 90, 85) if j == 0 else (70, 225, 240), -1)
        label = '#' + detection.id
        if distances and distance_scope == 'all' and verified_distance(obj):
            label += ' %.1f mm' % (obj.distance_m * 1000)
        if boxes or distances:
            cv2.putText(image, label, (max(0, x1), max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
    if distances and distance_scope == 'selected' and selected:
        detection = next((d for d in detections if d.id == selected[0]), None)
        if detection is not None:
            draw_selected_distance(image, detection, mapping.get(detection.id))
    if frame.metadata.get('is_synthetic'):
        text = 'SYNTHETIC RGB-D | ASSUMED SCALE / INTRINSICS | NOT CAMERA MEASUREMENTS'
        cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 26), (20, 30, 40), -1)
        cv2.putText(image, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 215, 100), 1, cv2.LINE_AA)
    return image
