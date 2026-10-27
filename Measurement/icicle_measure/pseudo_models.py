"""Optional CPU inference subprocess. The main GUI never imports torch/timm."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
from .io import read_image, read_json, write_json, load_detections
from .pseudo_depth import ROOT, MODEL_DIR, MODEL_PYTHON, PseudoConfig, file_digest, letterbox
from .pseudo_geometry import boundary_evidence, matches_prompts


def inference_key(rgb_path, labels_path, config, prompts):
    lock = MODEL_DIR / 'models.lock.json'
    if not MODEL_PYTHON.exists() or not lock.exists():
        raise RuntimeError('可选模型环境尚未安装；请运行 setup_pseudo_depth.py。原有离线测距不受影响。')
    manifest = read_json(lock)
    for weight in manifest['weights'].values():
        weight_path = ROOT / weight['file']
        if not weight_path.exists() or file_digest(weight_path) != weight['sha256']:
            raise RuntimeError('模型权重缺失或 SHA256 校验不符：' + str(weight_path))
    data = {'rgb_sha256': file_digest(rgb_path), 'labels_sha256': file_digest(labels_path),
            'model_lock_sha256': file_digest(lock), 'width': config.width, 'height': config.height,
            'model_long_edge': config.model_long_edge, 'prompts': prompts, 'adapter_version': 5}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), data


def run_inference(rgb_path, labels_path, config=None, prompts=None, progress=lambda value: None,
                  cancel=lambda: False, reuse=None, changed_ids=None):
    config, prompts = config or PseudoConfig(), prompts or {}
    config.validate()
    key, identity = inference_key(rgb_path, labels_path, config, prompts)
    cache = ROOT / 'outputs' / 'pseudo_model_cache'
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / (key + '.npz')
    if output.exists() and output.with_suffix('.json').exists():
        if cancel():
            raise InterruptedError('用户取消')
        progress({'stage': 'cache', 'message': '使用已校验来源的模型缓存'})
        return output
    request = dict(identity, rgb=str(rgb_path), labels=str(labels_path), output=str(output),
                   config=asdict(config), reuse=str(reuse) if reuse else None, changed_ids=changed_ids)
    request_path = cache / (key + '.request.json')
    write_json(request_path, request)
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', PYTHONNOUSERSITE='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    events = queue.Queue()
    log = output.with_suffix('.log')
    with log.open('w', encoding='utf-8') as stderr:
        process = subprocess.Popen([str(MODEL_PYTHON), '-u', '-m', 'icicle_measure.pseudo_models',
                                    '--request', str(request_path)], cwd=str(ROOT), env=env,
                                    stdout=subprocess.PIPE, stderr=stderr, text=True, encoding='utf-8',
                                    errors='replace', creationflags=creationflags)
        def read_output():
            for line in process.stdout:
                try:
                    events.put(json.loads(line))
                except (ValueError, TypeError):
                    pass
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while process.poll() is None or not events.empty():
                if cancel():
                    process.terminate()
                    process.wait(timeout=10)
                    raise InterruptedError('用户取消模型推理；已终止本任务的子进程')
                try:
                    progress(events.get(timeout=.1))
                except queue.Empty:
                    pass
            reader.join(timeout=2)
            if process.returncode:
                detail = log.read_text(encoding='utf-8', errors='replace')[-2500:]
                raise RuntimeError('模型推理失败；日志 %s\n%s' % (log, detail))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            process.stdout.close()
    if not output.exists() or not output.with_suffix('.json').exists():
        raise RuntimeError('模型未产生完整结果；请查看 ' + str(log))
    return output


def _emit(stage, message, **kwargs):
    print(json.dumps(dict(stage=stage, message=message, **kwargs), ensure_ascii=False), flush=True)


def _fit_mask(mask, bbox, shape):
    h, w = shape
    x1, y1, x2, y2 = bbox
    spatial = np.zeros(shape, bool)
    spatial[max(0, int(y1)):min(h, int(np.ceil(y2)) + 1),
            max(0, int(x1)):min(w, int(np.ceil(x2)) + 1)] = True
    original_area = max(int(mask.sum()), 1)
    if np.count_nonzero(mask & spatial) / original_area < .65:
        return np.zeros_like(mask), {'accepted': False, 'reason': '掩膜大部分位于所属检测框外', 'pixels': original_area}
    mask = mask & spatial
    box_area = max((x2 - x1) * (y2 - y1), 1)
    area = int(mask.sum())
    if area < 8:
        return np.zeros_like(mask), {'accepted': False, 'reason': '模型掩膜样本不足', 'pixels': area}
    if area / box_area > .94:
        return np.zeros_like(mask), {'accepted': False, 'reason': '候选近乎填满检测框，需人工确认', 'pixels': area}
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == biggest
    return mask, {'accepted': True, 'reason': '模型分割候选，非标注真值', 'pixels': int(mask.sum())}


def infer_request(request_path):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    request = read_json(request_path)
    cfg = PseudoConfig(**request['config']); cfg.validate()
    rgb = cv2.cvtColor(read_image(request['rgb'], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    detections = load_detections(request['labels'], w, h)
    _, transformed, valid, transform = letterbox(rgb, detections, cfg.width, cfg.height)
    left, top = transform['offset']; rw, rh = transform['resized_size']
    sx, sy = transform['scale_xy']; tx, ty = transform['pixel_center_translation']
    begin = time.perf_counter()
    sys.path.insert(0, str(MODEL_DIR / 'Depth-Anything-V2'))
    sys.path.insert(0, str(MODEL_DIR / 'MobileSAM'))
    output = Path(request['output'])
    reuse = Path(request['reuse']) if request.get('reuse') else None
    reused = False
    if reuse and reuse.exists():
        with np.load(str(reuse), allow_pickle=False) as data:
            relative, masks = data['relative'].copy(), data['masks'].copy()
        previous = read_json(reuse.with_suffix('.json'))
        segmentation = previous['segmentation']
        reused = True
    else:
        from depth_anything_v2.dpt import DepthAnythingV2
        _emit('load_depth', '加载 Depth Anything V2 Small（CPU）')
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(str(MODEL_DIR / 'depth_anything_v2_vits.pth'), map_location='cpu', weights_only=True))
        model.eval()
        # Bound the LONG edge rather than the official demo's short edge, so
        # ultra-wide/portrait photos do not explode CPU inference cost.
        nh = max(14, int(round(h * cfg.model_long_edge / max(h, w) / 14)) * 14)
        nw = max(14, int(round(w * cfg.model_long_edge / max(h, w) / 14)) * 14)
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
        small = (small - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        tensor = torch.from_numpy(small.transpose(2, 0, 1)[None]).contiguous()
        _emit('depth', '计算全图相对深度', model_size=[nw, nh])
        with torch.inference_mode():
            relative_small = model(tensor)[0].cpu().numpy()
        relative = np.zeros((cfg.height, cfg.width), np.float32)
        relative[top:top + rh, left:left + rw] = cv2.resize(relative_small, (rw, rh), interpolation=cv2.INTER_CUBIC)
        del model, tensor
        masks = np.zeros((len(detections), cfg.height, cfg.width), bool)
        segmentation = [{} for _ in detections]
    from mobile_sam import sam_model_registry, SamPredictor
    _emit('load_masks', '加载 MobileSAM（CPU）')
    mobile = sam_model_registry['vit_t'](checkpoint=str(MODEL_DIR / 'MobileSAM' / 'weights' / 'mobile_sam.pt'))
    mobile.eval()
    predictor = SamPredictor(mobile)
    changed = set(request.get('changed_ids') or [])
    for index, (d, td) in enumerate(zip(detections, transformed)):
        if reused and d.id not in changed:
            continue
        _emit('mask', '细分冰凌 %s（%d/%d）' % (d.id, index + 1, len(detections)), index=index, total=len(detections))
        x1, y1, x2, y2 = d.bbox
        padx, pady = max(6., (x2 - x1) * .18), max(6., (y2 - y1) * .08)
        ax, ay = max(0, int(x1 - padx)), max(0, int(y1 - pady))
        bx, by = min(w, int(np.ceil(x2 + padx))), min(h, int(np.ceil(y2 + pady)))
        if bx <= ax or by <= ay:
            segmentation[index] = {'accepted': False, 'reason': '检测框位于图像外'}
            continue
        predictor.set_image(rgb[ay:by, ax:bx])
        box = np.array([max(0, x1 - ax), max(0, y1 - ay), min(bx - ax - 1, x2 - ax), min(by - ay - 1, y2 - ay)])
        prompts = request['prompts'].get(d.id, [])
        coords, labels = [], []
        for point in prompts:
            # UI points are output-frame pixels. Invert the saved transform.
            px, py = (point['x'] - tx) / sx - ax, (point['y'] - ty) / sy - ay
            if 0 <= px < bx - ax and 0 <= py < by - ay:
                coords.append([px, py]); labels.append(int(point['positive']))
        with torch.inference_mode():
            candidates, scores, _ = predictor.predict(
                point_coords=np.asarray(coords, np.float32) if coords else None,
                point_labels=np.asarray(labels, np.int32) if labels else None,
                box=box, multimask_output=True)
        ranked = np.argsort(scores)[::-1]
        chosen = np.zeros((by - ay, bx - ax), bool)
        diagnosis = {'accepted': False, 'reason': '所有分割候选均需人工修正'}
        chosen_score = float(scores[ranked[0]])
        # RGB boundaries are a weak tie-breaker, never a substitute for manual review.
        options = []
        for candidate in ranked:
            candidate_mask, info = _fit_mask(candidates[candidate], box, chosen.shape)
            if info['accepted']:
                if not matches_prompts(candidate_mask, coords, labels):
                    diagnosis['reason'] = '分割候选不满足人工正负提示；请修正掩膜，未强行填充'
                    continue
                edge = boundary_evidence(rgb[ay:by, ax:bx], candidate_mask)
                options.append((float(scores[candidate]) + .12 * edge, candidate_mask, info, float(scores[candidate]), edge))
        if options:
            options.sort(key=lambda row: row[0], reverse=True)
            _, chosen, diagnosis, chosen_score, edge = options[0]
            diagnosis['boundary_evidence'] = edge
            diagnosis['candidate_rank_scores'] = [row[0] for row in options]
            ambiguous = False
            if len(options) > 1:
                other = options[1][1]
                iou = np.count_nonzero(chosen & other) / max(np.count_nonzero(chosen | other), 1)
                ambiguous = bool(options[0][0] - options[1][0] < .03 and iou < .75)
            diagnosis['needs_review'] = ambiguous or edge < .15
            if diagnosis['needs_review']:
                diagnosis['reason'] = '候选边界支持不足或候选差异大，请核对原 RGB 并修正掩膜'
        full = np.zeros((h, w), np.uint8)
        full[ay:by, ax:bx] = chosen.astype(np.uint8)
        mask = np.zeros((cfg.height, cfg.width), bool)
        mask[top:top + rh, left:left + rw] = cv2.resize(full, (rw, rh), interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
        masks[index] = mask & valid
        diagnosis.update(id=d.id, model_mask_score=chosen_score, score_is_probability=False,
                         crop=[ax, ay, bx, by], prompts=prompts)
        segmentation[index] = diagnosis
    metadata = {'model_lock': read_json(MODEL_DIR / 'models.lock.json'),
                'rgb_sha256': request['rgb_sha256'], 'labels_sha256': request['labels_sha256'],
                'segmentation': segmentation, 'seconds': time.perf_counter() - begin,
                'device': 'cpu', 'threads': 4, 'depth_reused': reused,
                'adapter_version': request['adapter_version'],
                'relative_depth_convention': 'larger_is_nearer_no_metric_scale'}
    # Commit the cache only after both models have completed successfully.
    temporary = output.with_suffix('.temporary.npz')
    np.savez_compressed(str(temporary), relative=relative.astype(np.float32), masks=masks)
    write_json(output.with_suffix('.json'), metadata)
    temporary.replace(output)
    _emit('done', '模型推理完成', seconds=metadata['seconds'], output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    infer_request(parser.parse_args().request)
