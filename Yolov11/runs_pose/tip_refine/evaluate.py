"""固定预测口径，按真实二维距离评估；输出原始点，不用框推点掩盖误差。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

HOME = Path(__file__).resolve().parent
DATA = HOME/'data_snapshot'
ORIGINAL = Path('D:/yolov11/runs_pose/ice80_yolo26/weights/best.pt')
OLD_DETECTOR = Path('D:/yolov11/runs/detect/train10/weights/best.pt')


def box_iou(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(-1)
    return inter / np.maximum(1e-9, (a[:, 2:] - a[:, :2]).prod(-1)[:, None] +
                              (b[:, 2:] - b[:, :2]).prod(-1)[None] - inter)


def containing_boxes(boxes, points):
    """保留点坐标，取框与所属三点的包围矩形；不合并不同实例。"""
    output=boxes.copy()
    if len(output):
        output[:,:2]=np.minimum(output[:,:2],points[...,:2].min(1))
        output[:,2:]=np.maximum(output[:,2:],points[...,:2].max(1))
    return output


def match(boxes, predictions, threshold=.5):
    """与原复核一致：按IoU从高到低一对一匹配，并独立报告未匹配目标。"""
    matrix = box_iou(boxes, predictions)
    pairs, used_gt, used_pred = [], set(), set()
    for idx in np.argsort(-matrix.ravel()):
        gi, pi = np.unravel_index(idx, matrix.shape)
        if matrix[gi, pi] < threshold:
            break
        if gi not in used_gt and pi not in used_pred:
            pairs.append((int(gi), int(pi), float(matrix[gi, pi])))
            used_gt.add(gi)
            used_pred.add(pi)
    return pairs


def read_gt(image):
    with Image.open(image) as im:
        width, height = im.size
    # 数据目录不动：images/<split>/x.jpg -> labels/<split>/x.txt。
    label = image.parent.parent.parent / 'labels' / image.parent.name / (image.stem + '.txt')
    a = np.loadtxt(label, ndmin=2)
    if a.shape[1] != 14:
        raise ValueError(f'{label}: 需要14列YOLO Pose标签')
    boxes = a[:, 1:5].copy()
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    boxes *= [width, height, width, height]
    points = a[:, 5:].reshape(-1, 3, 3).copy()
    points[..., :2] *= [width, height]
    return boxes, points


def describe(values):
    a = np.asarray(values)
    if not len(a):
        return {'n': 0, 'median': None, 'mean': None, 'p90': None}
    return dict(n=len(a), median=float(np.median(a)), mean=float(a.mean()), p90=float(np.quantile(a, .9)))


def evaluate(weight, split='val', imgsz=1280, output=None, data=DATA, nms=True, box_adjust='raw'):
    torch.set_num_threads(2)
    model = YOLO(str(weight))
    rows, cache = [], []
    counts = dict(gt=0, predictions=0, matches=0, gt_visible_tip=0, tip_hits10=0,
                  tip_outside=0, predicted_tips=0)
    for ip in sorted((Path(data) / 'images' / split).glob('*.jpg')):
        boxes, points = read_gt(ip)
        r = model.predict(str(ip), imgsz=imgsz, conf=.25, iou=.7, nms=nms,
                          device='cpu', verbose=False)[0]
        pb = r.boxes.xyxy.cpu().numpy()
        pk = None if r.keypoints is None else r.keypoints.data.cpu().numpy()
        if box_adjust=='expand' and pk is not None:
            pb=containing_boxes(pb,pk)
        pairs = match(boxes, pb)
        counts['gt'] += len(boxes)
        counts['predictions'] += len(pb)
        counts['matches'] += len(pairs)
        counts['gt_visible_tip'] += int((points[:, 0, 2] > 0).sum())
        if pk is not None:
            xy = pk[:, 0, :2]
            inside = ((xy >= pb[:, :2]) & (xy <= pb[:, 2:])).all(1)
            counts['tip_outside'] += int((~inside).sum())
            counts['predicted_tips'] += len(pk)
        cache.append(dict(image=str(ip), gt_boxes=boxes.tolist(), gt_points=points.tolist(),
                          boxes=pb.tolist(), points=None if pk is None else pk.tolist(),
                          scores=r.boxes.conf.cpu().tolist()))
        for gi, pi, iou in pairs:
            h = float(boxes[gi, 3] - boxes[gi, 1])
            row = dict(image=ip.stem, gi=gi, pi=pi, height=h, iou=iou,
                       bottom_error=float(abs(pb[pi, 3] - boxes[gi, 3])))
            for k, name in enumerate(('tip', 'base_left', 'base_right')):
                row[name] = (float(np.linalg.norm(pk[pi, k, :2] - points[gi, k, :2]))
                             if pk is not None and points[gi, k, 2] > 0 else None)
            if row['tip'] is not None:
                counts['tip_hits10'] += int(row['tip'] < .1 * h)
            rows.append(row)
    if not cache:
        raise ValueError(f'{data}/images/{split} 没有jpg图片')
    summary = dict(weight=str(weight), split=split, imgsz=imgsz, conf=.25, nms=nms, nms_iou=.7,
                   box_adjust=box_adjust,
                   matching_iou=.5, images=len(cache), counts=counts,
                   missed=counts['gt']-counts['matches'], extra=counts['predictions']-counts['matches'],
                   iou=describe([r['iou'] for r in rows]),
                   bottom_error=describe([r['bottom_error'] for r in rows]))
    for name in ('tip', 'base_left', 'base_right'):
        eligible = [r for r in rows if r[name] is not None]
        summary[name] = describe([r[name] for r in eligible])
        for pct in (.05, .1):
            summary[name][f'pck{int(pct*100)}_matched'] = (
                float(np.mean([r[name] < pct*r['height'] for r in eligible])) if eligible else None)
    summary['tip_success10_all_gt'] = (counts['tip_hits10']/counts['gt_visible_tip']
                                       if counts['predicted_tips'] and counts['gt_visible_tip'] else None)
    summary['tip_outside_rate'] = (counts['tip_outside']/counts['predicted_tips']
                                   if counts['predicted_tips'] else None)
    summary['height_bins'] = {}
    for lo, hi in ((0, 700), (700, 1200), (1200, float('inf'))):
        group = [r for r in rows if lo <= r['height'] < hi and r['tip'] is not None]
        summary['height_bins'][str(lo)] = describe([r['tip'] for r in group])
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
        (output/'predictions.json').write_text(json.dumps(cache, ensure_ascii=False), encoding='utf8')
        with (output/'instances.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image','gi','pi','height','iou','bottom_error','tip','base_left','base_right'])
            writer.writeheader()
            writer.writerows(rows)
    return summary


def compare_images(before, after, output, limit=8):
    """同一张图三栏：人工标注、原模型、候选模型；每个实例显示编号。"""
    a = json.loads(Path(before).read_text(encoding='utf8'))
    b = {Path(x['image']).stem:x for x in json.loads(Path(after).read_text(encoding='utf8'))}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 28)
    # 先看最长目标；固定排序，避免只挑改善明显的图片。
    a.sort(key=lambda x: -max((p[3]-p[1] for p in x['gt_boxes']), default=0))
    for entry in a[:limit]:
        other = b[Path(entry['image']).stem]
        original = Image.open(entry['image']).convert('RGB')
        gt_boxes = entry['gt_boxes']
        top = int(max(0, min(x[1] for x in gt_boxes)-100))
        bot = int(min(original.height, max(x[3] for x in gt_boxes)+100))
        panels=[]
        for title, boxes, points, color in [
            ('人工标注', gt_boxes, entry['gt_points'], '#00c878'),
            ('原YOLO26', entry['boxes'], entry['points'], '#ff4040'),
            ('候选模型（原始点）', other['boxes'], other['points'], '#00bfff')]:
            im=original.copy(); draw=ImageDraw.Draw(im)
            for index, box in enumerate(boxes):
                draw.rectangle(box, outline=color, width=4)
                draw.text((box[0]+3, box[1]+3), str(index), fill=color, font=font)
                if points is not None:
                    for ki, (x,y,v) in enumerate(points[index]):
                        if v<=0:continue
                        radius=10 if ki==0 else 6
                        draw.ellipse((x-radius,y-radius,x+radius,y+radius), fill='#ffff00' if ki==0 else color)
            crop=im.crop((0,top,im.width,bot)); crop.thumbnail((600,1100))
            panel=Image.new('RGB',(620,crop.height+50),'#161616');panel.paste(crop,(10,50))
            ImageDraw.Draw(panel).text((10,8),title,font=font,fill='white');panels.append(panel)
        canvas=Image.new('RGB',(1860,max(x.height for x in panels)),'#161616')
        for i, panel in enumerate(panels):canvas.paste(panel,(i*620,0))
        canvas.save(out/(Path(entry['image']).stem+'.jpg'),quality=90)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,default=ORIGINAL)
    p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--box-adjust',choices=['raw','expand'],default='raw',help='expand仅扩展各自的框以包含预测点，另行报告')
    args=p.parse_args()
    print(json.dumps(evaluate(args.weights,args.split,args.imgsz,args.output,box_adjust=args.box_adjust),ensure_ascii=False,indent=2))
