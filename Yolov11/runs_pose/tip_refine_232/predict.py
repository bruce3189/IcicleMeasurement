"""逐图推理。保留原始坐标；可将展示/导出点截回自己的框内，并标记人工复核。"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from evaluate import ORIGINAL, containing_boxes


def contained_points(points, boxes):
    """仅改变xy，不改变置信度、实例顺序或框；返回哪些点被移动。"""
    result=points.copy()
    result[..., :2]=np.clip(result[..., :2],boxes[:,None,:2],boxes[:,None,2:])
    changed=np.any(np.abs(result[..., :2]-points[..., :2])>1e-6,axis=-1)
    return result,changed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='单张图或图片文件夹')
    p.add_argument('--weights',type=Path,default=ORIGINAL,help='明确指定想比较的候选权重')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--containment',choices=['clip','expand','raw'],default='clip',help='clip移动点；expand保留点并扩展各自框；raw不施加约束')
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.output.exists():
        raise FileExistsError('输出目录已存在，请更换--output，避免覆盖比较结果')
    (a.output/'labels').mkdir(parents=True)
    (a.output/'raw').mkdir()
    m=YOLO(str(a.weights))
    count=flagged=0
    for r in m.predict(a.source,imgsz=1280,conf=.25,iou=.7,nms=True,device='cpu',
                       stream=True,save=False,verbose=False):
        if r.keypoints is None:
            raise ValueError('这里要求Pose权重，检测权重请用原mypredict.py')
        boxes=r.boxes.xyxy.cpu().numpy()
        raw_boxes=boxes.copy()
        raw=r.keypoints.data.cpu().numpy().copy()
        safe,changed=contained_points(raw,boxes)
        points=safe if a.containment=='clip' else raw
        if a.containment=='expand':
            boxes=containing_boxes(boxes,raw)
            box_data=r.boxes.data.clone()
            box_data[:,:4]=torch.as_tensor(boxes,device=box_data.device)
            r.update(boxes=box_data)
        h,w=r.orig_shape
        stem=Path(r.path).stem
        payload=dict(image=str(r.path),weights=str(a.weights),containment=a.containment,
                     boxes=boxes.tolist(),raw_points=raw.tolist(),output_points=points.tolist(),
                     raw_boxes=raw_boxes.tolist(),outside_original_box=changed.tolist(),
                     moved_points=(changed if a.containment=='clip' else np.zeros_like(changed)).tolist(),
                     needs_review=(changed.any(1)|(raw[:,0,2]<.3)).tolist())
        (a.output/'raw'/f'{stem}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
        r.update(keypoints=torch.as_tensor(points,device=r.keypoints.data.device))
        image=r.plot(line_width=1,kpt_radius=4,kpt_line=False,labels=False)
        # 被调整的点用橙色提醒，避免看着规整就误以为原始预测正确。
        for pi,ki in zip(*np.where(changed)):
            x,y=points[pi,ki,:2]
            cv2.circle(image,(round(float(x)),round(float(y))),7,(0,140,255),2)
        cv2.imencode('.jpg',image)[1].tofile(str(a.output/f'{stem}.jpg'))
        lines=[]
        for i,box in enumerate(boxes):
            xywh=[(box[0]+box[2])/2/w,(box[1]+box[3])/2/h,(box[2]-box[0])/w,(box[3]-box[1])/h]
            normalized=points[i].copy();normalized[:,:2]/=[w,h]
            fields=[int(r.boxes.cls[i].item()),*xywh,*normalized.flatten()]
            lines.append(' '.join(f'{v:.6f}' for v in fields))
        (a.output/'labels'/f'{stem}.txt').write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf8')
        count+=1;flagged+=sum(payload['needs_review'])
        print(f'{stem}: {len(boxes)}框，{sum(payload["needs_review"])}个待复核',flush=True)
    print(f'完成{count}图；待复核实例{flagged}。橙圈=原始点出框；仅clip会移动点，不代表定位准确。')


if __name__=='__main__':main()
