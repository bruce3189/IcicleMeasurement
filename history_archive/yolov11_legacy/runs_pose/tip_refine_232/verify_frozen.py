"""核对只训练关键点的实验，是否真的保留了框/分类和特征提取权重及BN状态。"""
import argparse
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from evaluate import ORIGINAL


def verify(weight):
    torch.set_num_threads(2)
    original=YOLO(str(ORIGINAL)).model
    candidate=YOLO(str(weight)).model
    before,after=original.state_dict(),candidate.state_dict()
    prefix=f'model.{len(original.model)-1}.'
    allowed=('cv4.','cv4_kpts.','cv4_sigma.','one2one_cv4.','one2one_cv4_kpts.',
             'one2one_cv4_sigma.','flow_model.')
    checked=0;changed=[];pose_changed=[]
    if before.keys()!=after.keys():
        raise ValueError('模型结构不同，不能声称检测部分被原样保留')
    for key,value in before.items():
        pose=key.startswith(prefix) and key[len(prefix):].startswith(allowed)
        equal=torch.equal(value,after[key])
        if pose:
            if not equal:pose_changed.append(key)
        else:
            checked+=1
            if not equal:changed.append(key)
    result=dict(source=str(ORIGINAL),candidate=str(weight),frozen_tensors_checked=checked,
                frozen_tensors_changed=changed,pose_tensors_changed=len(pose_changed),
                passed=not changed and bool(pose_changed))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    result=verify(a.weights)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)
