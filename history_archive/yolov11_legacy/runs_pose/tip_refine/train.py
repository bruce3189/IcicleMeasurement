"""透明的小规模tip加权实验。默认只预览；显式--run才训练。"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

def available_commit_gb():
    if os.name!='nt':return None
    import ctypes
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                  ('total_ram',ctypes.c_ulonglong),('available_ram',ctypes.c_ulonglong),
                  ('total_commit',ctypes.c_ulonglong),('available_commit',ctypes.c_ulonglong),
                  ('total_virtual',ctypes.c_ulonglong),('available_virtual',ctypes.c_ulonglong),
                  ('extended',ctypes.c_ulonglong)]
    state=Memory();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError('无法读取Windows内存状态')
    return state.available_commit/1024**3


if '--run' in sys.argv:
    free=available_commit_gb()
    if free is not None and free<2.:
        raise SystemExit(f'当前可提交内存仅{free:.2f}GB，本实验至少预留2GB再启动；请释放不用的应用。未启动训练。')

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import torch
import ultralytics
from ultralytics import YOLO
from weighted_pose import TipTrainer, point_weights
from evaluate import ORIGINAL, HOME, evaluate


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true',help='不加此项只打印实验说明，不训练')
    p.add_argument('--tip-weight',type=float,default=3.,help='tip相对单个base点的权重；1为对照')
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--lr',type=float,default=.0002)
    p.add_argument('--name',default='tip3_pilot')
    p.add_argument('--fraction',type=float,default=1.,help='只用于冒烟检查，正式对照必须为1')
    p.add_argument('--batch',type=int,default=1,help='默认1，减少笔记本内存占用')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--head-only',action='store_true',help='固定特征层和检测头（含BN），只训练关键点分支')
    p.add_argument('--tip-l1',type=float,default=0.,help='直接tip坐标辅助损失的系数；0关闭，位于pose总权重内')
    p.add_argument('--resume-from',type=Path,help='仅恢复尚未完成训练的last.pt；保持实验名和加权设置相同')
    return p


def main():
    a=parser().parse_args()
    if a.tip_l1<0:
        raise ValueError('--tip-l1不能为负数')
    torch.set_num_threads(a.threads)
    if ultralytics.__version__!='8.4.183':
        raise RuntimeError('本实验已核对Ultralytics 8.4.183，请使用D:/318/envs/yolo26/python.exe')
    weights=point_weights(a.tip_weight).tolist()
    project=HOME/'experiments'
    destination=project/a.name
    config=dict(data=str(HOME/'data_snapshot'/'dataset.yaml'),
                epochs=a.epochs,imgsz=1280,rect=True,batch=a.batch,workers=0,device='cpu',
                optimizer='AdamW',lr0=a.lr,lrf=.1,warmup_epochs=1.,warmup_bias_lr=0.,
                seed=42,deterministic=True,cache=False,fliplr=0.,flipud=0.,mosaic=0.,
                degrees=0.,translate=.05,scale=.3,pose=12.,rle=1.,kobj=1.,
                nms=True,iou=.7,project=str(project),name=a.name,exist_ok=False,
                save=True,save_period=-1,plots=False,patience=a.epochs+1,
                fraction=a.fraction,resume=bool(a.resume_from))
    explanation=dict(question='只加大tip的相对权重，是否改善真实尖端且不损伤框？',
        initialization=str(ORIGINAL),tip_base_ratio=[a.tip_weight,1,1],normalized_weights=weights,
        changes='OKS定位与RLE同时逐点加权；框损失、可见性损失和sigma不变',
        evaluation='val；1280；conf=.25；NMS=.7；匹配IoU>=.5；原始二维点，不贴框底',
        limitation='从已训练best.pt开始的开发实验；原始划分含相邻帧，不是独立泛化结论',
        outputs=str(destination),training=config)
    explanation['head_only']=a.head_only
    explanation['tip_l1']=a.tip_l1
    explanation['resume_from']=str(a.resume_from) if a.resume_from else None
    print(json.dumps(explanation,ensure_ascii=False,indent=2),flush=True)
    if not a.run:
        print('这是预览。确认参数后加 --run 才会训练。',flush=True)
        return
    if destination.exists() and not a.resume_from:
        raise FileExistsError(f'为保留实验记录，请更换--name：{destination}')
    if a.resume_from:
        if a.resume_from.resolve().parent.parent!=destination.resolve():
            raise ValueError('--name必须与断点所属实验目录一致')
        previous=json.loads((destination/'experiment.json').read_text(encoding='utf8'))
        if (previous['tip_base_ratio'][0]!=a.tip_weight or previous.get('tip_l1',0)!=a.tip_l1
                or previous.get('head_only',False)!=a.head_only):
            raise ValueError('续训必须沿用原实验的tip-weight、tip-l1和head-only设置')
        explanation['resume_checkpoint_sha256']=hashlib.sha256(a.resume_from.read_bytes()).hexdigest()
    if not Path(config['data']).exists():
        raise FileNotFoundError('请先运行 snapshot_data.py，训练仅使用工作区快照。')
    model=YOLO(str(a.resume_from or ORIGINAL))
    explanation['source_sha256']=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    def record_start(trainer):
        if trainer.head_only:
            trainer._model_train()
        explanation['effective_freeze']=trainer.args.freeze
        explanation['trainable_parameter_names']=[n for n,p in trainer.model.named_parameters() if p.requires_grad]
        explanation['loss_branch_weights']={
            name:getattr(trainer.model.criterion,name).keypoint_loss.weights.tolist()
            for name in ('one2many','one2one')}
        Path(trainer.save_dir,'experiment.json').write_text(
            json.dumps(explanation,ensure_ascii=False,indent=2),encoding='utf8')
    model.add_callback('on_train_start',record_start)
    def record_epoch(trainer):
        Path(trainer.save_dir,'progress.json').write_text(json.dumps(dict(
            completed_epochs=trainer.epoch+1,planned_epochs=trainer.epochs,
            status='training',last_checkpoint=str(trainer.last)),ensure_ascii=False,indent=2),encoding='utf8')
    model.add_callback('on_model_save',record_epoch)
    # 使用标准PoseModel，只有训练criterion定制；保存的权重无需自定义类即可推理。
    def trainer_factory(*args,**kwargs):
        return TipTrainer(*args,tip_weight=a.tip_weight,head_only=a.head_only,tip_l1=a.tip_l1,**kwargs)
    model.train(trainer=trainer_factory,**config)
    (destination/'experiment.json').write_text(json.dumps(explanation,ensure_ascii=False,indent=2),encoding='utf8')
    # 用固定训练轮数的last比较，不按综合Pose mAP挑选对tip有利的轮次。
    summary=evaluate(destination/'weights'/'last.pt',output=destination/'raw_val')
    (destination/'progress.json').write_text(json.dumps(dict(completed_epochs=a.epochs,planned_epochs=a.epochs,status='completed'),indent=2),encoding='utf8')
    print('FINAL_RAW_VAL',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
