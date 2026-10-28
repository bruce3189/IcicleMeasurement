"""Read-only source/data audit; writes diagnostic evidence only beside this script."""
from pathlib import Path
import json
import hashlib
import csv
import numpy as np
from PIL import Image
from ultralytics import YOLO
import ultralytics

ROOT = Path('C:/Users/Administrator/Desktop/Iceicle_Easy80_YOLOPose_521')
OUT = Path(__file__).resolve().parent

def ious(a, b):
    a, b = np.asarray(a).reshape(-1,4), np.asarray(b).reshape(-1,4)
    wh = np.maximum(0, np.minimum(a[:,None,2:], b[None,:,2:])-np.maximum(a[:,None,:2], b[None,:,:2]))
    inter=wh.prod(2)
    return inter/np.maximum(1e-9,np.prod(a[:,2:]-a[:,:2],1)[:,None]+np.prod(b[:,2:]-b[:,:2],1)[None,:]-inter)

def match(g,p,thr):
    mat=ious(g,p); pairs=[]; ug=set(); up=set()
    for index in np.argsort(-mat.ravel()):
        gi,pi=np.unravel_index(index,mat.shape)
        if mat[gi,pi]<thr:break
        if gi not in ug and pi not in up:
            pairs.append((int(gi),int(pi),float(mat[gi,pi])));ug.add(gi);up.add(pi)
    return pairs

def load(sp):
    out=[]
    for ip in sorted((ROOT/'images'/sp).glob('*.jpg')):
        w,h=Image.open(ip).size
        a=np.loadtxt(ROOT/'labels'/sp/(ip.stem+'.txt'),ndmin=2)
        b=a[:,1:5].copy();b[:,:2]-=b[:,2:]/2;b[:,2:]+=b[:,:2];b*=np.array([w,h,w,h])
        k=a[:,5:].reshape(-1,3,3).copy();k[:,:,:2]*=np.array([w,h])
        out.append((ip,b,k))
    return out

def stats(v):
    v=np.asarray(v)
    return dict(n=len(v),median=float(np.median(v)),mean=float(np.mean(v)),p90=float(np.quantile(v,.9))) if len(v) else dict(n=0)

def main():
    model=YOLO('D:/yolov11/runs_pose/ice80_yolo26/weights/best.pt')
    result={'ultralytics':ultralytics.__version__,'module':ultralytics.__file__,'runs':{}}
    print('ENV',result,flush=True)
    records=[]
    for sz in (640,1280,1536):
        rows=[];counts={}; cache=[]
        for ip,b,k in load('test'):
            r=model.predict(str(ip),imgsz=sz,conf=.25,device='cpu',verbose=False)[0]
            pb=r.boxes.xyxy.cpu().numpy();pk=r.keypoints.data.cpu().numpy();conf=r.boxes.conf.cpu().numpy()
            keep=[]
            for pi in np.argsort(-conf):
                if all(ious(pb[pi:pi+1],pb[j:j+1])[0,0]<.55 for j in keep):keep.append(int(pi))
            counts[ip.stem]={'gt':len(b),'pred':len(pb),'after_iou055':len(keep),'match03':len(match(b,pb,.3)),'match05':len(match(b,pb,.5)),'match075':len(match(b,pb,.75)),'after_match05':len(match(b,pb[keep],.5))}
            cache.append({'image':str(ip),'gt_boxes':b.tolist(),'gt_keypoints':k.tolist(),'pred_boxes':pb.tolist(),'pred_keypoints':pk.tolist(),'pred_conf':conf.tolist(),'keep_iou055':keep})
            for gi,pi,iou in match(b,pb,.3):
                h=b[gi,3]-b[gi,1]; target=k[gi,0,:2]; tip=pk[pi,0,:2]
                fix=np.array([(pk[pi,1,0]+pk[pi,2,0])/2,pb[pi,3]-.04*(pb[pi,3]-pb[pi,1])])
                yonly=np.array([tip[0],fix[1]])
                gf=np.array([(k[gi,1,0]+k[gi,2,0])/2,b[gi,3]-.04*h])
                row={'image':ip.stem,'gi':gi,'pi':pi,'height':h,'iou':iou,'conf':float(conf[pi]),'tip_conf':float(pk[pi,0,2]),'raw':float(np.linalg.norm(tip-target)),'raw_y':float(abs(tip[1]-target[1])),'fix':float(np.linalg.norm(fix-target)),'fix_y':float(abs(fix[1]-target[1])),'yonly':float(np.linalg.norm(yonly-target)),'gt_rule':float(np.linalg.norm(gf-target)),'outside':int(not(pb[pi,0]<=tip[0]<=pb[pi,2] and pb[pi,1]<=tip[1]<=pb[pi,3]))}
                for ki in (1,2):row['base'+str(ki)]=float(np.linalg.norm(pk[pi,ki,:2]-k[gi,ki,:2])) if k[gi,ki,2]>0 else float('nan')
                rows.append(row)
        summary={'counts':counts}
        for name in ('raw','raw_y','fix','fix_y','yonly','gt_rule'):
            summary[name]=stats([r[name] for r in rows]);summary[name]['pck10']=float(np.mean([r[name]<.1*r['height'] for r in rows]))
        summary['base']=stats([r['base'+str(k)] for r in rows for k in (1,2) if np.isfinite(r['base'+str(k)])])
        summary['bins']={}
        for lo,hi in ((0,700),(700,1200),(1200,np.inf)):
            sub=[r for r in rows if lo<=r['height']<hi]
            summary['bins'][str(lo)]={name:stats([r[name] for r in sub]) for name in ('raw','fix','yonly')}
            summary['bins'][str(lo)]['raw_relative_median']=float(np.median([r['raw']/r['height'] for r in sub]))
        summary['outside']=sum(r['outside'] for r in rows)
        summary['bad_tip_high_conf']=int(sum(r['raw']>.1*r['height'] and r['tip_conf']>=.3 for r in rows))
        summary['iou']=stats([r['iou'] for r in rows])
        result['runs'][str(sz)]=summary
        (OUT/f'predictions_{sz}.json').write_text(json.dumps(cache,ensure_ascii=False,indent=2),encoding='utf8')
        with (OUT/f'matches_{sz}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print('SIZE',sz,json.dumps(summary,ensure_ascii=False),flush=True)
    # Test the rule with exact GT boxes and exact GT base points on all three splits.
    result['gt_rules']={}
    for sp in ('train','val','test'):
        errs=[];outside=0;visible=0
        for ip,b,k in load(sp):
            for gi in range(len(b)):
                for ki in range(3):
                    if k[gi,ki,2]>0:
                        x,y=k[gi,ki,:2];visible+=1;outside+=int(not(b[gi,0]-2<=x<=b[gi,2]+2 and b[gi,1]-2<=y<=b[gi,3]+2))
                if np.all(k[gi,:,2]>0):
                    fix=np.array([(k[gi,1,0]+k[gi,2,0])/2,b[gi,3]-.04*(b[gi,3]-b[gi,1])])
                    errs.append(np.linalg.norm(fix-k[gi,0,:2]))
        result['gt_rules'][sp]={'error':stats(errs),'visible':visible,'outside2px':outside}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('GT_RULES',result['gt_rules'],flush=True)

if __name__=='__main__':main()
