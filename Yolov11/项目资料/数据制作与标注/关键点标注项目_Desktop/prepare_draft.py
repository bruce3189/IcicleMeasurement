from pathlib import Path
import csv, json, shutil, os
from collections import defaultdict
root = Path(r'D:\make_dataset\exports\ice_yolo_pose_labelme_20261013')
(root / '.ultralytics').mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(root / '.ultralytics')
from ultralytics import YOLO

src = Path(r'D:\make_dataset\dist\projects\e7b8015668b3-ice_clear_400_20261012\curated_runs\quality_filter-6aaf36e7839c\frames')
manifest = src.parent / 'frames_manifest.csv'
images = root / 'images'
labels = root / 'labels'
images.mkdir(parents=True, exist_ok=True)
labels.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig', newline='')))
byname = {r['image']: r for r in rows}
files = sorted(p for p in src.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
model = YOLO(r'D:\yolov11\runs\detect\train10\weights\best.pt')
# Predict in batches; this checkpoint detects icicles only and has no learned keypoints.
predictions = model.predict(source=str(src), imgsz=640, conf=0.25, iou=0.5, batch=2, stream=True, verbose=False, device='cpu')
results = {}
for result in predictions:
    path = Path(result.path)
    boxes=[]
    if result.boxes is not None:
        for b in result.boxes:
            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
            conf=float(b.conf[0])
            if x2>x1 and y2>y1:
                boxes.append([x1,y1,x2,y2,conf])
    results[path.name]=boxes

# Video-level 60/20/20 split avoids placing frames from the same source video across splits.
split_by_video = {'001.mp4':'train','002.mp4':'train','003.mp4':'train','004.mp4':'val','005.mp4':'test'}
counts=defaultdict(int); object_count=0; partial_count=0
for i,p in enumerate(files):
    row=byname.get(p.name,{})
    split=split_by_video.get(Path(row.get('source_video','')).name,'train')
    w=int(row.get('width') or 0); h=int(row.get('height') or 0)
    if not w or not h:
        from PIL import Image
        with Image.open(p) as im: w,h=im.size
    shutil.copy2(p, images/p.name)
    boxes=results.get(p.name,[])
    # Border crop or substantial overlap gives a conservative partial-visibility draft.
    classes=[]
    for j,b in enumerate(boxes):
        x1,y1,x2,y2,conf=b
        edge = x1<=3 or y1<=3 or x2>=w-4 or y2>=h-4
        overlap=False
        area=max(1,(x2-x1)*(y2-y1))
        for k,c in enumerate(boxes):
            if j==k: continue
            ix=max(0,min(x2,c[2])-max(x1,c[0])); iy=max(0,min(y2,c[3])-max(y1,c[1]))
            if ix*iy/min(area,max(1,(c[2]-c[0])*(c[3]-c[1]))) >= 0.12:
                overlap=True; break
        classes.append(1 if edge or overlap else 0)
    shapes=[]; yolo=[]
    for j,(b,cls) in enumerate(zip(boxes,classes),start=1):
        x1,y1,x2,y2,conf=b
        x1=max(0,min(w-1,x1)); x2=max(x1+1,min(w,x2)); y1=max(0,min(h-1,y1)); y2=max(y1+1,min(h,y2))
        label='部分可见' if cls==1 else '完整入镜'
        shapes.append({'label':label,'points':[[x1,y1],[x2,y2]],'group_id':j,'description':f'模型候选框，置信度 {conf:.3f}；请人工检查类别和边界','shape_type':'rectangle','flags':{'draft':True}})
        # Geometric initial points: the box's lower center is the icicle tip; upper corners estimate the base.
        pts=[((x1+x2)/2,y2,2),(x1,y1,2),(x2,y1,2)]
        for nm,(x,y,vis) in zip(('tip','base_left','base_right'),pts):
            shapes.append({'label':nm,'points':[[x,y]],'group_id':j,'description':'按检测框推算的关键点草稿，请在 Labelme 中移动到真实结构点','shape_type':'point','flags':{'draft':True}})
        cx=((x1+x2)/2)/w; cy=((y1+y2)/2)/h; bw=(x2-x1)/w; bh=(y2-y1)/h
        flat=[str(cls),f'{cx:.6f}',f'{cy:.6f}',f'{bw:.6f}',f'{bh:.6f}']
        for x,y,v in pts: flat += [f'{x/w:.6f}',f'{y/h:.6f}',str(v)]
        yolo.append(' '.join(flat)); object_count+=1; partial_count+=cls
    payload={'version':'5.8.1','flags':{'draft_prelabel':True},'shapes':shapes,'imagePath':p.name,'imageData':None,'imageHeight':h,'imageWidth':w}
    (images/(p.name+'.json')).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    (labels/(p.stem+'.txt')).write_text(('\n'.join(yolo)+'\n') if yolo else '',encoding='utf-8')
    counts[split]+=1
    with (root/'splits'/f'{split}.txt').open('a',encoding='utf-8') if False else open('NUL','w') as _:
        pass

(root/'splits').mkdir(exist_ok=True)
for split in ('train','val','test'):
    selected=[p.name for p in files if split_by_video.get(Path(byname.get(p.name,{}).get('source_video','')).name,'train')==split]
    (root/'splits'/f'{split}.txt').write_text(''.join(f'../images/{name}\n' for name in selected),encoding='utf-8')
(root/'dataset.yaml').write_text('path: .\ntrain: splits/train.txt\nval: splits/val.txt\ntest: splits/test.txt\nnames:\n  0: complete\n  1: partial\nkpt_shape: [3, 3]\nkpt_names:\n  0: [tip, base_left, base_right]\n',encoding='utf-8')
(root/'classes.txt').write_text('0 complete (完整入镜)\n1 partial (部分可见，边缘裁切或明显遮挡)\n',encoding='utf-8')
summary={'images':len(files),'objects':object_count,'partial_draft_objects':partial_count,'images_by_split':dict(counts),'empty_detection_images':sum(not results.get(p.name) for p in files)}
(root/'draft_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))



