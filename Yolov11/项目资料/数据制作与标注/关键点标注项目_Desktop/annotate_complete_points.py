from pathlib import Path
import json, subprocess, sys
ROOTS=[Path(r'D:\make_dataset\exports\ice_yolo_pose_labelme_20261013'),Path(r'C:\Users\Administrator\Desktop\Iceicle_Labelme_YOLOPose_20261013')]
for root in ROOTS:
    added=0; complete=0; partial=0
    image_dir=root/'images'
    for jf in sorted(image_dir.glob('*.json')):
        if not (image_dir/(jf.stem+'.jpg')).exists() and not (image_dir/(jf.stem+'.png')).exists():
            continue
        data=json.loads(jf.read_text(encoding='utf-8-sig'))
        shapes=[s for s in data.get('shapes',[]) if s.get('shape_type')!='point']
        for rect in [s for s in shapes if s.get('shape_type')=='rectangle']:
            if rect.get('label')!='完整入镜':
                partial+=1
                continue
            complete+=1
            (x1,y1),(x2,y2)=rect['points']; w=x2-x1; h=y2-y1; gid=rect.get('group_id')
            # Approximate triangle vertices on the icicle body: tapered tip and two upper shoulder points.
            pts={'tip':((x1+x2)/2, y2-0.025*h), 'base_left':(x1+0.10*w,y1+0.16*h), 'base_right':(x2-0.10*w,y1+0.16*h)}
            for label,(x,y) in pts.items():
                shapes.append({'label':label,'points':[[x,y]],'group_id':gid,'description':'按冰凌轮廓近似三角形布置在冰体上的候选点；请在 Labelme 中按可见边缘微调','shape_type':'point','flags':{'draft':True}})
                added+=1
        data['shapes']=shapes
        jf.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    # Convert from standard same-basename Labelme JSON after updating this copy.
    subprocess.run([sys.executable,str(root/'labelme_to_yolo_pose.py')],check=True,cwd=str(root),stdout=subprocess.DEVNULL)
    print(json.dumps({'project':str(root),'complete_objects':complete,'partial_objects':partial,'points':added},ensure_ascii=False))

