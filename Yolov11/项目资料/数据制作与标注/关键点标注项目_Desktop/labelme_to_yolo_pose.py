"""Convert Labelme JSON to the project's Ultralytics YOLO Pose labels.

Draw one rectangle per icicle and three point shapes on the icicle body:
tip, base_left, base_right. Give all shapes the same group_id when possible.
If Labelme leaves group_id empty, a point is assigned to the smallest box
containing it. Missing points become visibility=0 in the YOLO Pose output.
"""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parent
IMAGE_DIR, LABEL_DIR = ROOT/'images', ROOT/'labels'
CLASS_IDS = {'完整入镜': 0, '部分可见': 1, '严重重叠': 2}
POINT_NAMES = ('tip', 'base_left', 'base_right')
POINT_ALIASES = {'顶点':'tip', '底角_左':'base_left', '底角_右':'base_right'}

def convert_one(path):
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    w,h=int(data['imageWidth']),int(data['imageHeight'])
    items=[]; by_group={}; loose=[]
    for shape in data.get('shapes',[]):
        kind,label=shape.get('shape_type'),shape.get('label','')
        gid=shape.get('group_id')
        if kind=='rectangle':
            if label not in CLASS_IDS: raise ValueError(f'{path.name}: unknown box class {label!r}')
            a,b=shape['points']; box=(min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1]))
            if box[2]<=box[0] or box[3]<=box[1]: continue
            item={'box':box,'cls':CLASS_IDS[label],'points':{}}
            items.append(item)
            if gid is not None: by_group.setdefault(gid,item)
        elif kind=='point' and (name:=POINT_ALIASES.get(label,label)) in POINT_NAMES:
            record=(name,shape['points'][0],shape.get('flags',{}),gid)
            if gid is None: loose.append(record)
            elif gid in by_group: by_group[gid]['points'][name]=(record[1],record[2])
            else: loose.append(record)
    for name,xy,flags,gid in loose:
        x,y=xy
        candidates=[it for it in items if it['box'][0]<=x<=it['box'][2] and it['box'][1]<=y<=it['box'][3]]
        if candidates:
            item=min(candidates,key=lambda it:(it['box'][2]-it['box'][0])*(it['box'][3]-it['box'][1]))
            item['points'][name]=(xy,flags)
    out=[]
    for it in items:
        x1,y1,x2,y2=it['box']
        row=[str(it['cls']),f'{(x1+x2)/2/w:.6f}',f'{(y1+y2)/2/h:.6f}',f'{(x2-x1)/w:.6f}',f'{(y2-y1)/h:.6f}']
        for name in POINT_NAMES:
            point=it['points'].get(name)
            if point is None: row.extend(('0.000000','0.000000','0'))
            else:
                (x,y),flags=point
                row.extend((f'{x/w:.6f}',f'{y/h:.6f}',str(1 if flags.get('occluded') else 2)))
        out.append(' '.join(row))
    return '\n'.join(out)+('\n' if out else '')

def main():
    LABEL_DIR.mkdir(exist_ok=True)
    files=[p for p in sorted(IMAGE_DIR.glob('*.json')) if (IMAGE_DIR/(p.stem+'.jpg')).exists() or (IMAGE_DIR/(p.stem+'.png')).exists()]
    for p in files: (LABEL_DIR/(p.stem+'.txt')).write_text(convert_one(p),encoding='utf-8')
    print(f'Converted {len(files)} Labelme JSON files.')
if __name__=='__main__': main()
