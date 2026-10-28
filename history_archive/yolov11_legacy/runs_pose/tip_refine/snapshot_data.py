"""将训练必需的图片、标签复制到实验目录；缓存不会写回桌面标注目录。"""
import hashlib
import json
import shutil
from pathlib import Path

SOURCE=Path('C:/Users/Administrator/Desktop/Iceicle_Easy80_YOLOPose_521')
TARGET=Path(__file__).resolve().parent/'data_snapshot'


def snapshot():
    if TARGET.exists():
        raise FileExistsError(f'快照已存在，请保留；如需新数据请建立新版本：{TARGET}')
    records=[]
    for kind,extension in [('images','*.jpg'),('labels','*.txt')]:
        for split in ('train','val','test'):
            destination=TARGET/kind/split
            destination.mkdir(parents=True)
            for path in sorted((SOURCE/kind/split).glob(extension)):
                target=destination/path.name
                shutil.copy2(path,target)
                records.append(dict(path=target.relative_to(TARGET).as_posix(),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    text=f'path: {TARGET.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ice\nkpt_shape: [3, 3]\nkpt_names:\n  0: [tip, base_left, base_right]\n'
    (TARGET/'dataset.yaml').write_text(text,encoding='utf8')
    (TARGET/'source_manifest.json').write_text(json.dumps(dict(source=str(SOURCE),files=records),ensure_ascii=False,indent=2),encoding='utf8')
    print(f'已复制{len(records)}个文件，未修改原始数据：{TARGET}')


if __name__=='__main__':snapshot()
