from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
from ultralytics import YOLO

model = YOLO(str(PROJECT_ROOT / 'runs/detect/train10/weights/best.pt'))
results = model.predict(
    source=str(PROJECT_ROOT.parent / 'Measurement/assets/rgb'),
    project=str(PROJECT_ROOT / 'runs/detect'),
    save=True,
    show=False,
    save_txt=True,
    line_width=1,
    stream=True,  # 流式处理：逐张释放内存，避免整批结果堆在内存里导致 OOM
)
for r in results:
    pass  # 只为消费生成器，边跑边释放；如需统计可在此处理 r.boxes
