"""续训 YOLO26 pose（从 last.pt 恢复，接第 59 轮跑到 100）

环境：D:\318\envs\yolo26\python.exe
用法：在 D:\yolov11\runs_pose 下运行
      D:\318\envs\yolo26\python.exe -u resume_pose26.py > train26_resume.log 2>&1
"""
from pathlib import Path

from ultralytics import YOLO

LAST = Path(r"D:\yolov11\runs_pose\ice80_yolo26\weights\last.pt")

if __name__ == "__main__":
    if not LAST.exists():
        raise SystemExit("找不到 last.pt: {}".format(LAST))
    model = YOLO(str(LAST))
    # resume=True 会沿用 checkpoint 里的 args（data/imgsz/batch/project/name 全部保留），
    # 不要在这里重新传参数，否则会开一个新 run 目录、从第 1 轮重来
    model.train(resume=True)
    print("RESUME_DONE", LAST.parent / "best.pt")
