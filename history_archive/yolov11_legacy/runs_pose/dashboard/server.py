"""训练进度看板服务（标准库实现，无第三方依赖）

启动：D:\\318\\envs\\yolo26\\python.exe D:\\yolov11\\runs_pose\\dashboard\\server.py 8890
浏览器打开 http://localhost:8890
"""
import csv
import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent.parent          # D:\yolov11\runs_pose
HERE = Path(__file__).resolve().parent                 # ...\dashboard
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890


def list_runs():
    out = []
    for d in sorted(BASE.iterdir()):
        if d.is_dir() and (d / "results.csv").exists():
            mtime = (d / "results.csv").stat().st_mtime
            out.append({"name": d.name, "mtime": mtime})
    out.sort(key=lambda x: -x["mtime"])
    return [o["name"] for o in out]


def read_run(name):
    run = BASE / name
    csv_path = run / "results.csv"
    if not csv_path.exists():
        return {"error": "没有 results.csv"}

    total_epochs = 100
    args = run / "args.yaml"
    if args.exists():
        for line in args.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("epochs:"):
                try:
                    total_epochs = int(float(line.split(":", 1)[1].strip()))
                except ValueError:
                    pass
                break

    rows = []
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if lines:
        reader = csv.DictReader(lines)
        for r in reader:
            try:
                rows.append({
                    "epoch": int(float(r["epoch"])),
                    "time": float(r["time"]),
                    "box50": float(r["metrics/mAP50(B)"]),
                    "box5095": float(r["metrics/mAP50-95(B)"]),
                    "pose50": float(r["metrics/mAP50(P)"]),
                    "pose5095": float(r["metrics/mAP50-95(P)"]),
                    "poseP": float(r["metrics/precision(P)"]),
                    "poseR": float(r["metrics/recall(P)"]),
                    "box_loss": float(r["train/box_loss"]),
                    "pose_loss": float(r["train/pose_loss"]),
                    "lr": float(r.get("lr/pg0") or 0),
                })
            except (KeyError, ValueError):
                continue  # 跳过正在写入的半行

    done = len(rows)
    per = (rows[-1]["time"] / done) if done else 0.0
    eta = ((total_epochs - done) * per / 60.0) if done else 0.0
    best = max(rows, key=lambda r: 0.1 * r["pose50"] + 0.9 * r["pose5095"]) if rows else None

    age = datetime.now().timestamp() - csv_path.stat().st_mtime
    if done >= total_epochs:
        status = "finished"
    elif age < 180:
        status = "running"
    elif age < 900:
        status = "idle"
    else:
        status = "stopped"

    return {
        "run": name,
        "rows": rows,
        "done": done,
        "total": total_epochs,
        "per_epoch": round(per, 1),
        "eta_min": round(eta, 1),
        "best": best,
        "status": status,
        "csv_age": round(age, 1),
        "server_time": datetime.now().strftime("%H:%M:%S"),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
        elif u.path == "/api/runs":
            self._send(200, json.dumps({"runs": list_runs()}), "application/json; charset=utf-8")
        elif u.path == "/api/results":
            q = parse_qs(u.query)
            runs = list_runs()
            name = (q.get("run") or [runs[0] if runs else ""])[0]
            if not name:
                self._send(200, json.dumps({"error": "找不到任何运行目录"}), "application/json; charset=utf-8")
                return
            self._send(200, json.dumps(read_run(name)), "application/json; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")


if __name__ == "__main__":
    print(f"看板已启动 -> http://localhost:{PORT}   数据源 {BASE}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
