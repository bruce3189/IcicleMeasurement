"""Before/after UI, source-body, numerical and performance snapshots."""
import argparse
import ast
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM','windows')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def definition_digest(value):
    module=importlib.import_module(value.__module__)
    node=ast.parse(inspect.getsource(value))
    for child in ast.walk(node):
        if isinstance(child,ast.ImportFrom) and child.level:
            child.module=importlib.util.resolve_name('.'*child.level+(child.module or ''),module.__package__)
            child.level=0
    return digest(ast.dump(node,include_attributes=False).encode())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['before','after']);args=parser.parse_args()
    output=ROOT/'outputs/modularization_review'/args.stage;output.mkdir(parents=True,exist_ok=True)
    startup=[]
    snippet="import time,json,sys;t=time.perf_counter();import icicle_measure.gui;g=time.perf_counter();import icicle_measure.pseudo_gui;p=time.perf_counter();print(json.dumps({'gui_ms':(g-t)*1000,'total_ms':(p-t)*1000,'torch_loaded':'torch' in sys.modules,'camera_sdk_loaded':'pyrealsense2' in sys.modules}))"
    for _ in range(5):
        child=subprocess.run([sys.executable,'-c',snippet],cwd=str(ROOT),capture_output=True,text=True,check=True)
        startup.append(json.loads(child.stdout.strip().splitlines()[-1]))
    import numpy as np
    from PyQt5 import QtWidgets,QtCore,QtGui
    from icicle_measure import gui,pseudo_gui
    from icicle_measure.pseudo_depth import load_project
    from icicle_measure.simulator import SCENARIOS,generate
    from icicle_measure.core import measure_frame
    from dataclasses import asdict
    definitions={}
    names={gui:['configure_font','ImageView','PlotCanvas','HistogramCanvas','WindowCountsCanvas',
        'DepthTerrainCanvas','DepthTerrainPanel','MeasurementCriteriaPanel','DiagnosticsPlot','ConfigDialog',
        'Worker','PoseMeasurementWorker','MainWindow','main'],
        pseudo_gui:['scene_preview','EditView','PseudoTask','PseudoDepthPanel','PseudoDepthDialog']}
    for module,objects in names.items():
        for name in objects:definitions[module.__name__+'.'+name]=definition_digest(getattr(module,name))
    numerical={}
    for key in SCENARIOS:
        s=generate(key,42)
        result=measure_frame(s.frame,s.detections,s.intrinsics)
        numerical[key]=digest(json.dumps(asdict(result),ensure_ascii=False,sort_keys=True).encode())
    projects={}
    for stem in ('002_00002','002_00041'):
        result=load_project(ROOT/'outputs/pseudo_depth_refined'/stem)
        projects[stem]={'complete':digest(result.clean_raw.tobytes()),'observed':digest(result.observed_raw.tobytes()),
            'report':digest(json.dumps(result.report,ensure_ascii=False,sort_keys=True).encode())}
    untouched={p.name:digest(p.read_bytes()) for p in (ROOT/'icicle_measure').glob('*.py')
               if p.name not in ('gui.py','pseudo_gui.py')}
    app=QtWidgets.QApplication([]);gui.configure_font(app)
    w=gui.MainWindow()
    with patch.object(pseudo_gui.PseudoDepthPanel,'choose_sample'):w.open_pseudo_depth()
    p=w.pseudo_dialog
    p.receive({'result':load_project(ROOT/'outputs/pseudo_depth_refined/002_00002')})
    p.view_kind.setCurrentIndex(p.view_kind.findData('complete'))
    def settle():
        deadline=time.monotonic()+15
        while w.thread is not None or w._pending_pseudo is not None:
            if time.monotonic()>deadline:raise RuntimeError('measurement did not settle')
            app.processEvents()
        app.processEvents()
    settle()
    w.show();w.activateWindow()
    state=[];images={}
    def capture(name):
        app.processEvents()
        QtGui.QCursor.setPos(w.mapToGlobal(QtCore.QPoint(5,5)))
        p.show_pixel_info(None)
        app.processEvents()
        image=w.grab().toImage().convertToFormat(QtGui.QImage.Format_RGBA8888)
        data=image.bits();data.setsize(image.byteCount())
        images[name]=dict(size=[image.width(),image.height()],rgba_sha256=digest(bytes(data)))
        image.save(str(output/(name+'.png')))
    for width,height in ((1366,768),(1920,1080)):
        w.resize(width,height);app.processEvents()
        for index,label in ((0,'scene'),(1,'analysis')):
            w.workspace.setCurrentIndex(index);app.processEvents()
            p.refresh(fit=True);w.refresh_image(fit=True);app.processEvents()
            capture(f'{label}_{width}x{height}')
        state.append({'size':[w.width(),w.height()],'distance':w.distance_note.text(),
            'table':[[w.table.item(r,c).text() for c in range(w.table.columnCount())] for r in range(w.table.rowCount())],
            'tabs':[w.plot.tabs.tabText(i) for i in range(w.plot.tabs.count())]})
    w.resize(1366,768);app.processEvents()
    for i in range(w.plot.tabs.count()):
        w.plot.tabs.setCurrentIndex(i);app.processEvents();capture(f'chart_{i}')
    w.plot.tabs.setCurrentWidget(w.plot.terrain)
    c=w.plot.terrain.canvas;c.grab()
    geometry=c.relative_m.copy();origin=c.origin_xyz_m.copy()
    choices=np.flatnonzero(c._proj_category==3)
    index=int(c._proj_index[choices[len(choices)//2]])
    c.selected_pixel_id=int(c.sample_pixel_ids[index])
    w.plot.terrain.expand_button.setChecked(True);app.processEvents()
    c.focus_selected();app.processEvents();capture('focused_expanded')
    timings=[]
    for i in range(40):
        start=time.perf_counter();c.rotate_axis('y',9);c.grab()
        if i>=4:timings.append((time.perf_counter()-start)*1000)
    np.testing.assert_array_equal(geometry,c.relative_m);np.testing.assert_array_equal(origin,c.origin_xyz_m)
    report={'definitions':definitions,'numerical':numerical,'projects':projects,'unchanged_sources':untouched,
        'images':images,'ui_state':state,'startup_runs':startup,
        'startup_median_ms':statistics.median(row['total_ms'] for row in startup),
        'render_median_ms':float(np.median(timings)),'render_p95_ms':float(np.percentile(timings,95)),
        'sample_count':len(c.vertices)}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    w.close();app.processEvents()
    if args.stage=='after':
        before=json.loads((output.parent/'before/report.json').read_text(encoding='utf-8'))
        equal={key:report[key]==before[key] for key in ('definitions','numerical','projects','unchanged_sources','ui_state','images')}
        comparison={'equal':equal,'startup_before_ms':before['startup_median_ms'],'startup_after_ms':report['startup_median_ms'],
            'render_before_ms':before['render_median_ms'],'render_after_ms':report['render_median_ms'],
            'render_before_p95_ms':before['render_p95_ms'],'render_after_p95_ms':report['render_p95_ms']}
        (output.parent/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(comparison,ensure_ascii=False),flush=True)
    else:print(json.dumps({'stage':args.stage,'definitions':len(definitions),'images':len(images),
        'startup_ms':report['startup_median_ms'],'render_ms':report['render_median_ms']}),flush=True)


if __name__=='__main__':main()
