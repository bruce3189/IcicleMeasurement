"""Native Qt integration review; no model inference or threshold modifications."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.stdout=sys.stderr=(Path(__file__).parent/'unified_ui.log').open('w',encoding='utf-8',buffering=1)
import json
import csv
import traceback
from unittest.mock import patch
from PyQt5 import QtWidgets,QtCore
from icicle_measure.gui import MainWindow,configure_font
from icicle_measure.pseudo_gui import PseudoDepthPanel
from icicle_measure.pseudo_depth import ROOT,load_project
from icicle_measure.render import verified_distance
from icicle_measure.io import write_json

app=QtWidgets.QApplication([]); configure_font(app)
w=MainWindow(); w.resize(1366,768)
with patch.object(PseudoDepthPanel,'choose_sample'):
    w.open_pseudo_depth()
p=w.pseudo_dialog
folder=ROOT/'outputs/pseudo_depth_refined/002_00002'
p.receive({'result':load_project(folder),'root':folder})
p.view_kind.setCurrentIndex(p.view_kind.findData('complete'))
w.show()
output=ROOT/'outputs/unified_workspace_review'; output.mkdir(parents=True,exist_ok=True)
timer=QtCore.QTimer(); timer.setInterval(50)

def capture():
    if w.thread is not None or w._pending_pseudo is not None:
        return
    timer.stop()
    try:
        assert w.result is not None and p.measurement is w.result
        passed=next((o for o in w.result.objects if verified_distance(o)),None)
        if passed:
            row=next(i for i,(o,e) in enumerate(w.row_map) if o.id==passed.id)
            w.table.selectRow(row)
        p.view_kind.setCurrentIndex(p.view_kind.findData('rgb'))
        report={'embedded':not p.isWindow(),'tabs':[w.workspace.tabText(i) for i in range(2)],
                'depth_variant':w.frame.metadata['depth_variant'],'distance_note':w.distance_note.text(),
                'passed_count':sum(verified_distance(o) for o in w.result.objects),'sizes':[]}
        for width,height in [(1366,768),(1920,1080)]:
            w.resize(width,height); app.processEvents()
            assert w.width()==width and w.height()==height
            for index,name in [(0,'scene'),(1,'analysis')]:
                w.workspace.setCurrentIndex(index); app.processEvents()
                p.refresh(fit=True); w.refresh_image(fit=True); app.processEvents()
                if index==1:
                    assert w.plot.terrain.canvas.height()>=170
                w.grab().save(str(output/f'{name}_{width}x{height}.png'))
            report['sizes'].append([w.width(),w.height()])
        exported=output/'measurement_export'; w.export_to(exported)
        data=json.loads((exported/'measurement.json').read_text(encoding='utf-8'))
        if passed:
            exported_obj=next(o for o in data['objects'] if o['id']==passed.id)
            assert exported_obj['distance_m']==passed.distance_m
            assert ('%.2f mm'%(passed.distance_m*1000)) in w.distance_note.text()
            assert w.distance_note.text()==w.plot.terrain.distance_note.text()
            assert ('%.2f mm'%(passed.distance_m*1000)) in p.distance_note.text()
            records=list(csv.DictReader((exported/'measurement.csv').open(encoding='utf-8-sig')))
            assert all(float(r['distance_m'])==passed.distance_m for r in records if r['id']==passed.id)
            report['verified_distance_m']=passed.distance_m
            report['selected_id']=passed.id
        report['export_matches_ui']=True
        write_json(output/'validation_report.json',report)
        w.resize(1366,768); w.workspace.setCurrentIndex(1); app.processEvents(); w.refresh_image(fit=True)
        print(report,flush=True)
        if '--keep-open' not in sys.argv:
            w.close(); app.quit()
    except Exception:
        traceback.print_exc()
        w.close(); app.quit()

timer.timeout.connect(capture); timer.start()
sys.exit(app.exec_())
