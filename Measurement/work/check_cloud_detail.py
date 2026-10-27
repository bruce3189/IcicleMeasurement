"""Exercise real Qt selection, F focus, wheel zoom and inline expansion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.stdout=sys.stderr=(Path(__file__).parent/'cloud_detail_ui.log').open('w',encoding='utf-8',buffering=1)
import traceback
import time
import numpy as np
from unittest.mock import patch
from PyQt5 import QtCore,QtGui,QtWidgets,QtTest
from icicle_measure.gui import MainWindow,configure_font
from icicle_measure.pseudo_gui import PseudoDepthPanel
from icicle_measure.pseudo_depth import ROOT,load_project
from icicle_measure.io import write_json
from icicle_measure.render import verified_distance

app=QtWidgets.QApplication([]);configure_font(app)
w=MainWindow();w.resize(1366,768)
with patch.object(PseudoDepthPanel,'choose_sample'):w.open_pseudo_depth()
p=w.pseudo_dialog
folder=ROOT/'outputs/pseudo_depth_refined/002_00002'
p.receive({'result':load_project(folder),'root':folder})
p.view_kind.setCurrentIndex(p.view_kind.findData('complete'))
w.show();w.workspace.setCurrentIndex(1)
output=ROOT/'outputs/cloud_detail_review';output.mkdir(parents=True,exist_ok=True)
timer=QtCore.QTimer();timer.setInterval(50)

def screen(c,index):
    c.grab();plot=c.plot_rect();scale=c._fit_cache[1]*c.zoom
    center=np.asarray([plot.center().x(),plot.center().y()])-c._fit_cache[2]*scale+[c.pan.x(),c.pan.y()]
    return c._project(c.display_vertices()[[index]],scale,*center)[0][0]

def probe():
    if w.thread is not None or w._pending_pseudo is not None:return
    timer.stop()
    try:
        obj=next(o for o in w.result.objects if verified_distance(o))
        row=next(i for i,(o,e) in enumerate(w.row_map) if o.id==obj.id)
        w.table.selectRow(row);app.processEvents()
        panel=w.plot.terrain;c=panel.canvas
        geometry=c.relative_m.copy();origin=c.origin_xyz_m.copy()
        panel.expand_button.setChecked(True);app.processEvents();c.grab()
        w.grab().save(str(output/'expanded_overview.png'))
        choices=np.flatnonzero(c._proj_category==3)
        assert len(choices)>0
        point=c._proj_screen[choices[len(choices)//2]]
        position=QtCore.QPoint(int(round(point[0])),int(round(point[1])))
        QtTest.QTest.mouseClick(c,QtCore.Qt.LeftButton,pos=position)
        assert c.selected_pixel_id is not None
        selected=int(np.flatnonzero(c.sample_pixel_ids==c.selected_pixel_id)[0])
        QtTest.QTest.keyClick(c,QtCore.Qt.Key_F);app.processEvents();c.grab()
        assert c.orbit_focus_id==c.selected_pixel_id
        expected=np.asarray([c.plot_rect().center().x(),c.plot_rect().center().y()])
        np.testing.assert_allclose(screen(c,selected),expected,atol=1e-7)
        initial_zoom=c.zoom
        pan_before=QtCore.QPointF(c.pan)
        QtTest.QTest.keyPress(c,QtCore.Qt.Key_W);QtTest.QTest.qWait(90);QtTest.QTest.keyRelease(c,QtCore.Qt.Key_W)
        keyboard_delta=c.pan.y()-pan_before.y()
        assert keyboard_delta<0.
        stopped_pan=QtCore.QPointF(c.pan);QtTest.QTest.qWait(35)
        assert c.pan==stopped_pan
        c.focus_selected();app.processEvents()
        w.grab().save(str(output/'focused_detail.png'))
        anchor=screen(c,selected)
        pos=QtCore.QPointF(*anchor)
        for _ in range(60):
            event=QtGui.QWheelEvent(pos,pos,QtCore.QPoint(),QtCore.QPoint(0,120),QtCore.Qt.NoButton,QtCore.Qt.NoModifier,QtCore.Qt.NoScrollPhase,False)
            app.sendEvent(c,event)
        QtTest.QTest.qWait(160)
        self_check=screen(c,selected)
        np.testing.assert_allclose(self_check,anchor,atol=1e-7)
        assert c.zoom==80.
        w.grab().save(str(output/'zoom_80x.png'))
        timings=[]
        for _ in range(24):
            start=time.perf_counter();c.rotate_axis('y',15);c.grab();timings.append((time.perf_counter()-start)*1000)
        np.testing.assert_array_equal(geometry,c.relative_m)
        np.testing.assert_array_equal(origin,c.origin_xyz_m)
        report={'native_click_and_F':True,'focus_pixel_id':c.selected_pixel_id,
            'initial_focus_zoom':initial_zoom,'zoom_limit':c.zoom,'cursor_anchor_error_px':float(np.linalg.norm(self_check-anchor)),
            'sample_count':len(c.vertices),'render_median_ms':float(np.median(timings)),
            'render_p95_ms':float(np.percentile(timings,95)),
            'geometry_unchanged':True,'endpoint_origin_unchanged':True,
            'W_pan_delta_pixels':keyboard_delta,'key_release_stops':True,'smooth_wheel_step':c.WHEEL_STEP,
            'inline_canvas_size':[c.width(),c.height()],'new_windows_created':False}
        panel.expand_button.setChecked(False);app.processEvents()
        assert w.image.isVisible() and w.table.isVisible()
        report['collapse_restores_layout']=True
        write_json(output/'validation_report.json',report);print(report,flush=True)
        if '--keep-open' in sys.argv:
            c.reset_view();c.selected_pixel_id=report['focus_pixel_id'];panel.expand_button.setChecked(True)
            app.processEvents();c.focus_selected();c.setFocus()
        else:
            w.close();app.quit()
    except Exception:
        traceback.print_exc();w.close();app.quit()

timer.timeout.connect(probe);timer.start()
sys.exit(app.exec_())
