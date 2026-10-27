import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtTest
from icicle_measure.gui import MainWindow,configure_font
from icicle_measure.pseudo_gui import PseudoDepthDialog
from icicle_measure.pseudo_depth import ROOT,load_project
from icicle_measure.io import read_image,write_json

app=QtWidgets.QApplication([]); configure_font(app)
w=MainWindow(); w.show()
d=PseudoDepthDialog(w); w.pseudo_dialog=d
d.measurement_requested.connect(w.receive_pseudo_input)
directory=ROOT/'outputs/pseudo_depth_refined'
checks=[]
for stem in ('002_00002','002_00041'):
    folder=directory/stem; r=load_project(folder)
    for name,raw,frame in [('complete',r.clean_raw,r.clean_frame),('observed',r.observed_raw,r.frame)]:
        np.testing.assert_array_equal(raw,read_image(folder/('depth_'+name+'_z16.png'),-1))
        np.testing.assert_array_equal(frame.depth_m,np.load(folder/('depth_'+name+'_m.npy')))
        assert np.all(raw[~r.scene.valid]==0)
    d.receive({'result':r,'root':folder}); d.resize(1366,768); d.show(); app.processEvents()
    for kind in ('rgb','masks','model','background','complete','observed','uncertainty'):
        d.view_kind.setCurrentIndex(d.view_kind.findData(kind)); app.processEvents()
    d.object_choice.setCurrentIndex(1); app.processEvents(); d.object_choice.setCurrentIndex(0)
    d.view_kind.setCurrentIndex(d.view_kind.findData('complete')); app.processEvents()
    d.refresh(fit=True); app.processEvents()
    d.grab().save(str(folder/'gui_verified.png'))
    d.send_measurement()
    deadline=QtCore.QDeadlineTimer(15000)
    while w.thread and not deadline.hasExpired():
        app.processEvents(); QtTest.QTest.qWait(10)
    assert w.thread is None and w.frame.metadata['is_synthetic']
    checks.append({'stem':stem,'native_gui':True,'roundtrip_bit_exact':True,'measurement_source':'pseudo_rgbd'})
write_json(directory/'validation_report.json',{'samples':checks,'real_camera_accuracy_validated':False})
if '--keep-open' in sys.argv:
    d.choose_sample('002_00002')
    sys.exit(app.exec_())
d.close(); w.close(); app.processEvents()
