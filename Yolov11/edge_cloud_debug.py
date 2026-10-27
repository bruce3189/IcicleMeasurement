import os, sys
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'Measurement'))
import numpy as np
from PyQt5 import QtWidgets
from icicle_measure.gui import MainWindow, configure_font
from icicle_measure.simulator import generate
from icicle_measure.core import measure_frame
app=QtWidgets.QApplication([]); configure_font(app)
s=generate('edge',seed=42); w=MainWindow(); w.resize(1280,920); w.show(); app.processEvents()
w.set_input(s.frame,s.detections,s.intrinsics,'edge check')
r=measure_frame(s.frame,s.detections,s.intrinsics,w.config); w.receive_result(r); app.processEvents(); w.plot.tabs.setCurrentIndex(0); app.processEvents()
c=w.plot.terrain.canvas
p,_=c._project(c.vertices,1.,c.width()*.54+c.pan.x(),c.height()*.57+c.pan.y())
print('edge point:',s.detections[0].keypoints[0],'bbox:',s.detections[0].bbox)
print('canvas:',c.width(),c.height(),'radius:',c.radius,'result:',r.objects[0].status)
print('samples:',len(c.vertices),'category counts:',np.unique(c.vertex_categories,return_counts=True))
print('screen point bounds:',p.min(axis=0).round(1),p.max(axis=0).round(1),'origin:',[round(c.width()*.54,1),round(c.height()*.57,1)])
path=str(__import__('pathlib').Path(__file__).resolve().parent / 'edge_cloud_debug.png'); print('saved:',c.grab().save(path),path)
w.close(); app.processEvents()
