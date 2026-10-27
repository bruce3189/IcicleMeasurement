import os,sys
os.environ.setdefault('QT_QPA_PLATFORM','offscreen');sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'Measurement'))
from PyQt5 import QtWidgets
from icicle_measure.gui import MainWindow
from icicle_measure.simulator import generate
from icicle_measure.core import measure_frame
app=QtWidgets.QApplication([]);s=generate('edge');w=MainWindow();w.set_input(s.frame,s.detections,s.intrinsics,'edge')
r=measure_frame(s.frame,s.detections,s.intrinsics,w.config)
for o in r.objects:
 print('object',o.id,o.status,o.reason)
 for e in o.endpoints:
  print('endpoint',e.name,e.pixel,e.status,e.depth_m,e.reason,'history',len(e.history))
  if e.history:
   x=e.history[-1]
   print('radius',x.radius,'bounds',x.bounds,'counts',x.near_count,x.far_count,x.target_count,'center',x.center_m,'near',len(x.near_ids),'far',len(x.far_ids),'valid',len(x.valid_ids),'checks',x.checks,'reasons',x.reasons)
