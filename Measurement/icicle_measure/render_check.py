# -*- coding: utf-8 -*-
"""Render the repaired 3D point cloud offscreen for visual verification.

The widget is painted into a QImage instead of using QWidget.grab(), because
grab() needs a live window backend that is unavailable in this headless shell.
"""
import os
import sys

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5 import QtGui, QtWidgets

from icicle_measure.core import Strategies, endpoint_estimate
from icicle_measure.gui import DepthTerrainCanvas
from icicle_measure.models import Config
from icicle_measure.simulator import generate

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'pointcloud_verify')
WIDTH, HEIGHT = 900, 660
os.makedirs(OUT, exist_ok=True)


def ensure_font():
    """The headless shell exposes no font families; load a CJK face explicitly."""
    for path in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf'):
        if os.path.exists(path):
            fid = QtGui.QFontDatabase.addApplicationFont(path)
            fams = QtGui.QFontDatabase.applicationFontFamilies(fid)
            if fams:
                app = QtWidgets.QApplication.instance()
                app.setFont(QtGui.QFont(fams[0], 9))
                return fams[0]
    return None


app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
print('font:', ensure_font())

sim = generate('edge', seed=42)
frame, detection, intrinsics = sim.frame, sim.detections[0], sim.intrinsics
endpoint = endpoint_estimate(frame, detection, 0, intrinsics, Config(), Strategies(), lambda: False)

canvas = DepthTerrainCanvas()
canvas.set_data(endpoint, frame.depth_m, detection, intrinsics)
canvas.resize(WIDTH, HEIGHT)


def shot(tag):
    image = QtGui.QImage(WIDTH, HEIGHT, QtGui.QImage.Format_ARGB32)
    image.fill(QtGui.QColor('#ffffff'))
    painter = QtGui.QPainter(image)
    canvas.render(painter)
    painter.end()
    path = os.path.join(OUT, '点云_%s.png' % tag)
    image.save(path)
    print('saved', path)


CASES = (('透视_真实等比', 'perspective', 'true', 'local'),
         ('透视_深度夸张x4', 'perspective', 'exaggerated', 'local'),
         ('正交_真实等比', 'orthographic', 'true', 'local'),
         ('透视_全部点', 'perspective', 'true', 'all'))

for tag, projection, depth_mode, focus in CASES:
    canvas.set_focus_mode(focus)
    canvas.set_projection_mode(projection)
    canvas.set_depth_display_mode(depth_mode)
    shot(tag)

canvas.set_focus_mode('local')
canvas.set_projection_mode('perspective')
canvas.set_depth_display_mode('true')
canvas.rotate_axis('y', 35)
shot('透视_旋转35度')
canvas.rotate_axis('y', -35)

print('vertices', canvas.vertices.shape, '| scale=%.3f' % canvas.cloud_scale,
      '| max|Z|=%.3f' % float(abs(canvas.vertices[:, 2]).max()))
