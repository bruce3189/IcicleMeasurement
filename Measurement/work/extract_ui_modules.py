"""One-time mechanical extraction of the pre-refactor UI definitions."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'icicle_measure'
ui=ROOT/'ui'
gui=(ROOT/'gui.py').read_text(encoding='utf-8')
pseudo=(ROOT/'pseudo_gui.py').read_text(encoding='utf-8')


def nodes(source):
    return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}


def extract(source,name,relocate=True):
    node=nodes(source)[name]
    lines=source.splitlines(keepends=True)
    text=''.join(lines[node.lineno-1:node.end_lineno])
    if relocate:
        # Moving one package level deeper requires only local import levels to change.
        imports=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.ImportFrom) and n.level==1]
        local=text.splitlines(keepends=True)
        for item in imports:
            local[item.lineno-1]=local[item.lineno-1].replace('from .','from ..',1)
        text=''.join(local)
    return text.rstrip()+'\n'


def write(name,header,source,names):
    body='\n\n'.join(extract(source,key) for key in names)
    (ui/name).write_text(header.rstrip()+'\n\n\n'+body,encoding='utf-8')


if 'MainWindow' not in nodes(gui) or 'DepthTerrainCanvas' not in nodes(gui):
    raise RuntimeError('Extraction already applied or unexpected source layout')
ui.mkdir(exist_ok=True)
(ui/'__init__.py').write_text('"""UI components. Optional model and camera runtimes are loaded by their existing entrypoints."""\n',encoding='utf-8')
status=next(n for n in ast.parse(gui).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STATUS' for t in n.targets))
status_text=''.join(gui.splitlines(keepends=True)[status.lineno-1:status.end_lineno])
write('common.py','''"""Shared fonts, image view and algorithm configuration dialog."""
import json
import os
from pathlib import Path
from dataclasses import asdict
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..models import Config
\n'''+status_text,gui,['configure_font','ImageView','ConfigDialog'])
write('charts.py','''"""Depth distribution, history and sample-count canvases."""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..models import Config''',gui,['PlotCanvas','HistogramCanvas','WindowCountsCanvas'])
write('cloud_canvas.py','''"""Point-cloud geometry display, painting and mouse/keyboard navigation."""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..geometry import build_cloud, project as project_points''',gui,['DepthTerrainCanvas'])
write('cloud_panel.py','''"""Controls and detail readout around the stable Qt point-cloud canvas."""
import numpy as np
from PyQt5 import QtCore, QtWidgets
from .cloud_canvas import DepthTerrainCanvas''',gui,['DepthTerrainPanel'])
write('diagnostics.py','''"""Measurement decision table and movable diagnostic chart tabs."""
from PyQt5 import QtCore, QtGui, QtWidgets
from ..models import Config
from .common import STATUS
from .charts import PlotCanvas, HistogramCanvas, WindowCountsCanvas
from .cloud_panel import DepthTerrainPanel''',gui,['MeasurementCriteriaPanel','DiagnosticsPlot'])
write('pseudo_preview.py','''"""Pseudo-depth preview rendering and pixel-accurate editing view."""
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..pseudo_depth import fixed_depth_rgb''',pseudo,['scene_preview','EditView'])
write('pseudo_workspace.py','''"""Editable pseudo-depth workspace and its original background task lifecycle."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..pseudo_depth import (ROOT, DEFAULT_RGB_DIR, DEFAULT_LABEL_DIR, SAMPLE_STEMS,
    PseudoConfig, compose_scene, scene_from_inference, export_project, load_project, fixed_depth_rgb, depth_colorbar)
from ..pseudo_models import run_inference
from ..pixel_inspector import inspect_pixel, pixel_text
from ..render import draw_selected_distance, distance_summary
from .pseudo_preview import scene_preview, EditView''',pseudo,['PseudoTask','PseudoDepthPanel','PseudoDepthDialog'])

# The old modules remain real import entrypoints; aliases add no per-frame calls.
gui_prefix=''.join(gui.splitlines(keepends=True)[:status.lineno-1])
aliases='''from .ui.common import STATUS, configure_font, ImageView, ConfigDialog
from .ui.charts import PlotCanvas, HistogramCanvas, WindowCountsCanvas
from .ui.cloud_canvas import DepthTerrainCanvas
from .ui.cloud_panel import DepthTerrainPanel
from .ui.diagnostics import MeasurementCriteriaPanel, DiagnosticsPlot
'''
remaining='\n\n'.join(extract(gui,name,False) for name in ['Worker','PoseMeasurementWorker','MainWindow','main'])
(ROOT/'gui.py').write_text(gui_prefix.rstrip()+'\n\n'+aliases+'\n\n'+remaining,encoding='utf-8')
pseudo_prefix=''.join(pseudo.splitlines(keepends=True)[:nodes(pseudo)['scene_preview'].lineno-1])
aliases='''from .ui.pseudo_preview import scene_preview, EditView
from .ui.pseudo_workspace import PseudoTask, PseudoDepthPanel, PseudoDepthDialog
'''
(ROOT/'pseudo_gui.py').write_text(pseudo_prefix.rstrip()+'\n\n'+aliases,encoding='utf-8')
print('Extracted UI definitions with unchanged bodies; legacy import modules retained.')
