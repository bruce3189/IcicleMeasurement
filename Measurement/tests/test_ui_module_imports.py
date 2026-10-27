"""The historical entrypoints must still expose the exact component objects."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


class UiModuleImportTests(unittest.TestCase):
    def test_legacy_imports_are_component_aliases(self):
        from icicle_measure import gui,pseudo_gui
        from icicle_measure.ui import common,charts,cloud_canvas,cloud_panel,diagnostics,pseudo_preview,pseudo_workspace
        for component,names in [(common,['ImageView','ConfigDialog','configure_font','STATUS']),
                (charts,['PlotCanvas','HistogramCanvas','WindowCountsCanvas']),
                (cloud_canvas,['DepthTerrainCanvas']),(cloud_panel,['DepthTerrainPanel']),
                (diagnostics,['MeasurementCriteriaPanel','DiagnosticsPlot'])]:
            for name in names:self.assertIs(getattr(gui,name),getattr(component,name))
        for component,names in [(pseudo_preview,['scene_preview','EditView']),
                (pseudo_workspace,['PseudoTask','PseudoDepthPanel','PseudoDepthDialog'])]:
            for name in names:self.assertIs(getattr(pseudo_gui,name),getattr(component,name))

    def test_imports_do_not_load_optional_inference_or_camera_runtimes(self):
        script="import json,sys;import icicle_measure.gui;import icicle_measure.pseudo_gui;print(json.dumps([n for n in ('torch','timm','pyrealsense2','open3d','pyvista') if n in sys.modules]))"
        result=subprocess.run([sys.executable,'-c',script],cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout.strip()),[])
