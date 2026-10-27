import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.stdout = sys.stderr = (Path(__file__).parent / 'hover_ui.log').open('w',encoding='utf-8',buffering=1)
import faulthandler
faulthandler.enable()
print('boot')
import numpy as np
from PyQt5 import QtWidgets,QtCore,QtGui
from icicle_measure.gui import MainWindow,configure_font
from icicle_measure.pseudo_gui import PseudoDepthDialog
from icicle_measure.pseudo_depth import ROOT,load_project
from icicle_measure.io import write_json
print('imports')

app=QtWidgets.QApplication([]); configure_font(app)
print('application')
window=MainWindow(); window.show()
print('main')
dialog=PseudoDepthDialog(window); window.pseudo_dialog=dialog
print('dialog')
dialog.measurement_requested.connect(window.receive_pseudo_input)
folder=ROOT/'outputs/pseudo_depth_refined/002_00002'
dialog.receive({'result':load_project(folder),'root':folder})
print('received')
dialog.resize(1366,768); dialog.show(); dialog.raise_()
dialog.view_kind.setCurrentIndex(dialog.view_kind.findData('complete'))
dialog.panel.setCurrentIndex(2)

def probe():
    print('probe')
    dialog.refresh(fit=True); app.processEvents()
    ids=np.argwhere(dialog.result.instance_map==1)
    v,u=ids[len(ids)//2]
    position=dialog.view.mapFromScene(QtCore.QPointF(float(u)+.25,float(v)+.25))
    dialog.activateWindow()
    QtGui.QCursor.setPos(dialog.view.viewport().mapToGlobal(position))
    app.processEvents()
    move=QtGui.QMouseEvent(QtCore.QEvent.MouseMove,QtCore.QPointF(position),QtCore.Qt.NoButton,QtCore.Qt.NoButton,QtCore.Qt.NoModifier)
    app.sendEvent(dialog.view.viewport(),move)
    actual=dialog.view.pixel_at(position)
    assert '像素 (%d, %d)' % actual in dialog.hover_readout.text()
    assert '当前图层：complete' in dialog.pixel_details.toPlainText()
    dialog.grab().save(str(folder/'hover_details_verified.png'))
    write_json(folder/'hover_validation.json',{'native_mouse_event':True,'pixel':actual,
        'layer':'complete','readout':dialog.hover_readout.text()})
    if '--close' in sys.argv:
        dialog.close(); window.close(); app.quit()

def checked_probe():
    try:
        probe()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        app.quit()

QtCore.QTimer.singleShot(800,checked_probe)
sys.exit(app.exec_())
