"""Pseudo-depth preview rendering and pixel-accurate editing view."""
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..pseudo_depth import fixed_depth_rgb


def scene_preview(result, kind='masks', selected=0):
    s, cfg = result.scene, result.config
    if kind == 'rgb':
        image = s.rgb.copy()
    elif kind == 'relative':
        q = s.relative
        lo, hi = np.percentile(q[s.valid], [2, 98])
        gray = (np.clip((q - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        image = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        image[~s.valid] = 0
    elif kind == 'model':
        from ..pseudo_depth import _map_relative
        z, _ = _map_relative(s.relative, s.valid, cfg)
        image = fixed_depth_rgb(z, cfg.color_min_m, cfg.color_max_m)
    elif kind in ('complete', 'observed', 'background'):
        z = {'complete': result.clean_frame.depth_m, 'observed': result.frame.depth_m,
             'background': result.background_m}[kind]
        image = fixed_depth_rgb(z, cfg.color_min_m, cfg.color_max_m)
    elif kind == 'uncertainty':
        image = (s.rgb.astype(float) * .28).astype(np.uint8)
        for bit, color in [(2, (60, 160, 220)), (4, (255, 130, 40)), (8, (235, 45, 220)),
                            (16, (255, 255, 255)), (32, (255, 195, 35)), (64, (255, 80, 75))]:
            image[(result.uncertainty & bit) > 0] = color
    else:
        image = (s.rgb.astype(float) * .55).astype(np.uint8)
        for i, mask in enumerate(s.masks):
            hue = np.uint8([[[int((i * 49) % 180), 180, 250]]])
            color = cv2.cvtColor(hue, cv2.COLOR_HSV2RGB)[0, 0]
            image[mask] = (.45 * image[mask] + .55 * color).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (75, 255, 135) if i == selected else tuple(map(int, color)),
                             2 if i == selected else 1)
        image[result.uncertainty & 8 > 0] = (240, 55, 220)
    if kind in ('rgb', 'masks'):
        for i, d in enumerate(s.detections):
            x1, y1, x2, y2 = [round(v) for v in d.bbox]
            color = (95, 255, 140) if i == selected else (170, 190, 200)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, '#' + d.id, (max(0, x1), max(32, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
            for j, point in enumerate(d.keypoints):
                if point is not None:
                    cv2.circle(image, tuple(round(v) for v in point), 2, (255, 75, 65) if j == 0 else (75, 210, 255), -1)
    return image


class EditView(QtWidgets.QGraphicsView):
    stroke = QtCore.pyqtSignal(object)
    point_added = QtCore.pyqtSignal(object)
    stroke_finished = QtCore.pyqtSignal()
    pixel_hovered = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.pixmap_item = self.scene().addPixmap(QtGui.QPixmap())
        self.setBackgroundBrush(QtGui.QColor('#142231'))
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setResizeAnchor(self.AnchorViewCenter)
        self.setDragMode(self.ScrollHandDrag)
        self.mode, self.previous, self.shape = 'inspect', None, None
        self.setMinimumSize(470, 200)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

    def set_image(self, image, fit=False):
        h, w = image.shape[:2]
        qimage = QtGui.QImage(image.data, w, h, image.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.scene().setSceneRect(0, 0, w, h)
        self.shape = (h, w)
        if fit:
            self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def set_mode(self, mode):
        self.mode = mode
        self.setDragMode(self.ScrollHandDrag if mode == 'inspect' else self.NoDrag)
        self.setCursor(QtCore.Qt.CrossCursor if mode != 'inspect' else QtCore.Qt.ArrowCursor)

    def pixel(self, event):
        return self.pixel_at(event.pos())

    def pixel_at(self, position):
        p = self.mapToScene(position)
        if self.shape is None or not (0 <= p.x() < self.shape[1] and 0 <= p.y() < self.shape[0]):
            return None
        # Image items occupy cells [u,u+1); rounding picks the wrong neighbour
        # and can produce width/height at the final half-pixel.
        return (int(np.floor(p.x())), int(np.floor(p.y())))

    def mousePressEvent(self, event):
        point = self.pixel(event)
        if event.button() == QtCore.Qt.LeftButton and point and self.mode != 'inspect':
            if self.mode in ('positive', 'negative', 'plane'):
                self.point_added.emit({'x': point[0], 'y': point[1], 'positive': self.mode == 'positive', 'mode': self.mode})
            else:
                self.previous = point
                self.stroke.emit({'start': point, 'end': point, 'mode': self.mode})
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.pixel_hovered.emit(self.pixel(event))
        if self.previous is not None:
            point = self.pixel(event)
            if point:
                self.stroke.emit({'start': self.previous, 'end': point, 'mode': self.mode})
                self.previous = point
            event.accept()
            return
        super().mouseMoveEvent(event)

    def viewportEvent(self, event):
        if event.type() == QtCore.QEvent.Leave:
            self.pixel_hovered.emit(None)
        return super().viewportEvent(event)

    def mouseReleaseEvent(self, event):
        if self.previous is not None:
            self.previous = None
            self.stroke_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not delta:
            event.ignore(); return
        current = self.transform().m11()
        factor = float(np.exp(np.clip(delta / 120 * np.log(1.15), -.5, .5)))
        factor = min(8., max(.05, current * factor)) / current
        self.scale(factor, factor)
        self.pixel_hovered.emit(self.pixel_at(event.pos()))
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        event.accept()
