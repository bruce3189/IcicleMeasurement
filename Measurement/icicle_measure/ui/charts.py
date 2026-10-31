"""Depth distribution, history and sample-count canvases."""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..models import Config


class PlotCanvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.endpoint = None
        self.config = Config()
        self.depth = None
        self.detection = None
        self.interactive_enabled = self.__class__ is PlotCanvas
        self._view_xlim = None
        self._data_xlim = None
        self._plot_rect = None
        self._hover_samples = []
        self._drag_origin = None
        self._drag_xlim = None
        self.sample_count = 0
        self.cluster_counts = {'near': 0, 'far': 0, 'target': 0, 'outlier': 0}
        self.setMinimumHeight(260)
        self.setAutoFillBackground(True)
        if self.interactive_enabled:
            self.setMouseTracking(True)

    def set_data(self, endpoint, depth, detection=None):
        if endpoint is not self.endpoint or depth is not self.depth:
            self._view_xlim = None
            self._hover_samples = []
        self.endpoint, self.depth, self.detection = endpoint, depth, detection
        record = endpoint.history[-1] if endpoint and endpoint.history else None
        self.sample_count = len(record.valid_ids) if record else 0
        self.cluster_counts = ({'near': record.near_count, 'far': record.far_count,
                                'target': record.target_count,
                                'outlier': max(0, len(record.valid_ids) - len(record.near_ids) - len(record.far_ids))}
                               if record else {'near': 0, 'far': 0, 'target': 0, 'outlier': 0})
        self.update()

    def _clamp_view(self, low, high):
        if self._data_xlim is None:
            return low, high
        data_low, data_high = self._data_xlim
        span = min(max(high - low, (data_high - data_low) * .01), data_high - data_low)
        center = (low + high) * .5
        low, high = center - span * .5, center + span * .5
        if low < data_low:
            high += data_low - low
            low = data_low
        if high > data_high:
            low -= high - data_high
            high = data_high
        return low, high

    def wheelEvent(self, event):
        if (not self.interactive_enabled or self._plot_rect is None
                or not self._plot_rect.contains(QtCore.QPointF(event.pos()))):
            event.ignore()
            return
        low, high = self._view_xlim or self._data_xlim
        rect = self._plot_rect
        anchor = low + (event.pos().x() - rect.left()) / rect.width() * (high - low)
        factor = .8 if event.angleDelta().y() > 0 else 1.25
        new_span = min(max((high - low) * factor, (self._data_xlim[1] - self._data_xlim[0]) * .01),
                       self._data_xlim[1] - self._data_xlim[0])
        fraction = (anchor - low) / (high - low)
        self._view_xlim = self._clamp_view(anchor - fraction * new_span,
                                            anchor + (1 - fraction) * new_span)
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if (self.interactive_enabled and event.button() == QtCore.Qt.LeftButton
                and self._plot_rect is not None
                and self._plot_rect.contains(QtCore.QPointF(event.pos()))):
            self._drag_origin = event.pos()
            self._drag_xlim = self._view_xlim or self._data_xlim
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.interactive_enabled or self._plot_rect is None:
            return
        if self._drag_origin is not None:
            span = self._drag_xlim[1] - self._drag_xlim[0]
            shift = (event.pos().x() - self._drag_origin.x()) / self._plot_rect.width() * span
            self._view_xlim = self._clamp_view(self._drag_xlim[0] - shift, self._drag_xlim[1] - shift)
            self.update()
            return
        closest = None
        closest_distance = 8.0
        for sample_x, sample_y, pixel_id, depth_value, label in self._hover_samples:
            distance = ((sample_x - event.pos().x()) ** 2 + (sample_y - event.pos().y()) ** 2) ** .5
            if distance < closest_distance:
                closest = (pixel_id, depth_value, label)
                closest_distance = distance
        if closest:
            pixel_id, depth_value, label = closest
            self.setToolTip('%s\n深度 Z：%.5f m\n像素 (x=%d, y=%d)' %
                            (label, depth_value, pixel_id % self.depth.shape[1], pixel_id // self.depth.shape[1]))
        else:
            self.setToolTip('滚轮缩放深度轴 · 左键拖动平移 · 双击恢复全范围')

    def mouseReleaseEvent(self, event):
        if self._drag_origin is not None and event.button() == QtCore.Qt.LeftButton:
            self._drag_origin = None
            self._drag_xlim = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.interactive_enabled:
            self._view_xlim = None
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def panel(self, painter, rect, title):
        painter.setPen(QtGui.QColor('#dce5ef'))
        painter.setBrush(QtGui.QColor('#ffffff'))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QtGui.QColor('#526175'))
        painter.drawText(int(rect.left() + 14), int(rect.top() + 22), title)

    def draw_axes(self, painter, rect, x_label, y_label, xmin, xmax, ymin, ymax):
        painter.save()
        painter.setPen(QtGui.QPen(QtGui.QColor('#e6ebf1'), 1))
        for k in range(1, 5):
            y = rect.bottom() - k * rect.height() / 5
            painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
        painter.setPen(QtGui.QPen(QtGui.QColor('#aab7c5'), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        painter.setPen(QtGui.QColor('#78869a'))
        painter.drawText(int(rect.left()), int(rect.bottom() + 18), '%.3g' % xmin)
        painter.drawText(int(rect.right() - 36), int(rect.bottom() + 18), '%.3g' % xmax)
        y_format = '%.0f' if '像素数' in y_label else '%.4f'
        painter.drawText(int(rect.left() - 48), int(rect.top() + 4), y_format % ymax)
        painter.drawText(int(rect.left() - 48), int(rect.bottom()), y_format % ymin)
        painter.drawText(int(rect.center().x() - 48), int(rect.bottom() + 38), x_label)
        painter.save()
        painter.translate(int(rect.left() - 55), int(rect.center().y()))
        painter.rotate(-90)
        painter.drawText(-50, 0, y_label)
        painter.restore()
        painter.restore()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor('#f4f7fb'))
        if not self.endpoint or not self.endpoint.history:
            p.setPen(QtGui.QColor('#8794a7'))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, '选择一个端点，查看该窗口内每个有效像素样本')
            return
        record = self.endpoint.history[-1]
        p.setPen(QtGui.QColor('#253448'))
        if self.detection is None or not record.valid_ids:
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, '该窗口没有可绘制的有效深度样本')
            return
        ids = np.asarray(record.valid_ids, dtype=np.int64)
        z = self.depth.ravel()[ids].astype(float)
        near_ids = set(map(int, record.near_ids))
        far_ids = set(map(int, record.far_ids))
        target_ids = set(map(int, record.target_ids))
        valid_ids = set(map(int, record.valid_ids))
        # Make rows mutually exclusive: the target component is a subset of
        # the near cluster, so subtract it before drawing the remaining near samples.
        near_only_ids = near_ids - target_ids - far_ids
        groups = [
            ('目标簇', target_ids, '#17a77a'),
            ('近簇其它', near_only_ids, '#edae48'),
            ('远簇', far_ids - target_ids, '#4f7df3'),
            ('未分簇', valid_ids - near_ids - far_ids - target_ids, '#aab4c1'),
        ]
        data_min, data_max = float(z.min()), float(z.max())
        span = data_max - data_min
        margin = max(span * .08, .001)
        self._data_xlim = (data_min - margin, data_max + margin)
        if self._view_xlim is None:
            self._view_xlim = self._data_xlim
        xmin, xmax = self._clamp_view(*self._view_xlim)
        plot_left = 116
        plot_right = max(plot_left + 80, self.width() - 30)
        chart_top = 76
        chart_bottom = max(chart_top + 80, self.height() - 66)
        rect = QtCore.QRectF(plot_left, chart_top, plot_right - plot_left, chart_bottom - chart_top)
        self._plot_rect = rect
        self.panel(p, self.rect().adjusted(4, 4, -4, -4), '深度簇分布 · 最终窗口半径 r=%d px' % record.radius)
        p.setFont(QtGui.QFont(self.font().family(), 8))
        p.setPen(QtGui.QColor('#68778b'))
        p.drawText(QtCore.QRectF(rect.left(), 38, rect.width(), 34),
                   QtCore.Qt.TextWordWrap, '经验频数轮廓（非正态拟合）· 峰高独立缩放；数量看 n')

        # Depth ticks and light vertical guides make cluster separation easy to read.
        tick_count = 5
        p.setPen(QtGui.QPen(QtGui.QColor('#e6ebf1'), 1))
        for k in range(tick_count + 1):
            x = rect.left() + rect.width() * k / tick_count
            p.drawLine(QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom()))
            p.setPen(QtGui.QColor('#78869a'))
            p.drawText(int(x - 23), int(rect.bottom() + 18), '%.3f' % (xmin + (xmax - xmin) * k / tick_count))
            p.setPen(QtGui.QPen(QtGui.QColor('#e6ebf1'), 1))
        p.setPen(QtGui.QPen(QtGui.QColor('#aab7c5'), 1))
        p.drawLine(rect.bottomLeft(), rect.bottomRight())
        p.setPen(QtGui.QColor('#78869a'))
        p.drawText(int(rect.center().x() - 38), int(rect.bottom() + 38), '深度 Z (m)')
        lane_h = rect.height() / len(groups)
        sample_index = {int(pixel_id): i for i, pixel_id in enumerate(ids)}
        for lane, (label, members, color) in enumerate(groups):
            center_y = rect.top() + lane_h * (lane + .5)
            p.setPen(QtGui.QColor('#536277'))
            p.drawText(16, int(center_y + 4), label)
            p.setPen(QtGui.QPen(QtGui.QColor('#edf1f5'), 1))
            p.drawLine(QtCore.QPointF(rect.left(), rect.top() + lane_h * (lane + 1)),
                       QtCore.QPointF(rect.right(), rect.top() + lane_h * (lane + 1)))
            p.setPen(QtGui.QColor('#68778b'))
            p.drawText(78, int(center_y + 4), 'n=%d' % len(members))

        # Draw an empirical ridge for each category. This is a smoothed sample
        # frequency profile, not a fitted or assumed normal distribution.
        # Small rug marks retain the location of every individual sample.
        p.save()
        p.setClipRect(rect.adjusted(-2, -2, 2, 2))
        bin_count = max(48, min(120, int(rect.width() * .45)))
        bin_edges = np.linspace(xmin, xmax, bin_count + 1)
        kernel = np.array([1, 2, 3, 2, 1], dtype=float)
        kernel /= kernel.sum()
        profiles = []
        for label, members, color in groups:
            values = np.asarray([z[sample_index[i]] for i in members if i in sample_index], dtype=float)
            counts, _ = np.histogram(values, bins=bin_edges)
            smooth = np.convolve(counts.astype(float), kernel, mode='same') if len(values) else counts.astype(float)
            profiles.append((values, smooth, color, len(members)))
        self._hover_samples = []
        for lane, ((label, members, color), profile_data) in enumerate(zip(groups, profiles)):
            values, smooth, color, count = profile_data
            lane_top = rect.top() + lane_h * lane
            baseline = lane_top + lane_h * .80
            max_height = lane_h * .62
            peak = max(float(smooth.max()), 1.0) if len(smooth) else 1.0
            centers = (bin_edges[:-1] + bin_edges[1:]) * .5
            ridge = QtGui.QPainterPath()
            ridge.moveTo(rect.left(), baseline)
            for depth_value, density in zip(centers, smooth):
                x = rect.left() + (depth_value - xmin) / (xmax - xmin) * rect.width()
                y = baseline - float(density) / peak * max_height
                ridge.lineTo(x, y)
            ridge.lineTo(rect.right(), baseline)
            ridge.closeSubpath()
            color_value = QtGui.QColor(color)
            fill = QtGui.QColor(color_value)
            fill.setAlpha(58)
            p.setPen(QtGui.QPen(color_value, 1.5))
            p.setBrush(fill)
            p.drawPath(ridge)
            p.setBrush(QtCore.Qt.NoBrush)

            # One short tick per observed depth sample, at the lane baseline.
            p.setPen(QtGui.QPen(color_value, 1, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            for depth_value in values:
                if depth_value < xmin or depth_value > xmax:
                    continue
                x = rect.left() + (depth_value - xmin) / (xmax - xmin) * rect.width()
                p.drawLine(QtCore.QPointF(x, baseline + 1), QtCore.QPointF(x, baseline + 4))
            for pixel_id in members:
                i = sample_index.get(pixel_id)
                if i is None or z[i] < xmin or z[i] > xmax:
                    continue
                sample_x = rect.left() + (z[i] - xmin) / (xmax - xmin) * rect.width()
                self._hover_samples.append((sample_x, baseline + 2, pixel_id, float(z[i]), label))
        p.restore()

        # Mark the arithmetic mean used by the estimator as the primary reference.
        target_values = [z[sample_index[i]] for i in target_ids if i in sample_index]
        mean_depth = float(np.mean(target_values)) if target_values else None
        if mean_depth is not None:
            mean_x = rect.left() + (mean_depth - xmin) / (xmax - xmin) * rect.width()
            p.setPen(QtGui.QPen(QtGui.QColor('#087957'), 2, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(mean_x, rect.top()), QtCore.QPointF(mean_x, rect.bottom()))
            label_x = min(max(int(mean_x + 5), int(rect.left() + 2)), int(rect.right() - 112))
            p.setPen(QtGui.QColor('#087957'))
            p.setFont(QtGui.QFont(self.font().family(), 8, QtGui.QFont.Bold))
            p.drawText(label_x, int(rect.top() - 5), '%s %.4f m' % (
                '通过估深' if self.endpoint.status == 'accepted' else '候选均值', mean_depth))
        p.setPen(QtGui.QColor('#68778b'))
        p.setFont(QtGui.QFont(self.font().family(), 8))
        p.drawText(QtCore.QRectF(18, self.height() - 35, self.width() - 36, 32),
                   QtCore.Qt.TextWordWrap, '有效样本 %d · 短线=像素样本（可能重叠）· 越靠右，光轴深度越大' % len(ids))


class HistogramCanvas(PlotCanvas):
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor('#f4f7fb'))
        if not self.endpoint or not self.endpoint.history or self.depth is None:
            p.setPen(QtGui.QColor('#8794a7'))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, '选中端点查看深度簇与窗口中心变化')
            return
        record = self.endpoint.history[-1]
        left = QtCore.QRectF(68, 48, max(90, self.width() / 2 - 108), max(90, self.height() - 140))
        right = QtCore.QRectF(self.width() / 2 + 60, 48, max(90, self.width() / 2 - 97), max(90, self.height() - 140))
        self.panel(p, self.rect().adjusted(4, 4, -4, -4), '左：深度频数　|　右：扩窗候选均值')
        z = self.depth.ravel()
        vals = z[record.valid_ids] if record.valid_ids else np.array([])
        if len(vals):
            lo, hi = float(vals.min()), float(vals.max())
            hi = max(hi, lo + .001)
            bins = np.linspace(lo, hi, 41)
            all_counts = np.histogram(vals, bins)[0]
            ymax = max(int(all_counts.max()), 1)
            self.draw_axes(p, left, '深度 Z (m)', '像素数', lo, hi, 0, ymax)
            near, far, target = map(set, (record.near_ids, record.far_ids, record.target_ids))
            groups = ((set(record.valid_ids) - near - far - target, '#aab4c1'),
                      (far - target, '#4f7df3'), (near - target - far, '#edae48'),
                      (target, '#17a77a'))
            base_counts = np.zeros(40)
            for ids, color in groups:
                counts = np.histogram(z[list(ids)], bins)[0] if ids else np.zeros(40)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(color))
                for k, count in enumerate(counts):
                    height = left.height() * count / ymax
                    p.drawRect(QtCore.QRectF(left.left() + k * left.width() / 40,
                                            left.bottom() - left.height() * (base_counts[k] + count) / ymax,
                                            left.width() / 40 + .2, height))
                base_counts += counts
        records = [r for r in self.endpoint.history if r.center_m is not None]
        if records:
            ymin, ymax = min(r.center_m for r in records), max(r.center_m for r in records)
            pad = max((ymax - ymin) * .2, .002)
            ymin, ymax = ymin - pad, ymax + pad
            self.draw_axes(p, right, '窗口半径 (px)', '目标深度均值 (m)', 0,
                           max(r.radius for r in self.endpoint.history), ymin, ymax)
            path = QtGui.QPainterPath()
            p.setPen(QtGui.QPen(QtGui.QColor('#17a77a'), 2))
            p.setBrush(QtCore.Qt.NoBrush)
            for k, r in enumerate(records):
                pt = QtCore.QPointF(right.left() + r.radius / max(x.radius for x in self.endpoint.history) * right.width(),
                                   right.bottom() - (r.center_m - ymin) / (ymax - ymin) * right.height())
                path.moveTo(pt) if k == 0 else path.lineTo(pt)
                p.drawEllipse(pt, 3, 3)
            p.drawPath(path)
        p.setPen(QtGui.QColor('#65758a'))
        p.setFont(QtGui.QFont(self.font().family(), 8))
        p.drawText(QtCore.QRectF(18, self.height() - 35, self.width() - 36, 32),
                   QtCore.Qt.TextWordWrap, '绿 目标候选 · 橙 近簇其它 · 蓝 远簇 · 灰 未分簇；均值纵轴自动缩放，平稳不等于通过测距')


class WindowCountsCanvas(PlotCanvas):
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor('#f4f7fb'))
        if not self.endpoint or not self.endpoint.history:
            p.setPen(QtGui.QColor('#8794a7'))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, '计算后显示每次扩窗的簇样本数')
            return
        records = self.endpoint.history
        self.panel(p, self.rect().adjusted(4, 4, -4, -4), '逐窗口样本数 · 每组柱形对应一个半径，数值为 MAD 去噪后')
        chart = QtCore.QRectF(76, 54, max(100, self.width() - 108), max(90, self.height() - 140))
        max_n = max(max(r.near_count, r.far_count, r.target_count) for r in records)
        max_n = max(max_n, self.config.full_samples, 1) * 1.18
        xmax = max(r.radius for r in records)
        self.draw_axes(p, chart, '搜索半径 r (px)', '去噪后像素数', 0, xmax + 1, 0, max_n)
        thresholds = ([(self.config.near_min_exclusive+1, '近簇最少', '#d9534f'),
                       (self.config.far_min_exclusive+1, '远簇最少', '#68778b')]
                      if self.config.endpoint_depth_mode == 'near_cluster_simple' else
                      [(self.config.min_samples, '近簇下限', '#d9534f')])
        if self.config.endpoint_depth_mode != 'near_cluster_simple':
            thresholds.append((self.config.full_samples, '历史样本评分满分', '#68778b'))
        for threshold, label, color in thresholds:
            y = chart.bottom() - threshold / max_n * chart.height()
            p.setPen(QtGui.QPen(QtGui.QColor(color), 1, QtCore.Qt.DashLine))
            p.drawLine(QtCore.QPointF(chart.left(), y), QtCore.QPointF(chart.right(), y))
            p.drawText(int(chart.left() + 4), int(y - 4), '%s %d 点' % (label, threshold))
        series = [('近簇', 'near_count', '#edae48'), ('远簇', 'far_count', '#4f7df3'),
                  ('目标簇', 'target_count', '#17a77a')]
        group_width = min(24.0, chart.width() / max(len(records), 1) * .5)
        bar_width = max(2.0, group_width / 3.4)
        for record in records:
            center_x = chart.left() + record.radius / (xmax + 1) * chart.width()
            for series_index, (label, attr, color) in enumerate(series):
                value = getattr(record, attr)
                bar_x = center_x - group_width / 2 + series_index * (group_width / 3)
                bar_h = value / max_n * chart.height()
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QColor(color))
                p.drawRoundedRect(QtCore.QRectF(bar_x, chart.bottom() - bar_h, bar_width, bar_h), 2, 2)
            p.setPen(QtGui.QColor('#536277'))
            p.drawText(int(center_x - 10), int(chart.top() - 10), 'r=%d' % record.radius)
        p.setPen(QtGui.QColor('#65758a'))
        p.setFont(QtGui.QFont(self.font().family(), 8))
        p.drawText(QtCore.QRectF(18, self.height() - 35, self.width() - 36, 32),
                   QtCore.Qt.TextWordWrap, '橙 近簇总数 · 蓝 远簇 · 绿 目标候选（近簇的子集，不能相加）；三者均须达到硬下限')
