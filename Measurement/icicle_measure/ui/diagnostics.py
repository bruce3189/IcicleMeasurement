"""Measurement decision table and movable diagnostic chart tabs."""
from PyQt5 import QtCore, QtGui, QtWidgets
from ..models import Config
from .common import STATUS
from .charts import PlotCanvas, HistogramCanvas, WindowCountsCanvas
from .cloud_panel import DepthTerrainPanel


class MeasurementCriteriaPanel(QtWidgets.QWidget):
    """Explain actual estimator checks; never recompute acceptance in the GUI."""
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        self.window_choice = QtWidgets.QComboBox()
        self.window_choice.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.window_choice)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['判定项目', '本窗数值', '要求', '结果', '对冰凌测距的意义'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.table.setWordWrap(True)
        layout.addWidget(self.table)
        self.endpoint = None
        self.config = Config()

    def set_data(self, endpoint, config):
        self.endpoint, self.config = endpoint, config
        self.window_choice.blockSignals(True)
        self.window_choice.clear()
        for r in endpoint.history if endpoint else []:
            self.window_choice.addItem('搜索半径 r=%d px · %d 个有效样本' % (r.radius, len(r.valid_ids)))
        self.window_choice.setCurrentIndex(self.window_choice.count() - 1)
        self.window_choice.blockSignals(False)
        self.refresh()

    def refresh_simple(self, ep, cfg, r, index):
        rows = [
            ('近簇样本数', str(r.near_count), '> %d' % cfg.near_min_exclusive, r.checks.get('near_samples'), 'MAD去噪后保留的当前冰凌候选像素'),
            ('双群分离度 S', '—' if r.separation is None else '%.3f' % r.separation, '> %.2f' % cfg.separation_min_exclusive, r.checks.get('separation'), '两群均值差 / 两群方差和的平方根'),
            ('远簇比较样本', str(r.far_count), '> %d' % cfg.far_min_exclusive, r.checks.get('far_samples'), '远参考簇的有效样本数量；不要求占近簇的固定比例'),
            ('估计深度变化', '不适用' if r.center_change_m is None else '%.3f mm' % (r.center_change_m*1000), '仅记录，不拒绝', None, '同端点相邻可估深窗口的均值变化；首窗无此量'),
        ]
        self.table.setRowCount(len(rows))
        for row, (name, value, requirement, passed, meaning) in enumerate(rows):
            state = '仅记录' if row == 3 else '未评估' if passed is None else '通过' if passed else '未通过'
            for col, text in enumerate((name, value, requirement, state, meaning)):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        self.table.resizeRowsToContents()
        self.summary.setText('近簇、远簇数量与分离度同时达标即输出；否则扩至短边上限并拒绝；无评分。\n端点状态：%s；第%d/%d窗。%s' %
                             (STATUS.get(ep.status, ep.status), index+1, len(ep.history), '；'.join(r.reasons)))

    def refresh(self, *_):
        self.table.setRowCount(0)
        self.summary.setToolTip('')
        index = self.window_choice.currentIndex()
        if self.endpoint is None or index < 0:
            self.summary.setText('选择端点后，查看每次扩窗的实际判定值、阈值和拒绝原因。')
            return
        ep, cfg = self.endpoint, self.config
        r = ep.history[index]
        if cfg.endpoint_depth_mode == 'near_cluster_simple':
            self.refresh_simple(ep, cfg, r, index)
            return
        streak = r.consecutive_passes
        fmt = lambda value, factor=1: '—' if value is None else '%.3f' % (value * factor)
        tol = max(cfg.center_abs_m, cfg.center_rel * (r.center_m or 0))
        rows = [
            ('两簇分离度', fmt(r.separation), '≥ %.2f' % cfg.separation_min, 'separation', '中心差相对簇内噪声的大小，越大越易区分'),
            ('分位深度间隔', fmt(r.gap_m, 1000) + ' mm', '≥ %.1f mm' % (cfg.gap_min_m * 1000), 'gap', '远簇 P10 − 近簇 P90；防止重叠簇被当成前后景'),
            ('近 / 远 / 目标样本', '%d / %d / %d' % (r.near_count, r.far_count, r.target_count), '各 ≥ %d' % cfg.min_samples, 'samples', '去噪后数量；目标是近簇子集，不应相加'),
            ('空间归属', '%d 个目标候选点' % r.target_count, '连线带支持且归属明确', 'spatial', '检查近簇与框、端点连线及连通分量的关系'),
            ('端点锚区', '%s 点 / r=%s px' % (r.anchor.get('count', '—'), r.anchor.get('radius_px', '—')),
             '≥ %d 点；%s' % (cfg.anchor_min_samples, '当前窗内关联' if cfg.endpoint_depth_mode == 'near_cluster_mean'
                             else '/'.join(map(str, cfg.anchor_radii)) + ' px'), 'anchor',
             '目标分量须关联原端点，避免仅按面积选择邻近物体'),
            ('目标内部双群', '%s · 差 %s mm' % ('/'.join(map(str, r.subgroups.get('counts', []))) or '—',
             fmt(r.subgroups.get('center_gap_m'), 1000)), '不得存在可信双群', 'subgroups',
             '同时检查子群占比、数量、中心差、分离度与分位间隔；满足时拒绝混合平均'),
            ('均值代表性', '%s + %s mm' % (fmt(r.plane.get('bias_m'), 1000),
             fmt(r.plane.get('uncertainty_m'), 1000)), '合计 ≤ %.1f mm' % (cfg.plane_bias_max_m * 1000),
             'representative', '稳健平面只核验均值是否代表原端点，不提供输出深度；裕量不是概率区间'),
            ('空间覆盖', '%s / %s / %s 格' % (r.coverage.get('near', '—'), r.coverage.get('far', '—'),
             r.coverage.get('target', '—')), '各 ≥ %d 格（%d×%d px）' % (cfg.coverage_min_cells, cfg.coverage_cell_px,
             cfg.coverage_cell_px), 'coverage', '近 / 远 / 目标的像素网格覆盖，不是独立样本数'),
            ('均值变化', fmt(r.center_change_m, 1000) + ' mm', '≤ %.2f mm' % (tol * 1000), 'center', '与上一扩窗比较；首窗无前窗，不能通过'),
            ('成员稳定性 J', fmt(r.membership), '≥ %.2f' % cfg.membership_min, 'membership', '共有像素的目标归属 Jaccard；越接近 1 越稳定'),
            ('新增目标 / 远簇', '%s / %s 点' % (r.growth.get('new_target', '—'), r.growth.get('new_far', '—')),
             '各 ≥ %s / %s 点' % (r.growth.get('required_target', cfg.growth_min_samples),
             r.growth.get('required_far', cfg.growth_min_samples)), 'growth',
             '只统计前窗未观察的像素；重复或重分类像素不能制造新增证据'),
            ('新增区域一致性', fmt(r.growth.get('mean_difference_m'), 1000) + ' mm',
             '≤ %.1f mm' % (cfg.growth_mean_max_m * 1000), 'ring',
             '新增目标均值与保留目标均值的差；近簇近似模式仅提示，严格模式要求通过'),
            ('综合规则评分', fmt(r.score), '≥ %.2f' % cfg.confidence_min, 'confidence', '原四项评分的最小值；其余硬条件还须逐项通过，评分不是概率'),
            ('连续通过次数', str(streak), '≥ %d' % cfg.consecutive, None, '本窗全部硬条件与评分达标即接受；首窗缺跨窗比较项，不能通过'),
        ]
        self.table.setRowCount(len(rows))
        for row, (name, value, requirement, key, meaning) in enumerate(rows):
            passed = (streak >= cfg.consecutive if key is None else r.checks.get(key))
            state = '通过' if passed else '未评估' if passed is None else '未通过'
            advisory = key in r.advisories
            if advisory:
                passed = r.advisories[key]
                state = '正常（提示项）' if passed else '提示（不阻断）'
                requirement = '参考：' + requirement
                meaning += '；此项不参与通过/拒绝，也不替换算术均值'
            for col, text in enumerate((name, value, requirement, state, meaning)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                if passed is False:
                    item.setBackground(QtGui.QColor('#fff0e5'))
                if col == 3:
                    item.setForeground(QtGui.QColor('#087957' if passed else '#a66a16'))
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()
        first_failed = next((i for i, row in enumerate(rows) if row[3] is not None
                             and r.checks.get(row[3]) is False), None)
        if first_failed is not None:
            self.table.scrollToItem(self.table.item(first_failed, 0), QtWidgets.QAbstractItemView.PositionAtCenter)
        mode = '近簇均值近似' if cfg.endpoint_depth_mode == 'near_cluster_mean' else '严格局部一致'
        advice_names = {'representative': '均值代表性', 'ring': '新增区域一致性'}
        advisory_text = '、'.join(advice_names[k] for k, passed in r.advisories.items() if not passed)
        self.summary.setToolTip('；'.join(r.warnings))
        self.summary.setText('模式：%s。端点最终状态：%s。当前查看第 %d/%d 窗，候选均值 %s mm。\n本窗说明：%s%s' % (
            mode,
            STATUS.get(ep.status, ep.status), index + 1, len(ep.history), fmt(r.center_m, 1000),
            '；'.join(r.reasons) or '本窗硬条件通过',
            '\n仅提示：' + advisory_text + '（不阻断，详情见表格及判定记录）' if advisory_text else ''))


class DiagnosticsPlot(QtWidgets.QWidget):
    """Tab based diagnostics, retaining endpoint/depth attributes for callers."""
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMovable(True)
        self.scatter = PlotCanvas()
        self.histogram = HistogramCanvas()
        self.window_counts = WindowCountsCanvas()
        self.terrain = DepthTerrainPanel()
        self.criteria = MeasurementCriteriaPanel()
        self.guide = QtWidgets.QLabel()
        self.guide.setWordWrap(True)
        self.guide.setStyleSheet('color:#536277; padding:5px; background:#edf3f8;')
        self.guides = {
            self.terrain: '看空间位置：点来自最终像素搜索窗。绿为冰凌目标候选，蓝为远簇（背景候选）；颜色不是实物类别真值。Z 是光轴深度。',
            self.scatter: '看深度集中程度与两簇间隔：曲线由样本频数平滑得到，不是正态拟合或概率。每行峰高独立缩放；数量比较看 n。滚轮缩放、拖动平移。',
            self.histogram: '左图按统一数量轴显示各深度的像素数；右图显示扩窗候选均值，纵轴自动缩放。毫米级变化可能看起来明显；可靠性请查看“测距判定”。',
            self.window_counts: '看扩窗是否积累足够样本：绿色目标候选包含在橙色近簇中。虚线为当前配置阈值；达到样本下限后，还需分离、空间和稳定性条件通过。',
            self.criteria: '看端点为什么通过或拒绝：可选择历史窗口，逐项核对真实计算结果。两个端点都通过后才输出三维欧氏长度；候选均值不能作为已验证测量。',
        }
        for widget, title in ((self.terrain, '3D 深度点云'), (self.scatter, '深度簇分布'),
                              (self.histogram, '深度与均值变化'), (self.window_counts, '扩窗样本量'),
                              (self.criteria, '测距判定')):
            self.tabs.addTab(widget, title)
        self.tabs.currentChanged.connect(self.update_guide)
        layout.addWidget(self.guide)
        layout.addWidget(self.tabs)
        self.update_guide()
        self.endpoint = self.depth = self.detection = None
        self.setMinimumHeight(270)

    def update_guide(self, *_):
        self.guide.setText(self.guides.get(self.tabs.currentWidget(),
            '查看原始判定记录；评分表示规则满足程度，模拟真值仅用于模拟验收。'))
        self.tabs.tabBar().setToolTip(self.guide.text())

    def set_data(self, endpoint, depth, detection=None, intrinsics=None, config=None):
        self.endpoint, self.depth, self.detection = endpoint, depth, detection
        config = config or Config()
        for widget in (self.scatter, self.histogram, self.window_counts):
            widget.config = config
            widget.set_data(endpoint, depth, detection)
        self.terrain.set_data(endpoint, depth, detection, intrinsics)
        self.criteria.set_data(endpoint, config)
        if config.endpoint_depth_mode == 'near_cluster_simple':
            self.guides[self.window_counts] = ('看扩窗样本量：近簇与目标候选为同一组，不相加；'
                '近簇>%d、远簇>%d、S>%g同时满足即接受；均值变化仅记录。' %
                (config.near_min_exclusive, config.far_min_exclusive, config.separation_min_exclusive))
        else:
            self.guides[self.window_counts] = ('看扩窗是否积累足够样本：目标候选包含在近簇中。'
                '历史模式还需空间与稳定性条件通过。')
        self.update_guide()
