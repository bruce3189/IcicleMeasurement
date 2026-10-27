"""Point-cloud geometry display, painting and mouse/keyboard navigation."""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..geometry import build_cloud, project as project_points


class DepthTerrainCanvas(QtWidgets.QWidget):
    """Interactive, dependency-free 3D view of measured endpoint-neighborhood depth."""
    sample_selected = QtCore.pyqtSignal(object)
    navigation_changed = QtCore.pyqtSignal()
    COLORS = ['#9ba8b7', '#edae48', '#4f7df3', '#17a77a']
    LABELS = ['离群/未分簇', '近簇其它', '远簇', '目标簇']
    MAX_RENDER_POINTS = 10000
    PERSPECTIVE_DISTANCE_FACTOR = 6.0
    EXAGGERATION_FACTOR = 4.0
    MIN_ZOOM = .05
    MAX_ZOOM = 80.
    WHEEL_STEP = 1.08

    def __init__(self):
        super().__init__()
        self.endpoint = self.depth = self.detection = None
        self.intrinsics = None
        self.vertices = np.empty((0, 3), dtype=float)
        self.vertex_categories = np.empty((0,), dtype=np.int8)
        self.render_indices = np.empty((0,), dtype=np.int32)
        self.samples = []
        # Vectorised caches for the hot paths (paintEvent / mouseMoveEvent).
        # Storing samples as parallel numpy arrays lets both hot paths run
        # without per-point Python loops, which matters once the cloud hits
        # MAX_RENDER_POINTS (10000): a list-of-tuple scan costs ~10 ms per
        # mouse-move, the vectorised version is sub-millisecond.
        self.sample_pixel_ids = np.empty((0,), dtype=np.int64)
        self.sample_categories = np.empty((0,), dtype=np.int8)
        self.sample_depths = np.empty((0,), dtype=float)
        # Projected-screen caches rebuilt only inside paintEvent.
        self._proj_screen = np.empty((0, 2), dtype=float)
        self._proj_index = np.empty((0,), dtype=np.int32)
        self._proj_pixel_id = np.empty((0,), dtype=np.int64)
        self._proj_category = np.empty((0,), dtype=np.int8)
        self._proj_depth = np.empty((0,), dtype=float)
        self.radius = 1
        self.z_origin_depth = 0.0
        self.origin_depth_fallback = False
        self.origin_depth_source = 'none'
        # Metric state: one coordinate system for points and origin.
        self.origin_xyz_m = np.zeros(3, dtype=float)
        self.relative_m = np.empty((0, 3), dtype=float)
        self.cloud_scale = 1.0
        self.metric_available = False
        self.category_visibility = np.ones(4, dtype=bool)
        self.point_size_factor = 1.0
        self._fit_cache = None
        self.focus_radius_px = 10.0
        self.selected_pixel_id = None
        self.drag_moved = False
        # Focus window: only points within `focus_radius_m` of the endpoint are
        # Default to showing every valid pixel in the final search window so
        # both target and background clusters are visible.  'local' is kept
        # for zooming into the endpoint neighbourhood.
        self.focus_mode = 'all'
        self.focus_radius_m = 0.0
        self.visible_mask = None
        # Display modes.
        self.projection_mode = 'perspective'
        self.depth_display_mode = 'true'
        self.rotation = self._default_rotation()
        self.zoom = 1.0
        self.pan = QtCore.QPointF(0, 0)
        # Camera orbit pivot only; sample XYZ and the endpoint reference stay fixed.
        self.orbit_pivot_relative = np.zeros(3, dtype=float)
        self.orbit_focus_id = None
        self._camera_distance_cache = None
        self.show_x_axis = self.show_y_axis = self.show_z_axis = True
        self.drag_button = None
        self.last_pos = None
        self.setMinimumSize(360, 170)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._pan_keys = set()
        self._pan_fine = False
        self._pan_clock = QtCore.QElapsedTimer()
        self._pan_timer = QtCore.QTimer(self)
        self._pan_timer.setInterval(16)
        self._pan_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._pan_timer.timeout.connect(self._tick_keyboard_pan)
        self._zoom_animation = QtCore.QVariantAnimation(self)
        self._zoom_animation.setDuration(110)
        self._zoom_animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._zoom_animation.valueChanged.connect(self._animate_zoom_frame)
        self._zoom_target = 1.

    def stop_zoom_animation(self):
        self._zoom_animation.stop()
        self._zoom_target = self.zoom

    def stop_keyboard_pan(self):
        self._pan_keys.clear()
        self._pan_timer.stop()

    def _pan_direction(self):
        x = int(QtCore.Qt.Key_D in self._pan_keys)-int(QtCore.Qt.Key_A in self._pan_keys)
        y = int(QtCore.Qt.Key_S in self._pan_keys)-int(QtCore.Qt.Key_W in self._pan_keys)
        length = max(float(np.hypot(x,y)),1.)
        return QtCore.QPointF(x/length,y/length)

    def _tick_keyboard_pan(self):
        elapsed = min(self._pan_clock.restart()/1000.,.05)
        self.pan += self._pan_direction()*(300.*elapsed*(.2 if self._pan_fine else 1.))
        self.update()

    def animate_zoom_at(self, position, factor):
        self.ensure_fit()
        running = self._zoom_animation.state() == QtCore.QAbstractAnimation.Running
        reversing = running and ((factor<1. and self._zoom_target>self.zoom) or (factor>1. and self._zoom_target<self.zoom))
        target = (self._zoom_target if running and not reversing else self.zoom)*factor
        self._zoom_animation.stop()
        self._zoom_target = float(np.clip(target,self.MIN_ZOOM,self.MAX_ZOOM))
        self._zoom_from = self.zoom
        self._zoom_pan_from = QtCore.QPointF(self.pan)
        self._zoom_position = QtCore.QPointF(position)
        self._zoom_animation.setStartValue(0.)
        self._zoom_animation.setEndValue(1.)
        self._zoom_animation.start()

    def _animate_zoom_frame(self, progress):
        value = self._zoom_target if float(progress)>=1. else self._zoom_from*np.exp(float(progress)*np.log(self._zoom_target/self._zoom_from))
        value = float(np.clip(value,self.MIN_ZOOM,self.MAX_ZOOM))
        ratio = value/self._zoom_from
        offset = self._zoom_position-QtCore.QPointF(self.plot_rect().center())
        self.pan = self._zoom_pan_from*ratio+offset*(1.-ratio)
        self.zoom = float(value)
        self.update(); self.navigation_changed.emit()

    def focusOutEvent(self, event):
        self.stop_keyboard_pan(); self.stop_zoom_animation()
        super().focusOutEvent(event)

    def hideEvent(self, event):
        self.stop_keyboard_pan(); self.stop_zoom_animation()
        super().hideEvent(event)

    def resizeEvent(self, event):
        if hasattr(self,'_zoom_animation'):
            self.stop_zoom_animation()
        super().resizeEvent(event)

    def set_projection_mode(self, mode):
        """'perspective' (real pinhole camera) or 'orthographic'."""
        self.stop_zoom_animation()
        self.projection_mode = 'orthographic' if str(mode).startswith('ortho') else 'perspective'
        self._fit_cache = None
        self.update()

    def set_depth_display_mode(self, mode):
        """'true' (isotropic, metric) or 'exaggerated' (Z stretched, not metric)."""
        self.stop_zoom_animation()
        self.depth_display_mode = 'exaggerated' if str(mode).startswith('exag') else 'true'
        self._fit_cache = None
        self._camera_distance_cache = None
        self.update()

    def set_focus_mode(self, mode):
        """'local' (only points near the endpoint) or 'all' (whole window)."""
        self.focus_mode = mode if mode in ('all', 'pixel', 'local') else 'all'
        self._rebuild_vertices()
        self.update()

    def set_focus_radius(self, value):
        if self.focus_mode == 'pixel':
            self.focus_radius_px = max(1.0, value)
        else:
            self.focus_radius_m = max(.001, value / 1000.0)
        self._rebuild_vertices()
        self.update()

    def set_category_visible(self, category, visible):
        self.category_visibility[category] = bool(visible)
        self.update()

    def _rebuild_vertices(self):
        """Recompute the display cloud: one isotropic factor, no distortion."""
        self._fit_cache = None
        self._camera_distance_cache = None
        if not len(self.relative_m):
            self.vertices = np.empty((0, 3), dtype=float)
            self.visible_mask = None
            return
        distance = np.linalg.norm(self.relative_m, axis=1)
        if self.focus_mode == 'local' and self.focus_radius_m > 0:
            mask = distance <= self.focus_radius_m
        elif self.focus_mode == 'pixel' and self.depth is not None:
            width = self.depth.shape[1]
            u, v = self.endpoint.pixel
            ids = self.sample_pixel_ids
            mask = np.hypot(ids % width - u, ids // width - v) <= self.focus_radius_px
        else:
            mask = np.ones(len(distance), bool)
        self.visible_mask = mask
        subset = self.relative_m[mask]
        # Include the endpoint origin in fit bounds. A background-only cloud
        # must not be scaled by its tiny within-plane spread and fly offscreen.
        extent = max(float(np.linalg.norm(subset, axis=1).max()) if len(subset) else 0.0, 1e-6)
        self.cloud_scale = float(self.radius) / extent
        self.vertices = self.relative_m * self.cloud_scale

    def display_vertices(self):
        """Vertices actually drawn, honouring the (clearly labelled) Z exaggeration."""
        if len(self.vertices) == 0:
            return self.vertices
        if self.depth_display_mode == 'exaggerated':
            return self.vertices * np.asarray((1.0, 1.0, self.EXAGGERATION_FACTOR))
        return self.vertices


    def set_data(self, endpoint, depth, detection=None, intrinsics=None):
        self.stop_keyboard_pan(); self.stop_zoom_animation()
        if endpoint is not self.endpoint:
            self.rotation = self._default_rotation()
            self.zoom = 1.0
            self.pan = QtCore.QPointF(0, 0)
            self.orbit_pivot_relative = np.zeros(3, dtype=float)
            self.orbit_focus_id = None
        self._camera_distance_cache = None
        self.endpoint, self.depth, self.detection, self.intrinsics = endpoint, depth, detection, intrinsics
        self.vertices = np.empty((0, 3), dtype=float)
        self.vertex_categories = np.empty((0,), dtype=np.int8)
        self.render_indices = np.empty((0,), dtype=np.int32)
        self.samples = []
        self.selected_pixel_id = None
        self.drag_button = self.last_pos = None
        self.sample_pixel_ids = np.empty((0,), dtype=np.int64)
        self.sample_categories = np.empty((0,), dtype=np.int8)
        self.sample_depths = np.empty((0,), dtype=float)
        self._proj_screen = np.empty((0, 2), dtype=float)
        self._proj_index = np.empty((0,), dtype=np.int32)
        self._proj_pixel_id = np.empty((0,), dtype=np.int64)
        self._proj_category = np.empty((0,), dtype=np.int8)
        self._proj_depth = np.empty((0,), dtype=float)
        self.relative_m = np.empty((0, 3), dtype=float)
        self.origin_xyz_m = np.zeros(3, dtype=float)
        self.cloud_scale = 1.0
        self.focus_radius_m = 0.0
        self.visible_mask = None
        self.metric_available = False
        self.origin_depth_fallback = False
        self.origin_depth_source = 'none'
        self._proj_screen = np.empty((0, 2), dtype=float)
        self._proj_index = np.empty((0,), dtype=np.int32)
        self._proj_pixel_id = np.empty((0,), dtype=np.int64)
        self._proj_category = np.empty((0,), dtype=np.int8)
        self._proj_depth = np.empty((0,), dtype=float)
        if not endpoint or not endpoint.history or depth is None or detection is None:
            self.update()
            return
        record = endpoint.history[-1]
        if not record.valid_ids:
            self.update()
            return
        width, height = depth.shape[1], depth.shape[0]
        keypoint_index = 0 if endpoint.name == 'tip' else 1
        u, v = detection.keypoints[keypoint_index]
        self.radius = max(int(record.radius), 1)
        x0, x1 = max(0, int(np.floor(u - self.radius))), min(width, int(np.ceil(u + self.radius)) + 1)
        y0, y1 = max(0, int(np.floor(v - self.radius))), min(height, int(np.ceil(v + self.radius)) + 1)
        grid_depth = np.full((y1 - y0, x1 - x0), np.nan, dtype=float)
        grid_category = np.full(grid_depth.shape, -1, dtype=np.int8)
        ids = np.asarray(record.valid_ids, dtype=np.int64)
        px, py = ids % width, ids // width
        inside = (px >= x0) & (px < x1) & (py >= y0) & (py < y1)
        ids, px, py = ids[inside], px[inside], py[inside]
        if len(ids):
            grid_depth[py - y0, px - x0] = depth.ravel()[ids]
            categories_for_ids = np.zeros(len(ids), dtype=np.int8)
            categories_for_ids[np.isin(ids, np.asarray(record.near_ids, dtype=np.int64))] = 1
            categories_for_ids[np.isin(ids, np.asarray(record.far_ids, dtype=np.int64))] = 2
            categories_for_ids[np.isin(ids, np.asarray(record.target_ids, dtype=np.int64))] = 3
            grid_category[py - y0, px - x0] = categories_for_ids
        gy, gx = np.where(np.isfinite(grid_depth))
        if not len(gx):
            self.update()
            return
        measured = grid_depth[gy, gx]
        categories = grid_category[gy, gx]
        # For a rejected edge endpoint, a neighborhood median can belong to the
        # background. Prefer the estimator's target-cluster candidate as a visual
        # origin, while keeping the measurement rejected and labeling it unverified.
        self.origin_depth_fallback = endpoint.depth_m is None
        if endpoint.depth_m is not None:
            self.z_origin_depth = float(endpoint.depth_m)
            self.origin_depth_source = 'accepted'
        elif record.center_m is not None and record.target_ids:
            self.z_origin_depth = float(record.center_m)
            self.origin_depth_source = 'target_candidate'
        else:
            self.z_origin_depth = float(np.median(measured))
            self.origin_depth_source = 'neighborhood_median'
        if len(categories) <= self.MAX_RENDER_POINTS:
            render_indices = np.arange(len(categories), dtype=np.int32)
        else:
            # Keep the selected geometry independent of cluster labels.
            render_indices = np.linspace(0, len(categories) - 1,
                                         self.MAX_RENDER_POINTS, dtype=np.int32)
        measured = measured[render_indices]
        categories = categories[render_indices]
        pixel_x, pixel_y = gx[render_indices] + x0, gy[render_indices] + y0
        # One coordinate system for every point and for the origin: deproject
        # each sample into camera XYZ, then subtract the single explicit
        # endpoint reference point.  Cluster category never enters here.
        if intrinsics is not None:
            cloud = build_cloud(pixel_x, pixel_y, measured, intrinsics, (u, v),
                                self.z_origin_depth, self.radius)
            self.origin_xyz_m = cloud['origin_xyz']
            self.relative_m = cloud['relative_m']
            self.cloud_scale = cloud['scale']
            self.metric_available = True
        else:
            # Without intrinsics the depth cannot be deprojected: fall back to a
            # pixel plane and say so instead of mixing pixel and metric axes.
            self.origin_xyz_m = np.zeros(3, dtype=float)
            self.relative_m = np.column_stack((pixel_x - float(u),
                                               pixel_y - float(v),
                                               np.zeros(len(pixel_x))))
            self.cloud_scale = self.radius / max(float(np.ptp(self.relative_m[:, :2])), 1e-9)
            self.metric_available = False
        # A local view spans the search window at the reference depth. Cluster
        # labels do not affect which positions are shown or how they are scaled.
        if self.metric_available:
            du = float(np.max(np.abs(pixel_x - u)))
            dv = float(np.max(np.abs(pixel_y - v)))
            footprint = np.hypot(du * self.z_origin_depth / intrinsics.fx,
                                 dv * self.z_origin_depth / intrinsics.fy)
            self.focus_radius_m = max(0.03, 1.6 * float(footprint))
        self.vertex_categories = categories
        self.render_indices = render_indices
        # Keep the legacy list for any external reader, but also store parallel
        # numpy arrays so the paint/mouse-move hot paths avoid Python loops.
        rendered_ids = (pixel_y * width + pixel_x).astype(np.int64)
        self.sample_pixel_ids = rendered_ids
        self.sample_categories = categories.astype(np.int8)
        self.sample_depths = measured.astype(float)
        self.focus_radius_px = max(3.0, self.radius * .5)
        self._rebuild_vertices()
        self.samples = list(zip(rendered_ids.tolist(),
                                categories.tolist(),
                                measured.tolist()))
        self.update()


    @staticmethod
    def _quat_multiply(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.asarray((aw*bw-ax*bx-ay*by-az*bz,
                           aw*bx+ax*bw+ay*bz-az*by,
                           aw*by-ax*bz+ay*bw+az*bx,
                           aw*bz+ax*by-ay*bx+az*bw), dtype=float)

    @classmethod
    def _axis_quaternion(cls, axis, degrees):
        axis = np.asarray(axis, dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        half = np.deg2rad(degrees) * .5
        return np.r_[np.cos(half), axis * np.sin(half)]

    @staticmethod
    def _normalize_quaternion(quaternion):
        return quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)

    @classmethod
    def _quaternion_from_matrix(cls, matrix):
        """Convert a right-handed world-to-view matrix to a normalized wxyz quaternion."""
        matrix = np.asarray(matrix, dtype=float)
        trace = float(np.trace(matrix))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2
            quaternion = np.asarray((.25 * scale,
                                     (matrix[2, 1] - matrix[1, 2]) / scale,
                                     (matrix[0, 2] - matrix[2, 0]) / scale,
                                     (matrix[1, 0] - matrix[0, 1]) / scale))
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2
                quaternion = np.asarray(((matrix[2, 1] - matrix[1, 2]) / max(scale, 1e-12),
                                         .25 * scale,
                                         (matrix[0, 1] + matrix[1, 0]) / max(scale, 1e-12),
                                         (matrix[0, 2] + matrix[2, 0]) / max(scale, 1e-12)))
            elif axis == 1:
                scale = np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2
                quaternion = np.asarray(((matrix[0, 2] - matrix[2, 0]) / max(scale, 1e-12),
                                         (matrix[0, 1] + matrix[1, 0]) / max(scale, 1e-12),
                                         .25 * scale,
                                         (matrix[1, 2] + matrix[2, 1]) / max(scale, 1e-12)))
            else:
                scale = np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2
                quaternion = np.asarray(((matrix[1, 0] - matrix[0, 1]) / max(scale, 1e-12),
                                         (matrix[0, 2] + matrix[2, 0]) / max(scale, 1e-12),
                                         (matrix[1, 2] + matrix[2, 1]) / max(scale, 1e-12),
                                         .25 * scale))
        return cls._normalize_quaternion(quaternion)

    @classmethod
    def _default_rotation(cls):
        # Keep camera +X aligned with image-right and camera +Y with image-down.
        # A small camera tilt exposes depth without rolling the image plane.
        angle = np.deg2rad(35.0)
        eye = np.asarray((0.0, -np.sin(angle), -np.cos(angle)))
        up_hint = np.asarray((0.0, -1.0, 0.0))
        right = np.cross(up_hint, eye)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up = np.cross(eye, right)
        return cls._quaternion_from_matrix(np.vstack((right, up, eye)))

    def set_view_from_direction(self, eye_direction):
        """Set an axis-aligned camera view without changing the point cloud or zoom."""
        eye = np.asarray(eye_direction, dtype=float)
        self._fit_cache = None
        eye /= max(float(np.linalg.norm(eye)), 1e-12)
        # The camera coordinate convention uses +Y downward and +Z away from
        # the camera. For front and X-side views, screen-up is therefore -Y.
        up_hint = np.asarray((0., 0., 1.) if abs(float(eye[1])) > .95 else (0., -1., 0.))
        right = np.cross(up_hint, eye)
        right /= max(float(np.linalg.norm(right)), 1e-12)
        up = np.cross(eye, right)
        view_rotation = np.vstack((right, up, eye))
        self.rotation = self._quaternion_from_matrix(view_rotation)
        self.update()

    def _project(self, vertices, scale, center_x, center_y):
        """Project display-space vertices to screen pixels.

        'perspective' places a real pinhole camera `distance` display units
        behind the view origin along the eye axis, so nearer points are
        magnified exactly as a camera would render them.  'orthographic'
        keeps the old parallel projection.  Both use the same vertex cloud,
        so rotating the view never changes any 3D distance.
        """
        distance = self.view_distance()
        screen, view_depth = project_points(
            np.asarray(vertices) - self.display_pivot(), self.rotation, scale, center_x, center_y,
            mode=self.projection_mode, distance=distance)
        return screen, view_depth

    def display_pivot(self):
        stretch = self.EXAGGERATION_FACTOR if self.depth_display_mode == 'exaggerated' else 1.
        return self.orbit_pivot_relative * self.cloud_scale * [1., 1., stretch]

    def view_distance(self):
        if self._camera_distance_cache is None:
            drawn = self.display_vertices()
            extent = float(np.linalg.norm(drawn-self.display_pivot(),axis=1).max()) if len(drawn) else 0.
            self._camera_distance_cache = max(self.PERSPECTIVE_DISTANCE_FACTOR*max(float(self.radius),1e-6), extent*1.25)
        return self._camera_distance_cache

    def plot_rect(self):
        return self.rect().adjusted(12,42,-12,-44)

    def ensure_fit(self):
        plot = self.plot_rect()
        fit_size = (plot.width(),plot.height())
        if self._fit_cache is not None and self._fit_cache[0] == fit_size:
            return
        previous = self._fit_cache
        drawn = self.display_vertices()
        mask = self.visible_mask.copy() if self.visible_mask is not None else np.ones(len(drawn),bool)
        mask &= self.category_visibility[self.vertex_categories]
        subset = drawn[mask]
        unit, depths = self._project(subset,1.,0.,0.)
        valid = np.isfinite(unit).all(axis=1)
        if self.projection_mode == 'perspective':
            valid &= depths < self.view_distance()-1e-3
        unit = unit[valid]
        # Fit observed points only. Axes, the endpoint marker and camera arrow
        # are guides and must not shrink a small foreground cluster.
        if len(unit):
            lo,hi = unit.min(axis=0),unit.max(axis=0)
        else:
            lo,hi = np.asarray([-1.,-1.]),np.asarray([1.,1.])
        span = np.maximum(hi-lo,max(self.radius*1e-5,1e-9))
        factor = max(1e-6,min(max(plot.width()-32,20)/span[0],max(plot.height()-32,20)/span[1]))
        center = (lo+hi)*.5
        self._fit_cache = (fit_size,factor,center)
        if previous is not None:
            # Preserve the viewed centre across an inline expansion/window resize.
            old_scale = max(previous[1]*self.zoom,1e-12)
            anchor = previous[2]-np.asarray([self.pan.x(),self.pan.y()])/old_scale
            pan = (center-anchor)*factor*self.zoom
            self.pan = QtCore.QPointF(float(pan[0]),float(pan[1]))

    def focus_selected(self):
        self.stop_zoom_animation()
        if self.selected_pixel_id is None:
            return False
        matches = np.flatnonzero(self.sample_pixel_ids == self.selected_pixel_id)
        if not len(matches):
            return False
        index = int(matches[0])
        if not self.category_visibility[self.vertex_categories[index]] or (self.visible_mask is not None and not self.visible_mask[index]):
            return False
        self.orbit_pivot_relative = self.relative_m[index].copy()
        self.orbit_focus_id = int(self.selected_pixel_id)
        self._fit_cache = self._camera_distance_cache = None
        self.ensure_fit()
        drawn = self.display_vertices()
        distances = np.linalg.norm(self.relative_m-self.orbit_pivot_relative,axis=1)
        mask = self.visible_mask.copy() if self.visible_mask is not None else np.ones(len(drawn),bool)
        mask &= self.category_visibility[self.vertex_categories]
        candidates = np.flatnonzero(mask)
        nearest = candidates[np.argsort(distances[candidates],kind='stable')[:min(120,len(candidates))]]
        unit,_ = self._project(drawn[nearest],1.,0.,0.)
        spread = float(np.percentile(np.linalg.norm(unit,axis=1),90)) if len(unit)>1 else 0.
        target = .35*min(self.plot_rect().width(),self.plot_rect().height())
        zoom = target/max(spread*self._fit_cache[1],1e-9) if spread>1e-9 else max(self.zoom,8.)
        self.zoom = float(np.clip(zoom,1.,self.MAX_ZOOM))
        pan = self._fit_cache[2]*self._fit_cache[1]*self.zoom
        self.pan = QtCore.QPointF(float(pan[0]),float(pan[1]))
        self.update(); self.navigation_changed.emit()
        return True

    def return_to_endpoint(self):
        self.stop_zoom_animation()
        self.orbit_pivot_relative = np.zeros(3,dtype=float)
        self.orbit_focus_id = None
        self._fit_cache = self._camera_distance_cache = None
        if len(self.vertices):
            self.ensure_fit()
            pan = self._fit_cache[2]*self._fit_cache[1]*self.zoom
            self.pan = QtCore.QPointF(float(pan[0]),float(pan[1]))
        self.update(); self.navigation_changed.emit()


    def set_axis_visible(self, axis, visible):
        setattr(self, 'show_%s_axis' % axis, bool(visible))
        self.update()

    def rotate_axis(self, axis, degrees):
        self.stop_zoom_animation()
        if isinstance(axis, str):
            axis = {'x': (1., 0., 0.), 'y': (0., 1., 0.), 'z': (0., 0., 1.)}[axis.lower()]
        turn = self._axis_quaternion(axis, degrees)
        self.rotation = self._normalize_quaternion(self._quat_multiply(turn, self.rotation))
        self.update()

    def reset_view(self):
        self.stop_zoom_animation(); self.stop_keyboard_pan()
        self.rotation = self._default_rotation()
        self.zoom = 1.0
        self.pan = QtCore.QPointF(0, 0)
        self._fit_cache = None
        self.orbit_pivot_relative = np.zeros(3,dtype=float)
        self.orbit_focus_id = None
        self._camera_distance_cache = None
        self.update()
        self.navigation_changed.emit()

    def fit_view(self):
        self.stop_zoom_animation()
        self._fit_cache = None
        self.pan = QtCore.QPointF(0, 0)
        self.zoom = 1.0
        self.update()
        self.navigation_changed.emit()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        background = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        background.setColorAt(0, QtGui.QColor('#172838'))
        background.setColorAt(1, QtGui.QColor('#0e1926'))
        p.fillRect(self.rect(), background)
        p.setPen(QtGui.QColor('#d3e2ef'))
        if not len(self.vertices):
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, '选中端点，查看实际深度样本 · 空深度不会生成点云')
            return
        plot = self.plot_rect()
        drawn = self.display_vertices()
        self.ensure_fit()
        scale = self._fit_cache[1] * self.zoom
        center_x = plot.center().x() - self._fit_cache[2][0] * scale + self.pan.x()
        center_y = plot.center().y() - self._fit_cache[2][1] * scale + self.pan.y()
        screen, view_depth = self._project(drawn, scale, center_x, center_y)
        visible = (self.visible_mask.copy() if self.visible_mask is not None
                   else np.ones(len(self.vertices), bool))
        visible &= self.category_visibility[self.vertex_categories]
        visible &= np.isfinite(screen).all(axis=1)
        if self.projection_mode == 'perspective':
            distance = self.view_distance()
            visible &= view_depth < distance - 1e-3
        visible &= ((screen[:, 0] >= plot.left()) & (screen[:, 0] <= plot.right()) &
                    (screen[:, 1] >= plot.top()) & (screen[:, 1] <= plot.bottom()))
        origin, _ = self._project(np.zeros((1, 3)), scale, center_x, center_y)
        p.save()
        p.setClipRect(plot)
        # Faint XY reference grid; it never changes the measured geometry.
        grid = []
        for value in np.linspace(-.7, .7, 7) * self.radius:
            grid.extend([[-self.radius * .7, value, 0], [self.radius * .7, value, 0],
                         [value, -self.radius * .7, 0], [value, self.radius * .7, 0]])
        grid_screen, _ = self._project(np.asarray(grid), scale, center_x, center_y)
        p.setPen(QtGui.QPen(QtGui.QColor(113, 146, 169, 35), 1))
        for i in range(0, len(grid_screen), 2):
            p.drawLine(QtCore.QPointF(*grid_screen[i]), QtCore.QPointF(*grid_screen[i + 1]))
        stretch = self.EXAGGERATION_FACTOR if self.depth_display_mode == 'exaggerated' else 1.0
        axis_length = self.radius * .62
        metric_mm = axis_length / self.cloud_scale * 1000
        for show, tip, color, name in (
                (self.show_x_axis, [axis_length, 0, 0], '#ff7c78', 'X 右'),
                (self.show_y_axis, [0, axis_length, 0], '#65d7a5', 'Y 下'),
                (self.show_z_axis, [0, 0, axis_length * stretch], '#b5a1ff', 'Z 远')):
            if not show:
                continue
            pts, _ = self._project(np.asarray([[0, 0, 0], tip]), scale, center_x, center_y)
            p.setPen(QtGui.QPen(QtGui.QColor(color), 2.2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawLine(QtCore.QPointF(*pts[0]), QtCore.QPointF(*pts[1]))
            label = '%s · %.1f mm' % (name, metric_mm) if self.metric_available else name + ' (px)'
            p.drawText(QtCore.QPointF(*(pts[1] + [5, -5])), label)
        camera = -self.origin_xyz_m if self.metric_available else np.asarray([0., 0., -1.])
        camera = camera * [1, 1, stretch]
        camera /= max(float(np.linalg.norm(camera)), 1e-12)
        pts, _ = self._project(np.asarray([[0, 0, 0], camera * self.radius * .72]), scale, center_x, center_y)
        direction = pts[1] - pts[0]
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        normal = np.asarray([-direction[1], direction[0]])
        p.setPen(QtGui.QPen(QtGui.QColor('#ff6e7b'), 3.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawLine(QtCore.QPointF(*pts[0]), QtCore.QPointF(*pts[1]))
        for sign in (-1, 1):
            p.drawLine(QtCore.QPointF(*pts[1]), QtCore.QPointF(*(pts[1] - direction * 9 + normal * sign * 4)))
        count = max(int(visible.sum()), 1)
        width = self.depth.shape[1]
        ids = self.sample_pixel_ids
        adjacent = (np.hypot(np.diff(ids % width), np.diff(ids // width)) <= 2.1)
        adjacent &= visible[1:] & visible[:-1]
        spacing = np.linalg.norm(np.diff(screen, axis=0), axis=1)[adjacent]
        spacing = float(np.median(spacing)) if len(spacing) else 3.0
        # Density follows projected neighbour spacing, not empty viewport area.
        point_diameter = float(np.clip(spacing * .7, .85, 4.0) * self.point_size_factor)
        # Preserve global far-to-near ordering. Only consecutive equal drawing
        # styles may be batched; regrouping all points by category breaks occlusion.
        order = np.flatnonzero(visible)
        order = order[np.argsort(view_depth[order], kind='stable')]
        if len(order):
            lo, hi = float(view_depth[order[0]]), float(view_depth[order[-1]])
            shade = np.clip(((view_depth[order] - lo) / max(hi - lo, 1e-9) * 4).astype(int), 0, 4)
            styles = self.vertex_categories[order] * 5 + shade
            boundaries = np.r_[0, np.flatnonzero(np.diff(styles)) + 1, len(order)]
            for a, b in zip(boundaries[:-1], boundaries[1:]):
                cat, level = divmod(int(styles[a]), 5)
                color = QtGui.QColor(self.COLORS[cat]).lighter(100 + level * 7)
                color.setAlpha(180 + level * 18)
                p.setPen(QtGui.QPen(color, point_diameter, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                coords = screen[order[a:b]]
                p.drawPoints(QtGui.QPolygonF([QtCore.QPointF(float(x), float(y)) for x, y in coords]))
        # Hollow reference and selected-sample markers remain legible without
        # replacing any observed sample with the origin.
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor('#f3f8fc'), 1.5))
        p.drawEllipse(QtCore.QPointF(*origin[0]), 6, 6)
        p.drawText(QtCore.QPointF(*(origin[0] + [-74, -12])), '候选原点*' if self.origin_depth_fallback else '端点原点')
        if self.selected_pixel_id is not None:
            selected = np.flatnonzero((self.sample_pixel_ids == self.selected_pixel_id) & visible)
            if len(selected):
                pt = screen[selected[0]]
                p.setPen(QtGui.QPen(QtGui.QColor('#ffe4a0'), 2))
                p.drawEllipse(QtCore.QPointF(*pt), 8, 8)
        p.restore()
        p.setPen(QtGui.QColor('#cfdfec'))
        projection = '透视' if self.projection_mode == 'perspective' else '正交'
        mode = '真实等比' if self.depth_display_mode == 'true' else 'Z ×4 夸张 · 非等比'
        pivot = '绕端点' if self.orbit_focus_id is None else '绕选中样本'
        p.drawText(14,23,'3D 点云 / %s / %s / 相对适配 %.2f× / %s' % (projection,mode,self.zoom,pivot))
        p.setPen(QtGui.QColor('#98b0c5'))
        p.setFont(QtGui.QFont(self.font().family(), 8))
        p.drawText(QtCore.QRectF(14, self.height() - 37, self.width() - 28, 34),
                   QtCore.Qt.TextWordWrap,
                   '可见 %d / 抽样 %d / 有效 %d 点 · WASD 平移 · 左拖旋转 · 右拖平移 · Shift 精调 · F 聚焦 · 双击复位' %
                   (int(visible.sum()), len(self.vertices), len(self.endpoint.history[-1].valid_ids)))
        self._proj_index = order.astype(np.int32)
        self._proj_screen = screen[order]
        self._proj_pixel_id = self.sample_pixel_ids[order]
        self._proj_category = self.sample_categories[order]
        self._proj_depth = self.sample_depths[order]

    def _pick_sample(self, pos):
        if not len(self._proj_screen):
            return None
        delta = self._proj_screen - [pos.x(), pos.y()]
        distances = np.sum(delta * delta, axis=1)
        minimum = float(distances.min())
        if minimum >= 81:
            return None
        # Cache is in drawing order; choose the frontmost of coincident points.
        candidates = np.flatnonzero(distances <= minimum + .25)
        return int(candidates[-1])

    def mousePressEvent(self, event):
        if event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self.stop_zoom_animation()
            self.setFocus(QtCore.Qt.MouseFocusReason)
            self.press_pos = event.pos()
            self.drag_moved = False
            self.drag_button = event.button()
            self.last_pos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_button is not None and self.last_pos is not None:
            if (event.pos() - self.press_pos).manhattanLength() > 3:
                self.drag_moved = True
            delta = event.pos() - self.last_pos
            if self.drag_button == QtCore.Qt.LeftButton:
                distance = float(np.hypot(delta.x(), delta.y()))
                if distance:
                    # Incremental quaternion orbit has no Euler pitch clamp or pole lock.
                    camera_axis = (delta.y(), delta.x(), 0.0)
                    turn = self._axis_quaternion(camera_axis, distance * .62)
                    self.rotation = self._normalize_quaternion(
                        self._quat_multiply(turn, self.rotation))
            else:
                self.pan += QtCore.QPointF(delta)
            self.last_pos = event.pos()
            self.update()
            return
        # Vectorised nearest-point search: 10k points costs ~0.3 ms here
        # versus ~10 ms for the old Python list-of-tuple scan.  The 9 px
        # snap radius is kept identical (81 px² threshold) so tooltip
        # behaviour is unchanged.
        nearest = None
        nearest_dist = 9.0
        picked = self._pick_sample(event.pos())
        if picked is not None:
            mx = float(event.pos().x())
            my = float(event.pos().y())
            dx = self._proj_screen[:, 0] - mx
            dy = self._proj_screen[:, 1] - my
            d2 = dx * dx + dy * dy
            i = picked
            if float(d2[i]) < 81.0:
                pixel_id = int(self._proj_pixel_id[i])
                category = int(self._proj_category[i])
                depth_m = float(self._proj_depth[i])
                index = int(self._proj_index[i])
                nearest = (pixel_id, category, depth_m, index)
                nearest_dist = float(d2[i]) ** 0.5
        if nearest:
            pixel_id, category, depth_m, index = nearest
            pwidth = self.depth.shape[1]
            if self.metric_available and index < len(self.relative_m):
                offset = self.relative_m[index]
                self.setToolTip('%s\n深度 Z：%.5f m\n像素 (x=%d, y=%d)\n相对端点 (m)：X=%+.4f  Y=%+.4f  Z=%+.4f\n'
                                '与端点距离：%.4f m' % (
                                    self.LABELS[category], depth_m, pixel_id % pwidth, pixel_id // pwidth,
                                    offset[0], offset[1], offset[2], float(np.linalg.norm(offset))))
            else:
                self.setToolTip('%s\n深度 Z：%.5f m\n像素 (x=%d, y=%d)' %
                                (self.LABELS[category], depth_m, pixel_id % pwidth, pixel_id // pwidth))
        else:
            self.setToolTip('单击点云后 WASD 上下左右平移 · 左拖旋转 · 右拖平移 · 平滑滚轮缩放 · Shift 精调 · 双击复位')

    def mouseReleaseEvent(self, event):
        if event.button() == self.drag_button:
            if event.button() == QtCore.Qt.LeftButton and not self.drag_moved:
                picked = self._pick_sample(event.pos())
                if picked is not None:
                    index = int(self._proj_index[picked])
                    self.selected_pixel_id = int(self._proj_pixel_id[picked])
                    self.sample_selected.emit({'pixel_id': self.selected_pixel_id,
                        'depth_m': float(self.sample_depths[index]),
                        'relative_m': self.relative_m[index].tolist(),
                        'xyz_m': (self.relative_m[index] + self.origin_xyz_m).tolist(),
                        'category': self.LABELS[int(self.sample_categories[index])]})
                    self.update()
            self.drag_button = None
            self.last_pos = None
            self.setCursor(QtCore.Qt.ArrowCursor)

    def wheelEvent(self, event):
        plot = self.rect().adjusted(12, 42, -12, -44)
        position = event.posF()
        if not len(self.vertices) or not QtCore.QRectF(plot).contains(position):
            event.ignore()
            return
        pixel_delta = event.pixelDelta().y()
        angle_delta = event.angleDelta().y()
        if pixel_delta:
            exponent = float(np.clip(pixel_delta * .0025, -1.2, 1.2))
        elif angle_delta:
            exponent = float(np.clip(angle_delta / 120. * np.log(self.WHEEL_STEP), -1.2, 1.2))
        else:
            event.ignore()
            return
        if event.modifiers() & QtCore.Qt.ShiftModifier:
            exponent *= .2
        self.animate_zoom_at(position,float(np.exp(exponent)))
        event.accept()

    def zoom_at(self, position, factor):
        """Keep the projected point under the cursor fixed during wheel zoom.

        Focal scaling does not rotate the camera or translate measured XYZ.
        Compensating pan also works after orbiting or panning the fitted view.
        """
        self.stop_zoom_animation()
        old_zoom = self.zoom
        new_zoom = max(self.MIN_ZOOM,min(self.MAX_ZOOM,old_zoom*factor))
        ratio = new_zoom / old_zoom
        plot_center = QtCore.QPointF(self.rect().adjusted(12, 42, -12, -44).center())
        offset = position - plot_center
        self.pan = self.pan * ratio + offset * (1.0 - ratio)
        self.zoom = new_zoom
        self.update()
        self.navigation_changed.emit()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_W,QtCore.Qt.Key_A,QtCore.Qt.Key_S,QtCore.Qt.Key_D):
            if not len(self.vertices):
                event.ignore(); return
            self.stop_zoom_animation()
            self._pan_fine = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
            if not event.isAutoRepeat() and event.key() not in self._pan_keys:
                self._pan_keys.add(event.key())
                self.pan += self._pan_direction()*(6.*(.2 if self._pan_fine else 1.))
            if not self._pan_timer.isActive():
                self._pan_clock.start(); self._pan_timer.start()
            self.update(); event.accept(); return
        if event.key() == QtCore.Qt.Key_Shift:
            self._pan_fine = True
        elif event.key() == QtCore.Qt.Key_F:
            if self.focus_selected():
                event.accept(); return
        elif event.key() == QtCore.Qt.Key_Home:
            self.reset_view(); event.accept(); return
        super().keyPressEvent(event)

    def keyReleaseEvent(self,event):
        if event.key() in (QtCore.Qt.Key_W,QtCore.Qt.Key_A,QtCore.Qt.Key_S,QtCore.Qt.Key_D):
            if not event.isAutoRepeat():
                self._pan_keys.discard(event.key())
                if not self._pan_keys:
                    self._pan_timer.stop()
            event.accept(); return
        if event.key() == QtCore.Qt.Key_Shift:
            self._pan_fine = False
        super().keyReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.reset_view()
        event.accept()

    def panel(self, painter, rect, title):
        painter.setPen(QtGui.QColor('#dce5ef'))
        painter.setBrush(QtGui.QColor('#ffffff'))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QtGui.QColor('#526175'))
        painter.drawText(int(rect.left() + 14), int(rect.top() + 22), title)
