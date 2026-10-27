# Icicle RGB-D Measurement Implementation Plan

**Goal:** Implement the approved dual-endpoint depth estimator, simulator and PyQt5 desktop interface.

**Architecture:** A NumPy/OpenCV core with replaceable clustering, selection, scoring and aggregation strategies; file adapters and synthetic fixtures; a worker-thread Qt interface consuming the same core API.

**Tech Stack:** Python 3.9, NumPy, OpenCV, PyQt5, unittest.

## Execution checklist

1. Define input/result/config dataclasses, validate aligned metric depth and calibrated intrinsics. Create `icicle_measure/models.py` and `tests/test_measure.py`.
2. Implement candidate Otsu splitting, MAD rejection, spatial support, shared-pixel stability, conservative stopping and original-ray deprojection in `icicle_measure/core.py`. Run `python -m unittest discover -s tests -v` after implementation.
3. Implement bundle/YOLO input, strict unit conversion, JSON/CSV/overlay export and reproducible simulations in `icicle_measure/io.py`, `icicle_measure/simulator.py`, `icicle_measure/render.py`.
4. Implement PyQt5 worker, image views, configuration editing, per-endpoint diagnostics, cancellation and exports in `icicle_measure/gui.py`. Add GUI integration tests using Qt offscreen.
5. Exercise exact/noisy/missing/ambiguous/depth-drift cases, report accepted rate and error; run GUI import/config/select/cancel/export tests. Save a synthetic example, benchmark report and GUI screenshot.
6. Document formats, configurable thresholds, module extension interfaces, camera assumptions and startup commands in `README.md`; provide a Windows launcher.

## Acceptance

Both endpoints must pass before a distance is emitted. Synthetic noise benchmark reports acceptance rate, endpoint error <=10 mm and distance error <=20 mm for accepted outputs. Invalid/unmeasurable cases must carry reasons. Simulation verifies software, not transparent-ice sensor accuracy. No camera capture or YOLO inference in this version.

## Completed validation

- All six implementation steps completed, 2026-10-24.
- The original 22 unittest cases passed, including native Windows Qt import, configuration, selection, export, cancellation and rerun.
- User requested larger robust clusters and better visual diagnostics. Defaults increased to 64 minimum / 100 full score samples, search radii to 60 px, and bbox padding to 10 px. At confidence 0.8, accepted windows generally contain at least 80 samples in near, far and selected target sets.
- Added one-point-per-valid-pixel radial-depth scatter, separate depth histogram/center stability and grouped per-window sample count charts, plus compact XYZ table and styled diagnostic tabs.
- Expanded suite: 23 unittest cases passed on native Windows Qt, including samples in chart data and all four diagnostic page screenshots.
- 30/30 noisy synthetic cases accepted; maximum endpoint error 0.6283 mm and maximum length error 0.3075 mm.
- Invalid/ambiguous scene rejection checks passed. Outputs, input examples, GUI screenshot and logs are under `outputs/rgbd_validation`.
