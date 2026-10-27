"""Editable RGB -> pseudo-depth workbench. CPU models run outside the GUI runtime."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from .pseudo_depth import (ROOT, DEFAULT_RGB_DIR, DEFAULT_LABEL_DIR, SAMPLE_STEMS,
    PseudoConfig, compose_scene, scene_from_inference, export_project, load_project, fixed_depth_rgb, depth_colorbar)
from .pseudo_models import run_inference
from .pixel_inspector import inspect_pixel, pixel_text
from .render import draw_selected_distance, distance_summary

from .ui.pseudo_preview import scene_preview, EditView
from .ui.pseudo_workspace import PseudoTask, PseudoDepthPanel, PseudoDepthDialog
