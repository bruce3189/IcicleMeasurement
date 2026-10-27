"""Synthetic pseudo-depth scenes shared by the test modules."""
import numpy as np
from icicle_measure.models import Detection
from icicle_measure.pseudo_depth import PseudoConfig, PseudoScene, virtual_intrinsics


def fixture(width=320, height=240):
    cfg = PseudoConfig(width=width, height=height)
    valid = np.zeros((height, width), bool)
    valid[10:height - 10, 10:width - 10] = True
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[valid] = (150, 160, 170)
    q = np.broadcast_to(np.linspace(.1, 1., width, dtype=np.float32), (height, width)).copy()
    masks = np.zeros((2, height, width), bool)
    x, y0, y1 = width // 2, height // 5, height * 4 // 5
    masks[0, y0:y1 + 1, x - 5:x + 6] = True
    q[masks[0]] = 3.
    detections = [Detection('ice-A', (x - 12., y0 - 3., x + 15., y1 + 3.),
                            [(x + 10., y1), (x, y0)]),
                  Detection('ice-B', (20., 20., 35., 70.), [(25., 60.), None])]
    transform = {'source_size': [width, height], 'output_size': [width, height],
                 'resized_size': [width, height], 'offset': [0, 0], 'scale_xy': [1., 1.],
                 'pixel_center_translation': [0., 0.]}
    scene = PseudoScene(rgb, q, masks, valid, detections, virtual_intrinsics(cfg, transform), transform,
                        {'fixture': 'no model inference'}, original_masks=masks.copy(),
                        background_offset=np.zeros(valid.shape, np.float32))
    return scene, cfg
