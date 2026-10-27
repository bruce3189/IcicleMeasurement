import copy
from dataclasses import replace
import unittest
import cv2
import numpy as np
from test_pseudo_depth import fixture
from test_measure import temporary_directory
from icicle_measure.pseudo_depth import compose_scene, export_project, load_project
from icicle_measure.pseudo_geometry import apply_background_planes, boundary_evidence, matches_prompts


class RefinementTests(unittest.TestCase):
    def test_manual_negative_prompt_cannot_be_ignored(self):
        mask=np.ones((30,30),bool)
        self.assertFalse(matches_prompts(mask, [[10,10]], [0]))
        self.assertTrue(matches_prompts(mask, [[10,10]], [1]))

    def test_legacy_project_reloads_original_surface(self):
        import json
        from pathlib import Path
        s,cfg=fixture(); original=compose_scene(s,replace(cfg,generation_version=1))
        with temporary_directory() as folder:
            export_project(folder,original)
            path=Path(folder)/'pseudo_project.json'
            meta=json.loads(path.read_text(encoding='utf-8')); del meta['config']['generation_version']
            path.write_text(json.dumps(meta),encoding='utf-8')
            restored=load_project(folder)
            np.testing.assert_array_equal(original.clean_raw,restored.clean_raw)
            np.testing.assert_array_equal(original.observed_raw,restored.observed_raw)

    def test_plane_is_planar_in_camera_xyz_and_preserves_excluded_pixels(self):
        z = np.full((60, 80), 3., np.float32)
        valid = np.ones(z.shape, bool); valid[:2] = False
        excluded = np.zeros_like(valid); excluded[20:30, 30:35] = True
        plane = {'polygon': [[5, 5], [70, 5], [70, 55], [5, 55]], 'depths_m': [1.5, 2., 1.6]}
        result, edited, _ = apply_background_planes(z, valid, excluded, [plane])
        y, x = np.nonzero(edited)
        xyz = np.column_stack(((x-40)*result[y,x]/100, (y-30)*result[y,x]/100, result[y,x]))
        design = np.column_stack((xyz, np.ones(len(x))))
        _, _, vt = np.linalg.svd(design, full_matrices=False)
        self.assertLess(np.max(np.abs(design @ vt[-1])), 1e-6)
        np.testing.assert_array_equal(result[~edited], z[~edited])
        self.assertFalse(edited[excluded].any())

    def test_saturated_model_is_not_used_as_metric_evidence(self):
        s, cfg = fixture()
        a = compose_scene(s, replace(cfg, generation_version=1))
        b = compose_scene(s, replace(cfg, generation_version=2))
        self.assertEqual(a.report['objects'][0]['base_m'], cfg.near_m)
        row = b.report['objects'][0]
        self.assertTrue(row['model_depth_saturated'])
        self.assertEqual(row['base_m'], cfg.fallback_m)
        self.assertEqual(row['trend_source'], 'assumed_zero_axial_trend')

    def test_plane_edits_do_not_change_ice_and_are_reproducible(self):
        s, cfg = fixture(); before = compose_scene(s, cfg)
        s.edits['background_planes'] = [{'polygon': [[10,10],[309,10],[309,229],[10,229]], 'depths_m':[1.8,2.2,1.9]}]
        after = compose_scene(s, cfg)
        np.testing.assert_array_equal(before.clean_raw[s.masks[0]], after.clean_raw[s.masks[0]])
        self.assertEqual(after.frame.metadata['background_planes'][0]['depths_m'], [1.8,2.2,1.9])
        with temporary_directory() as directory:
            export_project(directory, after)
            restored = load_project(directory)
            np.testing.assert_array_equal(after.observed_raw, restored.observed_raw)
            self.assertEqual(restored.scene.edits, s.edits)

    def test_boundary_evidence_prefers_visible_silhouette(self):
        rgb = np.zeros((50,50,3), np.uint8); rgb[10:40,20:30] = 200
        accurate = np.zeros((50,50), bool); accurate[10:40,20:30] = True
        shifted = np.roll(accurate, 10, axis=1)
        self.assertGreater(boundary_evidence(rgb, accurate), boundary_evidence(rgb, shifted))

    def test_invalid_plane_rejected_without_modifying_input(self):
        z = np.ones((20,20), np.float32); initial=z.copy()
        for points, depths in [([[1,1],[2,2],[3,3]],[1,1,1]), ([[1,1],[18,1],[18,18],[1,18]],[.1,10,10])]:
            with self.assertRaises(ValueError):
                apply_background_planes(z, z>0, z<0, [{'polygon':points,'depths_m':depths}])
        np.testing.assert_array_equal(z, initial)
