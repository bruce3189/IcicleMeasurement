import csv
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from icicle_measure.models import Config, Detection, Frame, Intrinsics
from icicle_measure.core import measure_frame
from icicle_measure.io import export_results
from icicle_measure.window_analysis import WindowAnalysisConfig, analyze_endpoint, windows_to_short_side
from tests.helpers import temporary_directory


class StartWindowCleanupTests(unittest.TestCase):
    def fixture(self):
        z = np.full((31, 31), 2.)
        z[11:20, 11:14] = .8
        return Frame(np.zeros((31, 31, 3), np.uint8), z), Detection(
            'one', (0., 0., 30., 30.), [(15., 15.), (15., 14.)]), Intrinsics(100., 100., 15., 15.)

    def test_default_starts_with_9_and_can_accept_first_window(self):
        frame, det, k = self.fixture()
        result = measure_frame(frame, [det], k)
        ep = result.objects[0].endpoints[0]
        self.assertEqual(ep.status, 'accepted')
        self.assertEqual(len(ep.history), 1)
        self.assertEqual(ep.history[0].bounds, (11, 11, 20, 20))
        self.assertEqual(Config().radii, WindowAnalysisConfig().radii)

    def test_short_side_and_image_edge_still_limit_nominal_start(self):
        windows = windows_to_short_side((31, 31), (11., 0., 19.5, 30.), (15., 15.), Config().radii)
        self.assertEqual(windows, [(3, (12, 12, 19, 19))])
        frame, det, k = self.fixture()
        det.bbox = (11., 0., 19.5, 30.)
        self.assertEqual(measure_frame(frame, [det], k).objects[0].status, 'rejected')
        edge = windows_to_short_side((31, 31), (0., 0., 30., 30.), (0., 0.), Config().radii)
        self.assertEqual(edge[0][1], (0, 0, 5, 5))

    def test_only_initial_mad_remains_and_counts_balance(self):
        frame, det, k = self.fixture()
        frame.depth_m[:] = 2.
        initial = np.array([y*31+x for y in range(10,21) for x in range(10,21)])[:70]
        ring = np.array([y*31+x for y in range(7,24) for x in range(7,24)
                         if not (10 <= y < 21 and 10 <= x < 21)])
        frame.depth_m.flat[initial] = .81
        frame.depth_m.flat[ring[:70]] = .8
        frame.depth_m.flat[ring[70:72]] = .805
        det.bbox = (7., 7., 24., 24.)
        result = analyze_endpoint(frame, det, 0, WindowAnalysisConfig())
        first, last = result['windows'][0], result['windows'][-1]
        self.assertAlmostEqual(first['selected']['near_mean_m'], .81)
        self.assertFalse(last['internal_split'])
        self.assertAlmostEqual(last['selected']['near_mean_m'], .805)
        self.assertEqual(last['mad_removed_initial_n'], 0)
        self.assertEqual(last['mad_removed_internal_n'], 0)
        self.assertEqual(last['unselected_n'], 0)
        for w in result['windows']:
            selected = w['selected'] or {'near_n': 0, 'far_n': 0}
            self.assertEqual(w['area_n'], w['invalid_n'] + w['mad_removed_n'] +
                             selected['near_n'] + selected['far_n'] + w['unselected_n'])
            self.assertEqual(w['valid_n'], len(w['valid_ids']))

    def test_new_export_hides_legacy_fields_without_changing_runtime_or_legacy_exports(self):
        frame, det, k = self.fixture()
        result = measure_frame(frame, [det], k)
        with temporary_directory() as directory:
            out = export_results(directory, frame, [det], result)
            saved = json.loads((out/'measurement.json').read_text(encoding='utf-8'))
            ep = saved['objects'][0]['endpoints'][0]
            self.assertNotIn('confidence', ep)
            self.assertNotIn('score', ep['history'][0])
            self.assertNotIn('anchor', ep['history'][0])
            self.assertEqual(ep['depth_m'], result.objects[0].endpoints[0].depth_m)
            with (out/'measurement.csv').open(encoding='utf-8-sig') as f:
                self.assertNotIn('confidence', csv.DictReader(f).fieldnames)
            self.assertIsNone(result.objects[0].endpoints[0].confidence)
            # Serialization compatibility for explicitly selected historical modes.
            legacy = replace(result, config={'endpoint_depth_mode': 'near_cluster_mean'})
            export_results(directory, frame, [det], legacy)
            saved = json.loads((out/'measurement.json').read_text(encoding='utf-8'))
            self.assertIn('anchor', saved['objects'][0]['endpoints'][0]['history'][0])


if __name__ == '__main__':
    unittest.main()
