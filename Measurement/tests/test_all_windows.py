import unittest
import numpy as np
from icicle_measure.models import Frame,Detection
from icicle_measure.window_analysis import (WindowAnalysisConfig,candidate_checks,
    windows_to_short_side,analyze_frame_windows)


class AllWindowTests(unittest.TestCase):
    def setUp(self):
        self.cfg=WindowAnalysisConfig()
        self.z=np.full((81,81),2.,dtype=float)
        self.z[37:44,37:42]=.8
        self.det=Detection('one',(0.,0.,80.,80.),[(40.,40.),(40.,35.)])

    def analyze(self):
        return analyze_frame_windows(Frame(np.zeros((81,81,3),np.uint8),self.z),[self.det],self.cfg)

    def test_count_boundaries_are_strict(self):
        self.assertFalse(candidate_checks(26,47,4,self.cfg)['near_samples'])
        self.assertTrue(candidate_checks(27,47,4,self.cfg)['near_samples'])
        self.assertFalse(candidate_checks(27,46,4,self.cfg)['far_samples'])
        self.assertTrue(candidate_checks(27,47,4,self.cfg)['far_samples'])

    def test_far_ratio_is_not_an_acceptance_gate(self):
        checks=candidate_checks(100,47,4,self.cfg)
        self.assertTrue(all(checks.values()))
        self.assertNotIn('far_to_near', checks)

    def test_separation_boundary_is_strict(self):
        self.assertFalse(candidate_checks(31,51,3.,self.cfg)['separation'])
        self.assertTrue(candidate_checks(31,51,3.00001,self.cfg)['separation'])
        self.assertFalse(candidate_checks(31,51,None,self.cfg)['separation'])

    def test_windows_continue_after_first_candidate(self):
        ep=self.analyze()['objects'][0]['endpoints'][0]
        candidates=[i for i,r in enumerate(ep['windows']) if r['candidate']]
        self.assertTrue(candidates)
        self.assertLess(candidates[0],len(ep['windows'])-1)
        self.assertEqual(ep['windows'][-1]['radius_px'],39)

    def test_no_score_selected_depth_or_length(self):
        result=self.analyze();self.assertIsNone(result['distance_m'])
        self.assertIsNone(result['scoring_rule'])
        for obj in result['objects']:
            self.assertIsNone(obj['distance_m'])
            for ep in obj['endpoints']:
                for key in ('score','selected_window','depth_m','xyz_m'):self.assertIsNone(ep[key])
                for r in ep['windows']:self.assertIsNone(r['score'])

    def test_range_endpoints_and_outside_values(self):
        self.cfg.radii=(5,)
        self.det.bbox=(35.,35.,45.,45.)
        self.z[:]=2.
        values=[.001,5.,.000999,5.001,0.,-1.,np.nan,np.inf]
        self.z[38,36:44]=values
        result=self.analyze();ep=result['objects'][0]['endpoints'][0]
        row=ep['windows'][-1]
        self.assertEqual(row['invalid_n'],6)
        self.assertEqual(row['valid_n'],row['area_n']-6)
        self.assertEqual(result['effective_valid_range_m'],[.001,5.])

    def test_frame_defaults_and_values_are_not_clipped(self):
        self.z[40,40]=5.1;frame=Frame(np.zeros((81,81,3),np.uint8),self.z)
        self.assertEqual((frame.min_depth_m,frame.max_depth_m),(.001,5.))
        analyze_frame_windows(frame,[self.det]);self.assertEqual(frame.depth_m[40,40],5.1)

    def test_no_bbox_padding_crop(self):
        windows=windows_to_short_side((100,100),(40,10,70,90),(40,10),self.cfg.radii)
        r,b=windows[-1];self.assertEqual(r,14)
        self.assertEqual(b,(26,0,55,25))

    def test_unique_windows_and_cap_beyond_60(self):
        windows=windows_to_short_side((400,400),(0,0,300,300),(150,150),self.cfg.radii)
        self.assertEqual(windows[-1][0],149)
        self.assertEqual(len(windows),len({b for r,b in windows}))
        tiny=windows_to_short_side((5,5),(0,0,100,100),(2,2),self.cfg.radii)
        self.assertEqual(len(tiny),1)

    def test_mixed_near_cluster_is_not_split_in_current_scope(self):
        self.cfg.radii=(3,5,8,12,18,25,35,45,60)
        yy,xx=np.mgrid[35:46,35:46]
        ring=(np.maximum(abs(xx-40),abs(yy-40))>3)
        points=np.column_stack((yy[ring],xx[ring]))[:35]
        self.z[points[:,0],points[:,1]]=.6
        ep=self.analyze()['objects'][0]['endpoints'][0]
        self.assertAlmostEqual(ep['windows'][0]['selected']['near_mean_m'],.8)
        self.assertTrue(all(not r['internal_split'] for r in ep['windows']))
        self.assertAlmostEqual(ep['windows'][1]['selected']['near_mean_m'],.7)

    def test_identity_failure_is_recorded_instead_of_switching_target(self):
        self.cfg.radii=(3,5,8,12,18,25,35,45,60)
        yy,xx=np.mgrid[35:46,35:46]
        ring=(np.maximum(abs(xx-40),abs(yy-40))>3)
        points=np.column_stack((yy[ring],xx[ring]))[:40]
        self.z[points[:,0],points[:,1]]=.6
        ep=self.analyze()['objects'][0]['endpoints'][0]
        self.assertTrue(any('无法延续原簇' in r['reasons'] for r in ep['windows']))
        for row in ep['windows']:
            if row['selected'] is not None:self.assertAlmostEqual(row['selected']['near_mean_m'],.8)

    def test_empty_or_uniform_windows_remain_in_table(self):
        for value in (0.,1.):
            self.z[:]=value
            ep=self.analyze()['objects'][0]['endpoints'][0]
            self.assertEqual(ep['status'],'no_candidate')
            self.assertEqual(len(ep['windows']),len(windows_to_short_side(self.z.shape,self.det.bbox,self.det.keypoints[0],self.cfg.radii)))
            self.assertTrue(all(r['basic']['S'] is None for r in ep['windows']))

    def test_bad_endpoint_and_cancellation(self):
        self.det.keypoints[0]=None
        self.assertEqual(self.analyze()['objects'][0]['status'],'skipped')
        frame=Frame(np.zeros((81,81,3),np.uint8),self.z)
        result=analyze_frame_windows(frame,[],cancel=lambda:True)
        self.assertIsNone(result['distance_m'])

if __name__=='__main__':unittest.main()
