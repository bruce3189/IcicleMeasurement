import unittest
import numpy as np
from icicle_measure.models import Config, Frame, Detection, Intrinsics
from icicle_measure.core import endpoint_estimate, Strategies
from icicle_measure.window_analysis import WindowAnalysisConfig


class BasicNineTests(unittest.TestCase):
    def setUp(self):
        self.z=np.full((41,41),2.)
        self.frame=Frame(np.zeros((41,41,3),np.uint8),self.z)
        self.k=Intrinsics(100.,100.,20.,20.)
        self.det=Detection('one',(16.,16.,25.,25.),[(20.,20.),(20.,19.)])

    def estimate(self,config=None):
        return endpoint_estimate(self.frame,self.det,0,self.k,config or Config(),Strategies(),lambda:False)

    def test_9_square_accepts_27_near_47_far_and_7_invalid(self):
        ids=[y*41+x for y in range(16,25) for x in range(16,25)]
        self.z.flat[ids[:27]]=.8
        self.z.flat[ids[-7:]]=0
        ep=self.estimate()
        self.assertEqual(ep.status,'accepted')
        self.assertEqual((ep.history[0].near_count,ep.history[0].far_count),(27,47))
        self.assertEqual(ep.history[0].subgroups['invalid_n'],7)
        self.z.flat[ids[-8]]=0
        ep=self.estimate()
        self.assertEqual(ep.status,'rejected')
        self.assertEqual(ep.history[-1].far_count,46)

    def test_root_can_have_far_less_than_half_of_near(self):
        self.z[:]=.8
        ids=[y*41+x for y in range(8,33) for x in range(8,33)]
        self.z.flat[ids[:47]]=2.
        self.det.bbox=(0.,0.,40.,40.)
        cfg=Config(radii=(12,),far_to_near_min_exclusive=.99)
        ep=self.estimate(cfg)
        self.assertEqual(ep.status,'accepted')
        self.assertEqual((ep.history[0].near_count,ep.history[0].far_count),(578,47))
        self.assertNotIn('far_to_near',ep.history[0].checks)
        self.assertNotIn('far_to_near_min_exclusive',cfg.active_dict())

    def test_radius_zero_means_1_square_and_is_not_accepted(self):
        cfg=Config(radii=(0,1,2,3,4,5))
        cfg.validate();WindowAnalysisConfig(radii=(0,1)).validate()
        ep=self.estimate(cfg)
        first=ep.history[0]
        self.assertEqual(first.bounds,(20,20,21,21))
        self.assertEqual(len(first.valid_ids),1)
        self.assertFalse(all(first.checks.values()))
        self.assertIsNone(first.separation)
        with self.assertRaises(ValueError):Config(radii=(-1,0,4)).validate()


if __name__=='__main__':unittest.main()
