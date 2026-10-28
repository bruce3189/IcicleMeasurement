"""验证加权确实改了tip梯度，且不把缺失点或base顺序处理错。"""
import unittest
import torch
from ultralytics.utils.loss import KeypointLoss, RLELoss
from weighted_pose import WeightedKeypointLoss, point_weights, TipTrainer, tip_coordinate_loss
from evaluate import match, containing_boxes
from predict import contained_points
import numpy as np


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.sigma=torch.ones(3)/3
        self.pred=torch.tensor([[[.2,.2,1.],[.2,.2,1.],[.2,.2,1.]]],requires_grad=True)
        self.gt=torch.zeros_like(self.pred)
        self.mask=torch.ones((1,3),dtype=torch.bool)
        self.area=torch.ones((1,1))

    def test_unit_weight_matches_stock_value_and_gradient(self):
        old=KeypointLoss(self.sigma)(self.pred,self.gt,self.mask,self.area)
        new=WeightedKeypointLoss(self.sigma,point_weights(1))(self.pred,self.gt,self.mask,self.area)
        self.assertTrue(torch.allclose(old,new))
        self.assertTrue(torch.allclose(torch.autograd.grad(old,self.pred,retain_graph=True)[0],
                                       torch.autograd.grad(new,self.pred)[0]))

    def test_tip_gradient_is_three_times_either_base(self):
        value=WeightedKeypointLoss(self.sigma,point_weights(3))(self.pred,self.gt,self.mask,self.area)
        grad=torch.autograd.grad(value,self.pred)[0]
        self.assertTrue(torch.allclose(grad[:,0,:2],3*grad[:,1,:2]))
        self.assertTrue(torch.allclose(grad[:,1,:2],grad[:,2,:2]))

    def test_missing_tip_has_zero_gradient_and_all_missing_is_finite(self):
        mask=self.mask.clone();mask[:,0]=False
        loss=WeightedKeypointLoss(self.sigma,point_weights(3))
        grad=torch.autograd.grad(loss(self.pred,self.gt,mask,self.area),self.pred)[0]
        self.assertEqual(grad[0,0].abs().sum().item(),0)
        self.assertEqual(loss(self.pred,self.gt,torch.zeros_like(mask),self.area).item(),0)

    def test_rle_uses_same_weights(self):
        error=torch.ones((3,2),requires_grad=True)
        value=RLELoss()(torch.ones((3,2)),torch.zeros(3),error,point_weights(3))
        grad=torch.autograd.grad(value,error)[0]
        self.assertTrue(torch.allclose(grad[0],3*grad[1]))

    def test_direct_tip_error_has_gradient_for_large_offsets_and_masks_missing(self):
        predicted=torch.tensor([[500.,700.],[800.,900.]],requires_grad=True)
        value=tip_coordinate_loss(predicted,torch.zeros((2,2)),torch.tensor([1000.,1000.]),torch.tensor([True,False]))
        grad=torch.autograd.grad(value,predicted)[0]
        self.assertTrue(torch.all(grad[0]>0))
        self.assertEqual(grad[1].abs().sum().item(),0)

    def test_matching_never_reuses_prediction_and_handles_empty(self):
        boxes=[[0,0,10,10],[1,0,11,10]]
        self.assertEqual(len(match(boxes,[[0,0,10,10]])),1)
        self.assertEqual(match(boxes,[]),[])

    def test_containment_preserves_instance_order_and_confidence(self):
        boxes=np.array([[10,20,30,50],[100,200,130,250]])
        raw=np.array([[[0,60,.8],[15,25,.9],[25,30,.7]],
                      [[140,190,.6],[110,210,.9],[120,220,.7]]])
        safe,changed=contained_points(raw,boxes)
        self.assertTrue(np.array_equal(safe[...,-1],raw[...,-1]))
        self.assertTrue(np.all(safe[...,:2]>=boxes[:,None,:2]))
        self.assertTrue(np.all(safe[...,:2]<=boxes[:,None,2:]))
        self.assertEqual(changed.tolist(),[[True,False,False],[True,False,False]])
        self.assertEqual(raw[0,0,0],0)  # 原始预测不能被原地修改。

    def test_expansion_contains_own_points_without_merging_instances(self):
        boxes=np.array([[10.,20.,30.,50.],[100.,200.,130.,250.]])
        points=np.array([[[8.,60.,.8],[15.,25.,.9],[25.,30.,.7]],
                         [[140.,190.,.6],[110.,210.,.9],[120.,220.,.7]]])
        result=containing_boxes(boxes,points)
        self.assertEqual(result.tolist(),[[8.,20.,30.,60.],[100.,190.,140.,250.]])
        self.assertEqual(boxes[0,0],10.)

    def test_head_only_freezes_detection_parameters_and_batchnorm(self):
        from ultralytics import YOLO
        from evaluate import ORIGINAL
        trainer=object.__new__(TipTrainer)
        trainer.model=YOLO(str(ORIGINAL)).model
        trainer.head_only=True
        trainer.freeze_layer_names=[]
        trainer._model_train()
        trainable=[n for n,p in trainer.model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        prefix=f'model.{len(trainer.model.model)-1}.'
        self.assertTrue(all(n.startswith(prefix) for n in trainable))
        self.assertFalse(any(n.startswith(prefix+'cv2.') or n.startswith(prefix+'cv3.') for n in trainable))
        for module in trainer.model.model[:-1]:
            self.assertFalse(module.training)
        self.assertFalse(trainer.model.model[-1].cv2.training)
        self.assertTrue(trainer.model.model[-1].cv4.training)


if __name__=='__main__':unittest.main()
