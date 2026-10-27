"""只改变关键点定位损失的相对权重，不改安装包、模型结构或框损失。"""
from functools import partial

import torch
import torch.nn.functional as F
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils.loss import E2ELoss, KeypointLoss, PoseLoss26


def point_weights(tip_weight, device=None):
    """均值归一化：[3,1,1] -> [1.8,.6,.6]，避免无意增加总损失。"""
    if tip_weight <= 0:
        raise ValueError('tip_weight 必须大于0')
    weights = torch.tensor([tip_weight, 1., 1.], device=device)
    return weights / weights.mean()


def tip_coordinate_loss(predicted, target, heights, visible):
    """以GT框高归一化真实二维坐标；大误差保持梯度，不用预测框推算点。"""
    if not visible.any():
        return predicted.sum()*0
    delta=(predicted[visible]-target[visible])/heights[visible,None].clamp_min(1.)
    return F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=.05,reduction='none').sum(1).mean()


class WeightedKeypointLoss(KeypointLoss):
    def __init__(self, sigmas, weights):
        super().__init__(sigmas)
        self.register_buffer('weights', weights)

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        # 保持本地8.4.183的OKS损失及缺失点归一化，只在逐点求平均前加权。
        d = (pred_kpts[..., :2] - gt_kpts[..., :2]).square().sum(-1)
        factor = kpt_mask.shape[1] / (kpt_mask.sum(1) + 1e-9)
        e = d / ((2 * self.sigmas).square() * (area + 1e-9) * 2)
        return (factor[:, None] * (1 - torch.exp(-e)) * kpt_mask * self.weights).mean()


class WeightedPoseLoss(PoseLoss26):
    def __init__(self, model, tal_topk=10, tal_topk2=None, tip_weight=1., tip_l1=0.):
        super().__init__(model, tal_topk, tal_topk2)
        if list(self.kpt_shape) != [3, 3]:
            raise ValueError('本实验仅支持 tip/base_left/base_right 三点，kpt_shape=[3,3]')
        weights = point_weights(tip_weight, self.device)
        self.keypoint_loss = WeightedKeypointLoss(self.keypoint_loss.sigmas, weights)
        if self.rle_loss is not None:
            # YOLO26有OKS定位和RLE两项；两项必须使用一致的逐点权重。
            self.target_weights = weights
        self.tip_l1=tip_l1

    def calculate_keypoints_loss(self,masks,target_gt_idx,keypoints,stride_tensor,target_bboxes,pred_kpts):
        # super会原地把target_bboxes除以stride，因此辅助项在调用前取GT框高。
        auxiliary=pred_kpts[..., :0].sum()
        if self.tip_l1 and masks.any():
            gt=keypoints[masks.nonzero(as_tuple=True)[0],target_gt_idx[masks]]
            strides=stride_tensor.view(1,-1).expand(masks.shape[0],-1)[masks]
            heights=(target_bboxes[masks][:,3]-target_bboxes[masks][:,1])/strides
            auxiliary=tip_coordinate_loss(pred_kpts[masks][:,0,:2],gt[:,0,:2]/strides[:,None],heights,gt[:,0,2]!=0)
        location,visibility,rle=super().calculate_keypoints_loss(
            masks,target_gt_idx,keypoints,stride_tensor,target_bboxes,pred_kpts)
        return location+self.tip_l1*auxiliary,visibility,rle


class TipTrainer(PoseTrainer):
    def __init__(self, *args, tip_weight=1., head_only=False, tip_l1=0., **kwargs):
        self.tip_weight = tip_weight
        self.head_only = head_only
        self.tip_l1=tip_l1
        super().__init__(*args, **kwargs)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        # BaseTrainer默认把val batch翻倍。这里强制单图验证，避免验证阶段峰值反而更高。
        return super().get_dataloader(dataset_path,batch_size=1 if mode=='val' else batch_size,
                                      rank=rank,mode=mode)

    def set_model_attributes(self):
        super().set_model_attributes()
        if type(self.model.model[-1]).__name__ != 'Pose26':
            raise ValueError('本实验要求YOLO26 Pose模型')
        # 两个训练分支都加权；只改一个分支会让NMS/无NMS两条路径学习目标不一致。
        self.model.criterion = self._weighted_criterion()
        print('TIP_WEIGHTS 两分支定位/RLE权重:', point_weights(self.tip_weight).tolist(), flush=True)
        if self.head_only:
            # 在优化器创建前冻结特征提取层；检测头的cv2/cv3在每轮模式设置时冻结。
            self.args.freeze = list(range(len(self.model.model)-1))

    def _weighted_criterion(self):
        factory=partial(WeightedPoseLoss,tip_weight=self.tip_weight,tip_l1=self.tip_l1)
        return E2ELoss(self.model,factory)

    def resume_training(self,ckpt):
        super().resume_training(ckpt)
        if self.resume:
            # 8.4.183的父类恢复时会重建默认criterion，必须恢复自定义损失及分支调度进度。
            self.model.criterion=self._weighted_criterion()
            self.model.criterion.updates=self.start_epoch-1
            self.model.criterion.update()

    def _model_train(self):
        super()._model_train()
        if not self.head_only:
            return
        head_index=len(self.model.model)-1
        prefix=f'model.{head_index}.'
        pose_parts=('cv4.', 'cv4_kpts.', 'cv4_sigma.', 'one2one_cv4.',
                    'one2one_cv4_kpts.', 'one2one_cv4_sigma.', 'flow_model.')
        for name,parameter in self.model.named_parameters():
            parameter.requires_grad_(name.startswith(prefix) and name[len(prefix):].startswith(pose_parts))
        for module in self.model.model[:-1]:
            module.eval()
        head=self.model.model[-1]
        for name in ('cv2','cv3','one2one_cv2','one2one_cv3'):
            getattr(head,name).eval()

