"""
Author: 520Chris
Description: person search network based on resnet50.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign, RoIPool

from models.base_feat_layer import BaseFeatLayer
from models.proposal_feat_layer import ProposalFeatLayer
from oim.labeled_matching_layer import LabeledMatchingLayer
from oim.unlabeled_matching_layer import UnlabeledMatchingLayer
from rpn.proposal_target_layer import ProposalTargetLayer
from rpn.rpn_layer import RPN
from utils.config import cfg
from utils.net_utils import smooth_l1_loss


class Network(nn.Module):
    """Person search network."""

    def __init__(self):
        super(Network, self).__init__()
        rpn_depth = 1024  # depth of the feature map fed into RPN
        num_classes = 2   # bg and fg

        # Extracting feature layer
        self.base_feat_layer = BaseFeatLayer()
        self.proposal_feat_layer = ProposalFeatLayer()

        # RPN
        self.rpn = RPN(rpn_depth)
        self.proposal_target_layer = ProposalTargetLayer(num_classes=num_classes)
        self.rois = None  # proposals produced by RPN

        # Pooling layer
        pool_size = 14
        self.roi_align = RoIAlign((pool_size, pool_size), 1.0 / 16.0, 0)
        self.roi_pool = RoIPool((pool_size, pool_size), 1.0 / 16.0)

        # Identification layer
        self.cls_score = nn.Linear(2048, 2)
        self.bbox_pred = nn.Linear(2048, num_classes * 4)
        self.feat_lowdim = nn.Linear(2048, 256)
        self.labeled_matching_layer = LabeledMatchingLayer()
        self.unlabeled_matching_layer = UnlabeledMatchingLayer()

    def forward(self, im_data, im_info, gt_boxes, is_prob=False, rois=None):
        assert im_data.size(0) == 1, 'Single batch only.'

        # Extract basic feature from image data
        base_feat = self.base_feat_layer(im_data)

        if not is_prob:
            # Feed base feature map to RPN to obtain rois
            self.rois, rpn_loss_cls, rpn_loss_bbox = self.rpn(base_feat, im_info, gt_boxes)
        else:
            assert rois is not None, "RoIs is not given in detect probe mode."
            self.rois, rpn_loss_cls, rpn_loss_bbox = rois, 0, 0

        if self.training:
            # Sample 128 rois and assign them labels and bbox regression targets
            roi_data = self.proposal_target_layer(self.rois, gt_boxes)
            self.rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, pid_label = roi_data
        else:
            rois_label, rois_target, rois_inside_ws, rois_outside_ws, pid_label = [None] * 5

        # Do roi pooling based on region proposals
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.roi_align(base_feat, self.rois)
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.roi_pool(base_feat, self.rois)
        else:
            raise NotImplementedError("Only support roi_align and roi_pool.")

        # Extract the features of proposals
        proposal_feat = self.proposal_feat_layer(pooled_feat).squeeze()

        cls_score = self.cls_score(proposal_feat)
        bbox_pred = self.bbox_pred(proposal_feat)
        feat_lowdim = self.feat_lowdim(proposal_feat)

        if self.training:
            loss_cls = F.cross_entropy(cls_score, rois_label)
            loss_bbox = smooth_l1_loss(bbox_pred,
                                       rois_target,
                                       rois_inside_ws,
                                       rois_outside_ws)

            # OIM loss
            feat = F.normalize(feat_lowdim)
            labeled_matching_scores, id_labels = self.labeled_matching_layer(feat, pid_label)
            labeled_matching_scores *= 10
            unlabeled_matching_scores = self.unlabeled_matching_layer(feat, pid_label)
            unlabeled_matching_scores *= 10
            id_scores = torch.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
            id_prob = F.softmax(id_scores, dim=1)
            loss_id = F.cross_entropy(id_prob, id_labels, ignore_index=-1)
        else:
            loss_cls, loss_bbox, loss_id = 0, 0, 0

        return cls_score, bbox_pred, feat_lowdim, rpn_loss_cls, rpn_loss_bbox, loss_cls, loss_bbox, loss_id
