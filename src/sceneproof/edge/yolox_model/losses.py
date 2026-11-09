# Vendored from Megvii YOLOX (https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0.
# Source: yolox/models/yolo_head.py (get_losses, get_assignments, get_geometry_constraint, simota_matching,
# get_l1_target), yolox/models/losses.py (IOUloss) and yolox/utils/boxes.py (bboxes_iou).
# Modified for sceneproof: written as free functions over a YOLOXHead instance so the vendored head keeps its
# inference path unchanged; CUDA/AMP fallbacks and logging removed (CPU fine-tuning only).
# Copyright (c) Megvii Inc. All rights reserved.

from __future__ import annotations

import torch
import torch.nn.functional as F


def bboxes_iou(bboxes_a: torch.Tensor, bboxes_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of (cx, cy, w, h) boxes: [A,4] x [B,4] -> [A,B]."""
    tl = torch.max(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2, bboxes_b[:, :2] - bboxes_b[:, 2:] / 2)
    br = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2, bboxes_b[:, :2] + bboxes_b[:, 2:] / 2)
    area_a = torch.prod(bboxes_a[:, 2:], 1)
    area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)


def iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IOUloss(loss_type="iou"): 1 - IoU^2 per row, (cx, cy, w, h) boxes."""
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
    tl = torch.max(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
    br = torch.min(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
    area_p = torch.prod(pred[:, 2:], 1)
    area_g = torch.prod(target[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    iou = area_i / (area_p + area_g - area_i + 1e-16)
    return 1 - iou ** 2


def get_l1_target(l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
    l1_target[:, 0] = gt[:, 0] / stride - x_shifts
    l1_target[:, 1] = gt[:, 1] / stride - y_shifts
    l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
    l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
    return l1_target


def get_geometry_constraint(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, center_radius: float = 1.5):
    """Anchors whose centre lies within ``center_radius`` strides of a GT centre (fixed-centre prior)."""
    expanded_strides_per_image = expanded_strides[0]
    x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
    y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
    center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
    gt_l = gt_bboxes_per_image[:, 0:1] - center_dist
    gt_r = gt_bboxes_per_image[:, 0:1] + center_dist
    gt_t = gt_bboxes_per_image[:, 1:2] - center_dist
    gt_b = gt_bboxes_per_image[:, 1:2] + center_dist
    c_l = x_centers_per_image - gt_l
    c_r = gt_r - x_centers_per_image
    c_t = y_centers_per_image - gt_t
    c_b = gt_b - y_centers_per_image
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    anchor_filter = is_in_centers.sum(dim=0) > 0
    geometry_relation = is_in_centers[:, anchor_filter]
    return anchor_filter, geometry_relation


def simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
    matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
    n_candidate_k = min(10, pair_wise_ious.size(1))
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(cost[gt_idx], k=int(dynamic_ks[gt_idx]), largest=False)
        matching_matrix[gt_idx][pos_idx] = 1
    anchor_matching_gt = matching_matrix.sum(0)
    if anchor_matching_gt.max() > 1:  # one anchor matched several GTs: keep the cheapest
        multiple_match_mask = anchor_matching_gt > 1
        _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
        matching_matrix[:, multiple_match_mask] *= 0
        matching_matrix[cost_argmin, multiple_match_mask] = 1
    fg_mask_inboxes = anchor_matching_gt > 0
    num_fg = int(fg_mask_inboxes.sum().item())
    fg_mask[fg_mask.clone()] = fg_mask_inboxes
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
    gt_matched_classes = gt_classes[matched_gt_inds]
    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
    return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


@torch.no_grad()
def get_assignments(num_classes, batch_idx, num_gt, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                    expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds):
    """SimOTA: dynamic-k assignment of anchors to GT boxes."""
    fg_mask, geometry_relation = get_geometry_constraint(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts)
    bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
    cls_preds_ = cls_preds[batch_idx][fg_mask]
    obj_preds_ = obj_preds[batch_idx][fg_mask]
    num_in_boxes_anchor = bboxes_preds_per_image.shape[0]
    pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image)
    gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), num_classes).float()
    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
    cls_preds_ = (cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()).sqrt()
    pair_wise_cls_loss = F.binary_cross_entropy(
        cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
        gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
        reduction="none",
    ).sum(-1)
    cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 1e6 * (~geometry_relation)
    num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = simota_matching(
        cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
    return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg


def get_losses(head, x_shifts, y_shifts, expanded_strides, labels, outputs, origin_preds, dtype):
    """labels: [B, max_objs, 5] = (class, cx, cy, w, h) in input pixels; rows of zeros are padding."""
    bbox_preds = outputs[:, :, :4]
    obj_preds = outputs[:, :, 4:5]
    cls_preds = outputs[:, :, 5:]
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
    total_num_anchors = outputs.shape[1]
    x_shifts = torch.cat(x_shifts, 1)
    y_shifts = torch.cat(y_shifts, 1)
    expanded_strides = torch.cat(expanded_strides, 1)
    if head.use_l1:
        origin_preds = torch.cat(origin_preds, 1)
    cls_targets, reg_targets, l1_targets, obj_targets, fg_masks = [], [], [], [], []
    num_fg, num_gts = 0.0, 0.0
    for batch_idx in range(outputs.shape[0]):
        num_gt = int(nlabel[batch_idx])
        num_gts += num_gt
        if num_gt == 0:
            cls_target = outputs.new_zeros((0, head.num_classes))
            reg_target = outputs.new_zeros((0, 4))
            l1_target = outputs.new_zeros((0, 4))
            obj_target = outputs.new_zeros((total_num_anchors, 1))
            fg_mask = outputs.new_zeros(total_num_anchors).bool()
        else:
            gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
            gt_classes = labels[batch_idx, :num_gt, 0]
            gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = get_assignments(
                head.num_classes, batch_idx, num_gt, gt_bboxes_per_image, gt_classes, bbox_preds[batch_idx],
                expanded_strides, x_shifts, y_shifts, cls_preds, obj_preds)
            num_fg += num_fg_img
            cls_target = F.one_hot(gt_matched_classes.to(torch.int64), head.num_classes) * pred_ious_this_matching.unsqueeze(-1)
            obj_target = fg_mask.unsqueeze(-1)
            reg_target = gt_bboxes_per_image[matched_gt_inds]
            if head.use_l1:
                l1_target = get_l1_target(outputs.new_zeros((num_fg_img, 4)), gt_bboxes_per_image[matched_gt_inds],
                                          expanded_strides[0][fg_mask], x_shifts=x_shifts[0][fg_mask], y_shifts=y_shifts[0][fg_mask])
        cls_targets.append(cls_target)
        reg_targets.append(reg_target)
        obj_targets.append(obj_target.to(dtype))
        fg_masks.append(fg_mask)
        if head.use_l1:
            l1_targets.append(l1_target)
    cls_targets = torch.cat(cls_targets, 0)
    reg_targets = torch.cat(reg_targets, 0)
    obj_targets = torch.cat(obj_targets, 0)
    fg_masks = torch.cat(fg_masks, 0)
    num_fg = max(num_fg, 1)
    loss_iou = iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets).sum() / num_fg
    loss_obj = F.binary_cross_entropy_with_logits(obj_preds.view(-1, 1), obj_targets, reduction="none").sum() / num_fg
    loss_cls = F.binary_cross_entropy_with_logits(cls_preds.view(-1, head.num_classes)[fg_masks], cls_targets,
                                                  reduction="none").sum() / num_fg
    if head.use_l1:
        l1_targets = torch.cat(l1_targets, 0)
        loss_l1 = F.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets, reduction="none").sum() / num_fg
    else:
        loss_l1 = outputs.new_zeros(())
    reg_weight = 5.0
    loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1
    return loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1, num_fg / max(num_gts, 1)
