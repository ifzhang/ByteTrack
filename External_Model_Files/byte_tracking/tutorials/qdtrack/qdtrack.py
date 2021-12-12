import numpy as np
from mmdet.core import bbox2result
from mmdet.models import TwoStageDetector

from qdtrack.core import track2result
from ..builder import MODELS, build_tracker
from qdtrack.core import imshow_tracks, restore_result
from tracker import BYTETracker


@MODELS.register_module()
class QDTrack(TwoStageDetector):

    def __init__(self, tracker=None, freeze_detector=False, *args, **kwargs):
        self.prepare_cfg(kwargs)
        super().__init__(*args, **kwargs)
        self.tracker_cfg = tracker

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self._freeze_detector()

    def _freeze_detector(self):

        self.detector = [
            self.backbone, self.neck, self.rpn_head, self.roi_head.bbox_head
        ]
        for model in self.detector:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def prepare_cfg(self, kwargs):
        if kwargs.get('train_cfg', False):
            kwargs['roi_head']['track_train_cfg'] = kwargs['train_cfg'].get(
                'embed', None)

    def init_tracker(self):
#         self.tracker = build_tracker(self.tracker_cfg)
        self.tracker = BYTETracker()
    
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_match_indices,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)

        ref_x = self.extract_feat(ref_img)
        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas)

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore, **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.'
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.init_tracker()
        
        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, track_feats = self.roi_head.simple_test(x, img_metas, proposal_list, rescale)

        bboxes, labels, ids = self.tracker.update(det_bboxes, det_labels, frame_id, track_feats)
        
#         if track_feats is not None:
#             bboxes, labels, ids = self.tracker.match(
#                 bboxes=det_bboxes,
#                 labels=det_labels,
#                 track_feats=track_feats,
#                 frame_id=frame_id)
        
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes)

        if track_feats is not None:
            track_result = track2result(bboxes, labels, ids,
                                        self.roi_head.bbox_head.num_classes)
        else:
            track_result = [
                np.zeros((0, 6), dtype=np.float32)
                for i in range(self.roi_head.bbox_head.num_classes)
            ]
        return dict(bbox_results=bbox_result, track_results=track_result)

    def show_result(self,
                    img,
                    result,
                    thickness=1,
                    font_scale=0.5,
                    show=False,
                    out_file=None,
                    wait_time=0,
                    backend='cv2',
                    **kwargs):
        """Visualize tracking results.

        Args:
            img (str | ndarray): Filename of loaded image.
            result (dict): Tracking result.
                The value of key 'track_results' is ndarray with shape (n, 6)
                in [id, tl_x, tl_y, br_x, br_y, score] format.
                The value of key 'bbox_results' is ndarray with shape (n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            thickness (int, optional): Thickness of lines. Defaults to 1.
            font_scale (float, optional): Font scales of texts. Defaults
                to 0.5.
            show (bool, optional): Whether show the visualizations on the
                fly. Defaults to False.
            out_file (str | None, optional): Output filename. Defaults to None.
            backend (str, optional): Backend to draw the bounding boxes,
                options are `cv2` and `plt`. Defaults to 'cv2'.

        Returns:
            ndarray: Visualized image.
        """
        assert isinstance(result, dict)
        track_result = result.get('track_results', None)
        bboxes, labels, ids = restore_result(track_result, return_ids=True)
        img = imshow_tracks(
            img,
            bboxes,
            labels,
            ids,
            classes=self.CLASSES,
            thickness=thickness,
            font_scale=font_scale,
            show=show,
            out_file=out_file,
            wait_time=wait_time,
            backend=backend)
        return img
