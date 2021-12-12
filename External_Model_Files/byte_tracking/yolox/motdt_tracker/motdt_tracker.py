import numpy as np
#from numba import jit
from collections import OrderedDict, deque
import itertools
import os
import cv2
import torch
from torch._C import dtype
import torchvision

from yolox.motdt_tracker import matching
from .kalman_filter import KalmanFilter
from .reid_model import load_reid_model, extract_reid_features
from yolox.data.dataloading import get_yolox_datadir

from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):

    def __init__(self, tlwh, score, max_n_features=100, from_det=True):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.max_n_features = max_n_features
        self.curr_feature = None
        self.last_feature = None
        self.features = deque([], maxlen=self.max_n_features)

        # classification
        self.from_det = from_det
        self.tracklet_len = 0
        self.time_by_tracking = 0

        # self-tracking
        self.tracker = None

    def set_feature(self, feature):
        if feature is None:
            return False
        self.features.append(feature)
        self.curr_feature = feature
        self.last_feature = feature
        # self._p_feature = 0
        return True

    def predict(self):
        if self.time_since_update > 0:
            self.tracklet_len = 0

        self.time_since_update += 1

        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

        if self.tracker:
            self.tracker.update_roi(self.tlwh)

    def self_tracking(self, image):
        tlwh = self.tracker.predict(image) if self.tracker else self.tlwh
        return tlwh

    def activate(self, kalman_filter, frame_id, image):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter  # type: KalmanFilter
        self.track_id = self.next_id()
        # cx, cy, aspect_ratio, height, dx, dy, da, dh
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        # self.tracker = sot.SingleObjectTracker()
        # self.tracker.init(image, self.tlwh)

        del self._tlwh

        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, image, new_id=False):
        # self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(new_track.tlwh))
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.time_since_update = 0
        self.time_by_tracking = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

        self.set_feature(new_track.curr_feature)

    def update(self, new_track, frame_id, image, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.time_since_update = 0
        if new_track.from_det:
            self.time_by_tracking = 0
        else:
            self.time_by_tracking += 1
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        if update_feature:
            self.set_feature(new_track.curr_feature)
            if self.tracker:
                self.tracker.update(image, self.tlwh)

    @property
    #@jit
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    #@jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    #@jit
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    def tracklet_score(self):
        # score = (1 - np.exp(-0.6 * self.hit_streak)) * np.exp(-0.03 * self.time_by_tracking)

        score = max(0, 1 - np.log(1 + 0.05 * self.time_by_tracking)) * (self.tracklet_len - self.time_by_tracking > 2)
        # score = max(0, 1 - np.log(1 + 0.05 * self.n_tracking)) * (1 - np.exp(-0.6 * self.hit_streak))
        return score

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class OnlineTracker(object):

    def __init__(self, model_folder, min_cls_score=0.4, min_ap_dist=0.8, max_time_lost=30, use_tracking=True, use_refind=True):

        self.min_cls_score = min_cls_score
        self.min_ap_dist = min_ap_dist
        self.max_time_lost = max_time_lost

        self.kalman_filter = KalmanFilter()

        self.tracked_stracks = []   # type: list[STrack]
        self.lost_stracks = []      # type: list[STrack]
        self.removed_stracks = []   # type: list[STrack]

        self.use_refind = use_refind
        self.use_tracking = use_tracking
        self.classifier = None
        self.reid_model = load_reid_model(model_folder)

        self.frame_id = 0

    def update(self, output_results, img_info, img_size, img_file_name):
        img_file_name = os.path.join(get_yolox_datadir(), 'mot', 'train', img_file_name)
        image = cv2.imread(img_file_name)
        # post process detections
        output_results = output_results.cpu().numpy()
        confidences = output_results[:, 4] * output_results[:, 5]
        
        bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        bbox_xyxy = bboxes
        tlwhs = self._xyxy_to_tlwh_array(bbox_xyxy)
        remain_inds = confidences > self.min_cls_score
        tlwhs = tlwhs[remain_inds]
        det_scores = confidences[remain_inds]

        self.frame_id += 1

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        """step 1: prediction"""
        for strack in itertools.chain(self.tracked_stracks, self.lost_stracks):
            strack.predict()

        """step 2: scoring and selection"""
        if det_scores is None:
            det_scores = np.ones(len(tlwhs), dtype=float)
        detections = [STrack(tlwh, score, from_det=True) for tlwh, score in zip(tlwhs, det_scores)]
        if self.use_tracking:
            tracks = [STrack(t.self_tracking(image), 0.6 * t.tracklet_score(), from_det=False)
                        for t in itertools.chain(self.tracked_stracks, self.lost_stracks) if t.is_activated]
            detections.extend(tracks)
        rois = np.asarray([d.tlbr for d in detections], dtype=np.float32)
        scores = np.asarray([d.score for d in detections], dtype=np.float32)
        # nms
        if len(detections) > 0:
            nms_out_index = torchvision.ops.batched_nms(
            torch.from_numpy(rois),
            torch.from_numpy(scores.reshape(-1)).to(torch.from_numpy(rois).dtype),
            torch.zeros_like(torch.from_numpy(scores.reshape(-1))),
            0.7,
            )
            keep = nms_out_index.numpy()
            mask = np.zeros(len(rois), dtype=np.bool)
            mask[keep] = True
            keep = np.where(mask & (scores >= self.min_cls_score))[0]
            detections = [detections[i] for i in keep]
            scores = scores[keep]
            for d, score in zip(detections, scores):
                d.score = score
        pred_dets = [d for d in detections if not d.from_det]
        detections = [d for d in detections if d.from_det]

        # set features
        tlbrs = [det.tlbr for det in detections]
        features = extract_reid_features(self.reid_model, image, tlbrs)
        features = features.cpu().numpy()
        for i, det in enumerate(detections):
            det.set_feature(features[i])

        """step 3: association for tracked"""
        # matching for tracked targets
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        dists = matching.nearest_reid_distance(tracked_stracks, detections, metric='euclidean')
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)
        for itracked, idet in matches:
            tracked_stracks[itracked].update(detections[idet], self.frame_id, image)

        # matching for missing targets
        detections = [detections[i] for i in u_detection]
        dists = matching.nearest_reid_distance(self.lost_stracks, detections, metric='euclidean')
        dists = matching.gate_cost_matrix(self.kalman_filter, dists, self.lost_stracks, detections)
        matches, u_lost, u_detection = matching.linear_assignment(dists, thresh=self.min_ap_dist)
        for ilost, idet in matches:
            track = self.lost_stracks[ilost]  # type: STrack
            det = detections[idet]
            track.re_activate(det, self.frame_id, image, new_id=not self.use_refind)
            refind_stracks.append(track)

        # remaining tracked
        # tracked
        len_det = len(u_detection)
        detections = [detections[i] for i in u_detection] + pred_dets
        r_tracked_stracks = [tracked_stracks[i] for i in u_track]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            r_tracked_stracks[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
        for it in u_track:
            track = r_tracked_stracks[it]
            track.mark_lost()
            lost_stracks.append(track)

        # unconfirmed
        detections = [detections[i] for i in u_detection if i < len_det]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, image, update_feature=True)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """step 4: init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if not track.from_det or track.score < 0.6:
                continue
            track.activate(self.kalman_filter, self.frame_id, image)
            activated_starcks.append(track)

        """step 6: update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.tracked_stracks.extend(activated_starcks)
        self.tracked_stracks.extend(refind_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)

        # output_stracks = self.tracked_stracks + self.lost_stracks

        # get scores of lost tracks
        output_tracked_stracks = [track for track in self.tracked_stracks if track.is_activated]

        output_stracks = output_tracked_stracks

        return output_stracks
    
    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        if isinstance(bbox_xyxy, np.ndarray):
            bbox_tlwh = bbox_xyxy.copy()
        elif isinstance(bbox_xyxy, torch.Tensor):
            bbox_tlwh = bbox_xyxy.clone()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        return bbox_tlwh
