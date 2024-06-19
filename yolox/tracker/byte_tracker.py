import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState
from typing import List

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        self.features = []
        self.feat_framenum = 0
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
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
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
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

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args,extractor, frame_rate=30):
        self.extractor = extractor
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, frame, output_results, img_info, img_size):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        print("IW...1")
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        print("IW...2")
###############################################################################################
################################# OUR EDIT ####################################################
        detections = [detections[i] for i in u_detection]
        cropped_regions = [crop_roi(frame,det) for det in detections]
        
        if cropped_regions:
            det_features = self.extractor(cropped_regions)
            for track in self.removed_stracks:
                print("feat_len",len(track.features))
                # print("active?:",track.is_activated)
            rm_stracks = [track for track in self.removed_stracks if (track.state==TrackState.Removed and len(track.features)>0)]
            print("hello")
            rm_stracks_feat = [tr.features for tr in rm_stracks]
            print("nice...")
            track_iS, det_iS, u_detection = find_best_matches(rm_stracks_feat,det_features)
            print("IW...3")
            print("MATCHING:",len(track_iS),len(det_iS),len(rm_stracks),len(self.removed_stracks),len(cropped_regions))
            rm_activated = []
            for tracki, deti in zip(track_iS,det_iS):
                detection= detections[deti] 
                rm_track = rm_stracks[tracki]
                if rm_track.state == TrackState.Tracked:
                    rm_track.update(detection,self.frame_id)
                    activated_starcks.append(rm_track)            
                else:
                    rm_track.re_activate(detection, self.frame_id, new_id=False)
                    refind_stracks.append(rm_track)
                rm_activated.append(rm_track)
            
            self.removed_stracks = sub_stracks(self.removed_stracks,rm_activated)
######################################## OUR EDIT #############################################
###############################################################################################

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        print("IW...4")
###################################################################################################
######################################### OUR EDIT ################################################
        for track in output_stracks:
            cropped_region = crop_roi(frame=frame,strack=track)
            cropped_feature = self.extractor(cropped_region)
            if (track.tracklet_len in [0,1,2,5,10,30]) or self.frame_id-track.feat_framenum>50:
            # if (self.frame_id-track.feat_framenum)<50 and self.frame_id>3 and track.feat_framenum>3:
            #     continue
                if len(track.features)>=5:
                    track.features.pop(0)
                track.features.append(cropped_feature)
                track.feat_framenum = self.frame_id
        print("IW...5")
######################################### OUR EDIT ################################################
###################################################################################################
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def crop_roi(frame, strack):
    """Crops a region of interest from a frame based on STrack coordinates.

    Args:
        frame: The input image/frame (NumPy array).
        strack: An STrack object containing detection coordinates (tlwh format).
        margin: Optional margin (in pixels) to add around the cropped region.

    Returns:
        The cropped region of interest as a NumPy array.
    """

    tlbr = strack.tlbr
    x1, y1, x2, y2 = tlbr

    # Add margins
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Crop the ROI
    roi = frame[y1:y2, x1:x2]

    return roi

def find_best_matches(l1:List[List[torch.Tensor]], l2:torch.Tensor, threshold=0.4):
    """Finds the absolute best matches between features in a nested list (l1) and a flat list (l2).

    Args:
        l1: A nested list where each inner list contains PyTorch tensors representing features.
        l2: A list of PyTorch tensors representing features.
        threshold: The cosine similarity threshold to consider a match valid.

    Returns:
        track_indices: A list of integers representing the track indices where the absolute best matches were found.
        feature_indices_l1: A list of integers representing the feature indices within l1 for the absolute best matches.
        feature_indices_l2: A list of integers representing the feature indices within l2.
    """
    
    track_indices = []
    det_i = []
    u_detection = []
    for det_index, feature2 in enumerate(l2):
        best_similarity = -1
        best_track_index = None

        for track_index, track_features in enumerate(l1):
            for feature1 in track_features:
                similarity = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0)
                similarity = torch.mean(similarity).item()
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_track_index = track_index

        if best_track_index is not None:
            track_indices.append(best_track_index)
            det_i.append(det_index)
        else:
            u_detection.append(det_index)
    return track_indices, det_i, u_detection

# def compare_features(features1, features2):
#     """Calculates cosine similarities between two lists of features and returns the indices of maximum matches.

#     Args:
#         features1: A list of PyTorch tensors representing features.
#         features2: Another list of PyTorch tensors representing features.

#     Returns:
#         A list of tuples (row_index, column_index) where each tuple represents the indices 
#         of the maximum cosine similarity for each feature in features1.
#     """

#     similarity_matrix = torch.zeros(len(features1), len(features2))  # Initialize matrix

#     for i, feature1 in enumerate(features1):
#         for j, feature2 in enumerate(features2):
#             similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0)
            
#     # Find the column index with the maximum similarity for each row
#     max_indices = torch.argmax(similarity_matrix, dim=1)

#     # Get the row indices
#     row_indices = torch.arange(len(features1))

#     # Create a list of tuples (row_index, column_index)
#     max_similarity_indices = list(zip(row_indices.tolist(), max_indices.tolist()))

#     return max_similarity_indices
