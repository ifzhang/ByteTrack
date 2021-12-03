from collections import deque
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary

from core.mot.general import non_max_suppression_and_inds, non_max_suppression_jde, non_max_suppression, scale_coords
from core.mot.torch_utils import intersect_dicts
from models.mot.cstrack import Model

from mot_online import matching
from mot_online.kalman_filter import KalmanFilter
from mot_online.log import logger
from mot_online.utils import *

from mot_online.basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

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
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
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
        if update_feature:
            self.update_features(new_track.curr_feat)

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


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if int(opt.gpus[0]) >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')

        ckpt = torch.load(opt.weights, map_location=opt.device)  # load checkpoint
        self.model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=1).to(opt.device)  # create
        exclude = ['anchor'] if opt.cfg else []  # exclude keys
        if type(ckpt['model']).__name__ == "OrderedDict":
            state_dict = ckpt['model']
        else:
            state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
        self.model.load_state_dict(state_dict, strict=False)  # load
        self.model.cuda().eval()
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')


        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.kalman_filter = KalmanFilter()
        self.low_thres = 0.1
        self.high_thres = self.opt.conf_thres + 0.1

    def update(self, im_blob, img0,seq_num, save_dir):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        dets = []

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob, augment=False)
            pred, train_out = output[1]

        pred = pred[pred[:, :, 4] > self.low_thres]
        detections = []
        if len(pred) > 0:
            dets,x_inds,y_inds = non_max_suppression_and_inds(pred[:,:6].unsqueeze(0), 0.1, self.opt.nms_thres,method='cluster_diou')
            dets = dets.numpy()
            if len(dets) != 0:
                scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
                id_feature = output[0][0, y_inds, x_inds, :].cpu().numpy()

                remain_inds = dets[:, 4] > self.opt.conf_thres
                inds_low = dets[:, 4] > self.low_thres
                inds_high = dets[:, 4] < self.opt.conf_thres
                inds_second = np.logical_and(inds_low, inds_high)
                dets_second = dets[inds_second]
                if id_feature.shape[0] == 1:
                    id_feature_second = id_feature
                else:
                    id_feature_second = id_feature[inds_second]
                dets = dets[remain_inds]
                id_feature = id_feature[remain_inds]

                detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                              (tlbrs, f) in zip(dets[:, :5], id_feature)]
                
            else:
                detections = []
                dets_second = []
                id_feature_second = []
        else:
            detections = []
            dets_second = []
            id_feature_second = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # vis
        track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track = [],[],[],[],[]
        if self.opt.vis_state == 1 and self.frame_id % 20 == 0:
            if len(dets) != 0:
                for i in range(0, dets.shape[0]):
                    bbox = dets[i][0:4]
                    cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0, 255, 0), 2)
                track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track = matching.vis_id_feature_A_distance(strack_pool, detections)
            vis_feature(self.frame_id,seq_num,img0,track_features,
                                  det_features, cost_matrix, cost_matrix_det, cost_matrix_track, max_num=5, out_path=save_dir)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                                 (tlbrs, f) in zip(dets_second[:, :5], id_feature_second)]
        else:
            detections_second = []
        second_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
        dists = matching.iou_distance(second_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.4)
        for itracked, idet in matches:
            track = second_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = second_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.high_thres:
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

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

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

def vis_feature(frame_id,seq_num,img,track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track,max_num=5, out_path='/home/XX/'):
    num_zero = ["0000","000","00","0"]
    img = cv2.resize(img, (778, 435))

    if len(det_features) != 0:
        max_f = det_features.max()
        min_f = det_features.min()
        det_features = np.round((det_features - min_f) / (max_f - min_f) * 255)
        det_features = det_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*512
        for d_f in det_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        det_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(det_features_img, (435, 435))
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "det_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((img, feature_img2), axis=1)

    if len(cost_matrix_det) != 0 and len(cost_matrix_det[0]) != 0:
        max_f = cost_matrix_det.max()
        min_f = cost_matrix_det.min()
        cost_matrix_det = np.round((cost_matrix_det - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_det)*10
        for c_m in cost_matrix_det:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_det_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_det_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_det", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(track_features) != 0:
        max_f = track_features.max()
        min_f = track_features.min()
        track_features = np.round((track_features - min_f) / (max_f - min_f) * 255)
        track_features = track_features.astype(np.uint8)
        d_F_M = []
        cutpff_line = [40]*512
        for d_f in track_features:
            for row in range(45):
                d_F_M += [[40]*3+d_f.tolist()+[40]*3]
            for row in range(3):
                d_F_M += [[40]*3+cutpff_line+[40]*3]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        track_features_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(track_features_img, (435, 435))
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "track_features", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix_track) != 0 and len(cost_matrix_track[0]) != 0:
        max_f = cost_matrix_track.max()
        min_f = cost_matrix_track.min()
        cost_matrix_track = np.round((cost_matrix_track - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix_track)*10
        for c_m in cost_matrix_track:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_track_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_track_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix_track", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    if len(cost_matrix) != 0 and len(cost_matrix[0]) != 0:
        max_f = cost_matrix.max()
        min_f = cost_matrix.min()
        cost_matrix = np.round((cost_matrix - min_f) / (max_f - min_f) * 255)
        d_F_M = []
        cutpff_line = [40]*len(cost_matrix[0])*10
        for c_m in cost_matrix:
            add = []
            for row in range(len(c_m)):
                add += [255-c_m[row]]*10
            for row in range(10):
                d_F_M += [[40]+add+[40]]
        d_F_M = np.array(d_F_M)
        d_F_M = d_F_M.astype(np.uint8)
        cost_matrix_img = cv2.applyColorMap(d_F_M, cv2.COLORMAP_JET)
        feature_img2 = cv2.resize(cost_matrix_img, (435, 435))
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        feature_img2 = np.zeros((435, 435))
        feature_img2 = feature_img2.astype(np.uint8)
        feature_img2 = cv2.applyColorMap(feature_img2, cv2.COLORMAP_JET)
        #cv2.putText(feature_img2, "cost_matrix", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    feature_img = np.concatenate((feature_img, feature_img2), axis=1)

    dst_path = out_path + "/" + seq_num + "_" + num_zero[len(str(frame_id))-1] + str(frame_id) + '.png'
    cv2.imwrite(dst_path, feature_img)
