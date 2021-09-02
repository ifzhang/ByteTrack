import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import copy
from sklearn.metrics.pairwise import cosine_similarity as cosine


class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.reset()
        self.nID = 10000
        self.alpha = 0.1

    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                # active and age are never used in the paper
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)
                self.nID = 10000
                self.embedding_bank = np.zeros((self.nID, 128))
                self.cat_bank = np.zeros((self.nID), dtype=np.int)

    def reset(self):
        self.id_count = 0
        self.nID = 10000
        self.tracks = []
        self.embedding_bank = np.zeros((self.nID, 128))
        self.cat_bank = np.zeros((self.nID), dtype=np.int)
        self.tracklet_ages = np.zeros((self.nID), dtype=np.int)
        self.alive = []

    def step(self, results_with_low, public_det=None):
        results = [item for item in results_with_low if item['score'] >= self.opt.track_thresh]
        
        # first association
        N = len(results)
        M = len(self.tracks)
        self.alive = []

        track_boxes = np.array([[track['bbox'][0], track['bbox'][1],
                                 track['bbox'][2], track['bbox'][3]] for track in self.tracks], np.float32)  # M x 4
        det_boxes = np.array([[item['bbox'][0], item['bbox'][1],
                               item['bbox'][2], item['bbox'][3]] for item in results], np.float32)  # N x 4
        box_ious = self.bbox_overlaps_py(det_boxes, track_boxes)

        dets = np.array(
            [det['ct'] + det['tracking'] for det in results], np.float32)  # N x 2
        track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
                                (track['bbox'][3] - track['bbox'][1])) \
                               for track in self.tracks], np.float32)  # M
        track_cat = np.array([track['class'] for track in self.tracks], np.int32)  # M
        item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
                               (item['bbox'][3] - item['bbox'][1])) \
                              for item in results], np.float32)  # N
        item_cat = np.array([item['class'] for item in results], np.int32)  # N
        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2
        dist = (((tracks.reshape(1, -1, 2) - \
                  dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

        if self.opt.dataset == 'youtube_vis':
            invalid = ((dist > track_size.reshape(1, M)) + \
                       (dist > item_size.reshape(N, 1)) + (box_ious < self.opt.overlap_thresh)) > 0
        else:
            invalid = ((dist > track_size.reshape(1, M)) + \
                       (dist > item_size.reshape(N, 1)) + \
                       (item_cat.reshape(N, 1) != track_cat.reshape(1, M)) + (box_ious < self.opt.overlap_thresh)) > 0
        dist = dist + invalid * 1e18

        if self.opt.hungarian:
            item_score = np.array([item['score'] for item in results], np.float32)  # N
            dist[dist > 1e18] = 1e18
            matched_indices = linear_assignment(dist)
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))
        unmatched_dets = [d for d in range(dets.shape[0]) \
                          if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) \
                            if not (d in matched_indices[:, 1])]

        if self.opt.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ret = []
        for m in matches:
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['active'] = self.tracks[m[1]]['active'] + 1
            if 'embedding' in track:
                self.alive.append(track['tracking_id'])
                self.embedding_bank[self.tracks[m[1]]['tracking_id'] - 1, :] = self.alpha * track['embedding'] \
                                                                               + (1 - self.alpha) * self.embedding_bank[
                                                                                                    self.tracks[m[1]][
                                                                                                        'tracking_id'] - 1,
                                                                                                    :]
                self.cat_bank[self.tracks[m[1]]['tracking_id'] - 1] = track['class']
            ret.append(track)

        if self.opt.public_det and len(unmatched_dets) > 0:
            # Public detection: only create tracks from provided detections
            pub_dets = np.array([d['ct'] for d in public_det], np.float32)
            dist3 = ((dets.reshape(-1, 1, 2) - pub_dets.reshape(1, -1, 2)) ** 2).sum(
                axis=2)
            matched_dets = [d for d in range(dets.shape[0]) \
                            if not (d in unmatched_dets)]
            dist3[matched_dets] = 1e18
            for j in range(len(pub_dets)):
                i = dist3[:, j].argmin()
                if dist3[i, j] < item_size[i]:
                    dist3[i, :] = 1e18
                    track = results[i]
                    if track['score'] > self.opt.new_thresh:
                        self.id_count += 1
                        track['tracking_id'] = self.id_count
                        track['age'] = 1
                        track['active'] = 1
                        ret.append(track)
        else:
            # Private detection: create tracks for all un-matched detections
            for i in unmatched_dets:
                track = results[i]
                if track['score'] > self.opt.new_thresh:
                    if 'embedding' in track:
                        max_id, max_cos = self.get_similarity(track['embedding'], False, track['class'])
                        if max_cos >= 0.3 and self.tracklet_ages[max_id - 1] < self.opt.window_size:
                            track['tracking_id'] = max_id
                            track['age'] = 1
                            track['active'] = 1
                            self.embedding_bank[track['tracking_id'] - 1, :] = self.alpha * track['embedding'] \
                                                                               + (1 - self.alpha) * self.embedding_bank[track['tracking_id'] - 1,:]
                        else:
                            self.id_count += 1
                            track['tracking_id'] = self.id_count
                            track['age'] = 1
                            track['active'] = 1
                            self.embedding_bank[self.id_count - 1, :] = track['embedding']
                            self.cat_bank[self.id_count - 1] = track['class']
                        self.alive.append(track['tracking_id'])
                        ret.append(track)
                    else:
                        self.id_count += 1
                        track['tracking_id'] = self.id_count
                        track['age'] = 1
                        track['active'] = 1
                        ret.append(track)

        self.tracklet_ages[:self.id_count] = self.tracklet_ages[:self.id_count] + 1
        for track in ret:
            self.tracklet_ages[track['tracking_id'] - 1] = 1
        
        
        # second association
        results_second = [item for item in results_with_low if item['score'] < self.opt.track_thresh]
        self_tracks_second = [self.tracks[i] for i in unmatched_tracks if self.tracks[i]['active'] > 0]
        second2original = [i for i in unmatched_tracks if self.tracks[i]['active'] > 0]
        
        N = len(results_second)
        M = len(self_tracks_second)
        
        if N > 0 and M > 0:

            track_boxes_second = np.array([[track['bbox'][0], track['bbox'][1],
                                 track['bbox'][2], track['bbox'][3]] for track in self_tracks_second], np.float32)  # M x 4
            det_boxes_second = np.array([[item['bbox'][0], item['bbox'][1],
                                  item['bbox'][2], item['bbox'][3]] for item in results_second], np.float32)  # N x 4
            box_ious_second = self.bbox_overlaps_py(det_boxes_second, track_boxes_second)

            dets = np.array(
                [det['ct'] + det['tracking'] for det in results_second], np.float32)  # N x 2
            track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
                                    (track['bbox'][3] - track['bbox'][1])) \
                                   for track in self_tracks_second], np.float32)  # M
            track_cat = np.array([track['class'] for track in self_tracks_second], np.int32)  # M
            item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
                                   (item['bbox'][3] - item['bbox'][1])) \
                                  for item in results_second], np.float32)  # N
            item_cat = np.array([item['class'] for item in results_second], np.int32)  # N
            tracks_second = np.array(
                [pre_det['ct'] for pre_det in self_tracks_second], np.float32)  # M x 2
            dist = (((tracks_second.reshape(1, -1, 2) - \
                      dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

            invalid = ((dist > track_size.reshape(1, M)) + \
                       (dist > item_size.reshape(N, 1)) + \
                       (item_cat.reshape(N, 1) != track_cat.reshape(1, M)) + (box_ious_second < 0.3)) > 0
            dist = dist + invalid * 1e18
            
            matched_indices_second = greedy_assignment(copy.deepcopy(dist), 1e8)
            unmatched_tracks_second = [d for d in range(tracks_second.shape[0]) \
                                       if not (d in matched_indices_second[:, 1])]                        
            matches_second = matched_indices_second            
            
            for m in matches_second:
                track = results_second[m[0]]
                track['tracking_id'] = self_tracks_second[m[1]]['tracking_id']
                track['age'] = 1
                track['active'] = self_tracks_second[m[1]]['active'] + 1
                if 'embedding' in track:
                    self.alive.append(track['tracking_id'])
                    self.embedding_bank[self_tracks_second[m[1]]['tracking_id'] - 1, :] = self.alpha * track['embedding'] \
                                                                                   + (1 - self.alpha) * self.embedding_bank[self_tracks_second[m[1]]['tracking_id'] - 1,:]
                    self.cat_bank[self_tracks_second[m[1]]['tracking_id'] - 1] = track['class']
                ret.append(track)            
            
            unmatched_tracks = [second2original[i] for i in unmatched_tracks_second] + \
            [i for i in unmatched_tracks if self.tracks[i]['active'] == 0]
        
        
        # Never used
        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.opt.max_age:
                track['age'] += 1
                track['active'] = 1  # 0
                bbox = track['bbox']
                ct = track['ct']
                v = [0, 0]
                track['bbox'] = [
                    bbox[0] + v[0], bbox[1] + v[1],
                    bbox[2] + v[0], bbox[3] + v[1]]
                track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                ret.append(track)
        for r_ in ret:
            del r_['embedding']
        self.tracks = ret
        return ret

    def get_similarity(self, feat, stat, cls):
        max_id = -1
        max_cos = -1
        if stat:
            nID = self.id_count
        else:
            nID = self.id_count

        a = feat[None, :]
        b = self.embedding_bank[:nID, :]
        if len(b) > 0:
            alive = np.array(self.alive, dtype=np.int) - 1
            cosim = cosine(a, b)
            cosim = np.reshape(cosim, newshape=(-1))
            cosim[alive] = -2
            cosim[nID - 1] = -2
            cosim[np.where(self.cat_bank[:nID] != cls)[0]] = -2
            max_id = int(np.argmax(cosim) + 1)
            max_cos = np.max(cosim)
        return max_id, max_cos

    def bbox_overlaps_py(self, boxes, query_boxes):
        """
        determine overlaps between boxes and query_boxes
        :param boxes: n * 4 bounding boxes
        :param query_boxes: k * 4 bounding boxes
        :return: overlaps: n * k overlaps
        """
        n_ = boxes.shape[0]
        k_ = query_boxes.shape[0]
        overlaps = np.zeros((n_, k_), dtype=np.float)
        for k in range(k_):
            query_box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            for n in range(n_):
                iw = min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1
                if iw > 0:
                    ih = min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1
                    if ih > 0:
                        box_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                        all_area = float(box_area + query_box_area - iw * ih)
                        overlaps[n, k] = iw * ih / all_area
        return overlaps



def greedy_assignment(dist, thresh=1e16):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < thresh:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)
