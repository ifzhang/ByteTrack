import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
# from numba import jit
import copy


class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.reset()

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

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def step(self, results_with_low, public_det=None):
        
        results = [item for item in results_with_low if item['score'] >= self.opt.track_thresh]
        
        # first association
        N = len(results)
        M = len(self.tracks)

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

        invalid = ((dist > track_size.reshape(1, M)) + \
                   (dist > item_size.reshape(N, 1)) + \
                   (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
        dist = dist + invalid * 1e18
        
        if self.opt.hungarian:
            assert not self.opt.hungarian, 'we only verify centertrack with greedy_assignment'
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
            assert not self.opt.hungarian, 'we only verify centertrack with greedy_assignment'
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
            ret.append(track)
        
        if self.opt.public_det and len(unmatched_dets) > 0:
            assert not self.opt.public_det, 'we only verify centertrack with private detection'
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
                    self.id_count += 1
                    track['tracking_id'] = self.id_count
                    track['age'] = 1
                    track['active'] = 1
                    ret.append(track)
        
        # second association
        results_second = [item for item in results_with_low if item['score'] < self.opt.track_thresh]
        
        self_tracks_second = [self.tracks[i] for i in unmatched_tracks if self.tracks[i]['active'] > 0]
        second2original = [i for i in unmatched_tracks if self.tracks[i]['active'] > 0]
        
        N = len(results_second)
        M = len(self_tracks_second)
        
        if N > 0 and M > 0:
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
                       (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
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
                ret.append(track)
                        
            unmatched_tracks = [second2original[i] for i in unmatched_tracks_second] + \
            [i for i in unmatched_tracks if self.tracks[i]['active'] == 0]

#.      for debug        
#         unmatched_tracks = [i for i in unmatched_tracks if self.tracks[i]['active'] > 0] + \
#         [i for i in unmatched_tracks if self.tracks[i]['active'] == 0]
    
        for i in unmatched_tracks:
            track = self.tracks[i]
            if track['age'] < self.opt.max_age:
                track['age'] += 1
                track['active'] = 0
                bbox = track['bbox']
                ct = track['ct']
                v = [0, 0]
                track['bbox'] = [
                    bbox[0] + v[0], bbox[1] + v[1],
                    bbox[2] + v[0], bbox[3] + v[1]]
                track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                ret.append(track)
        self.tracks = ret
        return ret


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
