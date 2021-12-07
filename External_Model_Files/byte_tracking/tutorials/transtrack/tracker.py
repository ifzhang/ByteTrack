"""
Copyright (c) https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun, Rufeng Zhang
"""
# coding: utf-8
import torch
from scipy.optimize import linear_sum_assignment
from util import box_ops
import copy

class Tracker(object):
    def __init__(self, score_thresh, max_age=32):        
        self.score_thresh = score_thresh
        self.low_thresh = 0.2
        self.high_thresh = score_thresh + 0.1
        self.max_age = max_age        
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
        self.reset_all()
        
    def reset_all(self):
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
    
    def init_track(self, results):

        scores = results["scores"]
        classes = results["labels"]
        bboxes = results["boxes"]  # x1y1x2y2
        
        ret = list()
        ret_dict = dict()
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["tracking_id"] = self.id_count
                obj['active'] = 1
                obj['age'] = 1
                ret.append(obj)
                ret_dict[idx] = obj
        
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)

    
    def step(self, output_results):
        scores = output_results["scores"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2
        
        results = list()
        results_dict = dict()
        results_second = list()

        tracks = list()
        
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None:
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()               
                results.append(obj)        
                results_dict[idx] = obj
            elif scores[idx] >= self.low_thresh:
                second_obj = dict()
                second_obj["score"] = float(scores[idx])
                second_obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                results_second.append(second_obj)
                results_dict[idx] = second_obj
        
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        # for trackss in tracks:
        #     print(trackss.keys())
        N = len(results)
        M = len(tracks)
        
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]

        if N > 0 and M > 0:
            det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4                
            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M

            matched_indices = linear_sum_assignment(cost_bbox)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

            matches = [[],[]]
            for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                if cost_bbox[m0, m1] > 1.2:
                    unmatched_dets.append(m0)
                    unmatched_tracks.append(m1)
                else:
                    matches[0].append(m0)
                    matches[1].append(m1)

            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                ret.append(track)
        
        # second association
        N_second = len(results_second)
        unmatched_tracks_obj = list()
        for i in unmatched_tracks:
            #print(tracks[i].keys())
            track = tracks[i]
            if track['active'] == 1:
                unmatched_tracks_obj.append(track)
        M_second = len(unmatched_tracks_obj)
        unmatched_tracks_second = [t for t in range(M_second)]

        if N_second > 0 and M_second > 0:
            det_box_second = torch.stack([torch.tensor(obj['bbox']) for obj in results_second], dim=0) # N_second x 4        
            track_box_second = torch.stack([torch.tensor(obj['bbox']) for obj in unmatched_tracks_obj], dim=0) # M_second x 4                
            cost_bbox_second = 1.0 - box_ops.generalized_box_iou(det_box_second, track_box_second) # N_second x M_second

            matched_indices_second = linear_sum_assignment(cost_bbox_second)
            unmatched_tracks_second = [d for d in range(M_second) if not (d in matched_indices_second[1])]

            matches_second = [[],[]]
            for (m0, m1) in zip(matched_indices_second[0], matched_indices_second[1]):
                if cost_bbox_second[m0, m1] > 0.8:
                    unmatched_tracks_second.append(m1)
                else:
                    matches_second[0].append(m0)
                    matches_second[1].append(m1)

            for (m0, m1) in zip(matches_second[0], matches_second[1]):
                track = results_second[m0]
                track['tracking_id'] = unmatched_tracks_obj[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                ret.append(track)

        for i in unmatched_dets:
            trackd = results[i]
            if trackd["score"] >= self.high_thresh:
                self.id_count += 1
                trackd['tracking_id'] = self.id_count
                trackd['age'] = 1
                trackd['active'] =  1
                ret.append(trackd)
        
        # ------------------------------------------------------ #
        ret_unmatched_tracks = []
        
        for j in unmatched_tracks:
            track = tracks[j]
            if track['active'] == 0 and track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ret.append(track)
                ret_unmatched_tracks.append(track)   

        for i in unmatched_tracks_second:
            track = unmatched_tracks_obj[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ret.append(track)
                ret_unmatched_tracks.append(track)
        
        # for i in unmatched_tracks:
        #     track = tracks[i]
        #     if track['age'] < self.max_age:
        #         track['age'] += 1
        #         track['active'] = 0
        #         ret.append(track)
        #         ret_unmatched_tracks.append(track)
        #print(len(ret_unmatched_tracks))
        
        self.tracks = ret
        self.tracks_dict = {red_ind:red for red_ind, red in results_dict.items() if 'tracking_id' in red}
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)
