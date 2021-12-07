import numpy as np
from norfair import Detection, Tracker

from Result_Modules.Results import Results


class Norfair():
    """
    This class is used for real-time 2D objects tracking
    Written by Faizan
    Reference https://github.com/tryolabs/norfair/blob/master/demos/yolov4/yolov4demo.py
    """

    def __init__(self, args):
        self.max_distance_between_points = args.max_dist
        self.min_confidence = args.min_confidence
        self.tracker = None
        self.tracked_objects = None

    def euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def set_model(self):
        print("==========       Loading Norfair Model...             ==========")
        self.tracker = Tracker(distance_function=self.euclidean_distance,
                               distance_threshold=self.max_distance_between_points)

        print("==========       Norfair Model Loaded Successfully!   ==========")

    def get_centroid(self, bbox):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def preprocess(self, detections):
        # pass
        bbox = []
        confs = []

        for det in detections:
            bbox.append(self.get_centroid(det.bbox))
            confs.append(np.array(det.confidence))

        return np.array(bbox), np.array(confs)

    def infer(self, detections, image_frame):
        tracker_results = Results.get_tracker_results()

        # bbox_cxcy, confidences = self.preprocess(detections, image_frame)

        # detections = [Detection(bbox_cxcy) for conf in confidences if conf>self.min_confidence]
        detections = [
            Detection(self.get_centroid(det.bbox), data=det)
            for det in detections
            if det.confidence > self.min_confidence
        ]

        self.tracked_objects = self.tracker.update(detections=detections)

        # for iter, track in enumerate(self.tracked_objects):
        #     if track.live_points.any():
        #         track_id = track.id
        #         tracker_results.add_tracked_object(detections[iter].data.bbox, track_id)
        if len(self.tracked_objects) == len(detections):
            for iter, det in enumerate(detections):
                track_id = self.tracked_objects[iter].id
                tracker_results.add_tracked_object(det.data.bbox, track_id)

        return tracker_results
