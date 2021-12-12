import numpy as np

from External_Model_Files.deep_sort.deep.feature_extractor import Extractor
from External_Model_Files.deep_sort.sort.detection import Detection
from External_Model_Files.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from External_Model_Files.deep_sort.sort.preprocessing import non_max_suppression
from External_Model_Files.deep_sort.sort.tracker import Tracker
from helpers import Helpers
from Result_Modules.Results import Results


class DeepSort():
    """
    This class sets the deepsort model used for object tracking
    """

    def __init__(self, args):
        self.min_confidence = args.min_confidence
        self.nms_max_overlap = args.nms_max_overlap
        self.model_path = args.model_path
        self.extractor = Extractor(self.model_path, use_cuda=args.use_cuda)

        self.max_cosine_distance = args.max_dist

        self.nn_budget = args.nn_budget  # 10 is best

        self.tracker = None
        self.metric = None

    def set_model(self):

        print("==========       Loading DeepSort Model...             ==========")

        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        self.tracker = Tracker(metric)
        self.metric = metric

        print("==========       DeepSort Model Loaded Successfully!   ==========")

    def preprocess(self, detections):

        bbox_xcycwhs = []
        confs = []
        for det in detections:
            bbox_xcycwhs.append(np.array(Helpers._txtybxby_to_cxcywh(det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3])))
            confs.append(np.array(det.confidence))

        return np.array(bbox_xcycwhs), np.array(confs)

    def infer(self, detections, ori_img, cached_features=None):

        tracker_results = Results.get_tracker_results()

        bbox_xywh, confidences = self.preprocess(detections)
        self.height, self.width = ori_img.shape[0], ori_img.shape[1]

        if not cached_features:
            features = self._get_features(bbox_xywh, ori_img)
        else:
            features = cached_features

        bbox_tlwh = Helpers._xcycwh_to_tlwh(bbox_xywh)

        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()

        # tracker.update returns features...
        self.tracker.update(detections)

        # output bbox identities
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = Helpers._tlwh_to_txtybxby(box, self.width, self.height)
            track_id = track.track_id
            tracker_results.add_tracked_object(np.array([x1, y1, x2, y2]), track_id)

        return tracker_results

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = Helpers._cxcywh_to_txtybxby(box, self.width, self.height)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)

        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])

        return features

    def get_batch_features(self, person_rois, img_batch):
        im_crops = []
        img_idx_mask = []
        for box in person_rois:
            x1, y1, x2, y2 = (box.position[0].item(), box.position[1].item(), box.position[0].item() + box.size[0].item(), box.position[1].item() + box.size[1].item())
            img_idx_mask.append(box.img_idx)
            im = img_batch[box.img_idx][y1:y2, x1:x2]
            im_crops.append(im)

        img_idx_mask = np.array(img_idx_mask)

        if im_crops:

            features = self.extractor(im_crops)
        else:
            features = np.array([])

        return features, img_idx_mask
