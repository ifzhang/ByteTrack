import sys

sys.path.append("./External_Model_Files/byte_tracking")

import numpy as np

from yolox.tracker.byte_tracker import BYTETracker
from Result_Modules.Results import ObjectTrackerResults
class ByteTracker:
    def __init__(self, args) -> None:
        self.aspect_ratio_thresh = args.aspect_ratio_thresh
        self.min_box_area = args.min_box_area
        self.frame_rate = args.frame_rate
        self.object_tracker = BYTETracker(args, self.frame_rate)

    def infer(self, ori_img, detections):
        image_height = ori_img.shape[0]
        image_width = ori_img.shape[1]

        detections_decoded = []
        for d in detections:
            x_top, y_top, x_bottom, y_bottom = d.bbox
            detections_decoded.append((x_top, y_top, x_bottom, y_bottom, d.confidence))
        detections_decoded = np.array(detections_decoded)
        targets = self.object_tracker.update(output_results=detections_decoded, img_info=[image_height, image_width], img_size=(image_height, image_width))
        
        # post processing
        tracking_results = ObjectTrackerResults()
        for t in targets:
            tlwh = t.tlwh
            x_min, y_min, width, hight = tlwh
            x_max = x_min + width
            y_max = y_min + hight
            tid = t.track_id
            vertical = t.tlwh[2] / t.tlwh[3] > self.aspect_ratio_thresh
            if t.tlwh[2] * t.tlwh[3] > self.min_box_area and not vertical:
                tracking_results.add_tracked_object(bbox=np.array([x_min, y_max, x_max, y_min]), id=tid)
        
        return(tracking_results)
