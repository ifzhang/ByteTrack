import time

import cv2
from Model_Classes.YoloV5_6 import YoloV5_6
from Model_Classes.byte_tracker import ByteTracker
from Model_Classes.deep_sort import DeepSort

video_path = "intersection_sample.mp4"
deep_sort_times = []
byte_track_times = []

class YoloV6DefaultArgs():
    def __init__(self):
        self.weights = './weights/yolov5s6.pt'
        self.device = "cpu"
        self.img_size = 1280
        self.conf_thresh = 0.5
        self.iou_thresh = 0.5
        self.agnostic_nms = False
        self.augment = False
        self.classes = None
        self.filter_persons = True


class ByteTrackerDefaultArgs():
    def __init__(self) -> None:
        self.track_thresh = 0.5
        self.mot20 = False
        self.match_thresh = 0.8
        self.track_buffer = 30
        self.frame_rate = 25
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10


class DeepsortDefaultArgs():
    def __init__(self):
        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0
        self.model_path = "./weights/deepsort_base_vRepo.t7"
        self.use_cuda = True
        self.max_dist = 0.2
        self.nn_budget = 10


def show_bounding_boxes(image, results):
    if len(results) == 0:
        return image
    for d in results:
        bbox = d.bbox
        cn = d.class_name
        conf = d.confidence
        xmin, xmax, ymin, ymax = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
        pt1 = (xmin, ymax)
        pt2 = (xmax, ymin)
        cv2.rectangle(image, pt1, pt2, (0,255,0), 2)
        return image
        

def show_tracking_results(image, tracking_results):
    if len(tracking_results.Tracked_Objects) == 0:
        return image

    for t in tracking_results.Tracked_Objects:
        bbox = t.bbox
        id = t.Id
        xmin, xmax, ymin, ymax = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
        pt1 = (xmin, ymax)
        pt2 = (xmax, ymin)
        print(pt1, pt2)
        cv2.rectangle(image, pt1, pt2, (0,255,0), 2)
        cv2.putText(image, str(id), pt1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    
    return image


if __name__ == "__main__":
    # Video Capturing initialization
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output_test_intersect_byte.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    yolo_args = YoloV6DefaultArgs()
    person_detector = YoloV5_6(yolo_args)
    person_detector.set_model()

    byte_args = ByteTrackerDefaultArgs()
    byte_person_tracker = ByteTracker(byte_args)

    # deep_sort_default_aruments = DeepsortDefaultArgs()
    # deep_person_tracker = DeepSort(deep_sort_default_aruments)
    # deep_person_tracker.set_model() 

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            object_detection_results = person_detector.infer(frame)
            
            # s = time.time()
            # deep_sort_object_tracking_results  = deep_person_tracker.infer(ori_img=frame, detections=object_detection_results.Detections)
            # e = time.time()
            # deep_sort_times.append(e-s)

            s = time.time()
            byte_track_object_tracking_results  = byte_person_tracker.infer(ori_img=frame, detections=object_detection_results.Detections)
            e = time.time()
            byte_track_times.append(e-s)
            detected_frame = show_tracking_results(frame, byte_track_object_tracking_results) 
            out.write(detected_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
            pass
    
    # print(f"Deep sort avg times: {sum(deep_sort_times)/len(deep_sort_times):.5f} seconds")
    print(f"Byte track avg times: {sum(byte_track_times)/len(byte_track_times):.5f} seconds")
    
    cap.release()
    cv2.destroyAllWindows()
