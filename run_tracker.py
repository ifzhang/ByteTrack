import cv2
from Model_Classes.YoloV5_6 import YoloV5_6
from Model_Classes.byte_tracker import ByteTracker

video_path = "test_samples/stream6_Trim.mp4"

class YoloV6DefaultArgs():
    def __init__(self):
        self.weights = './weights/yolov5s6.pt'
        self.device = "0"
        self.img_size = 640
        self.conf_thresh = 0.5
        self.iou_thresh = 0.5
        self.agnostic_nms = False
        self.augment = False
        self.classes = None
        self.filter_persons = True

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
        

if __name__ == "__main__":
    # Video Capturing initialization
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    yolo_args = YoloV6DefaultArgs()
    person_detector = YoloV5_6(yolo_args)
    person_detector.set_model()


    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            results = person_detector.infer(frame)
            results = results.Detections
            detected_frame = show_bounding_boxes(frame, results)
            cv2.imshow("im", detected_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
