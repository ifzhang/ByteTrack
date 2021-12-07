import cv2
from Model_Classes.YoloV5_6 import YoloV5_6
from Model_Classes.byte_tracker import ByteTracker

video_path = "test_samples/stream6_Trim.mp4"

if __name__ == "__main__":
    # Video Capturing initialization
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("im", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()
