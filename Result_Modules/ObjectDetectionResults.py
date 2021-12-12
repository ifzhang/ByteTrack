import numpy as np


class Detection:
    """
    This class is a intended to be used as a standard results class to store
    every kind of results for an object detection task.
    """

    def __init__(self):
        """
            This class is intended to be used in Object Detection Results.
            Every object will contain the following three variables:
                - bbox: [np.array](xyxy format)
                - confidence: [float]
                - class_name: [string]
        """

        self.bbox = None
        self.confidence = None
        self.class_name = None


class ObjectDetectionResults():

    def __init__(self):
        self.Detections = []

    def add_detection(self, bbox, confidence, class_name):
        """
            This function expects the following three inputs in the said format:
                - bbox: [np.array](x-top, y-top, x-bottom, y-bottom format)
                - confidence: [float]
                - class_name: [string]
        """

        ########    TYPE CHECKING TO FOR STANDARDIZATION    ########
        assert isinstance(bbox, np.ndarray), Exception("Bounding Box Must be a numpy array")
        assert isinstance(confidence, float), Exception("Confidence value must be a floating point number")
        assert isinstance(class_name, str), Exception("Class Name must be a string")

        ########    VALUE SETTING IF TYPES VALID    ########
        det = Detection()
        det.bbox = bbox
        det.confidence = confidence
        det.class_name = class_name

        ########    APPENDING TO THE MAIN DETECTIONS LIST    ########
        self.Detections.append(det)
