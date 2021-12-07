import numpy as np


class TrackedObject:
    """
    This class is a intended to be used as a standard results class to store
    every kind of results for an object tracking task.
    """

    def __init__(self):
        """
            This class is intended to be used in Object Tracker Results.
            Every object will contain the following two variables:
                - bbox: [np.array](xyxy format)
                - Id: The ID associated with the tracked object
        """

        self.bbox = None
        self.Id = None


class ObjectTrackerResults():

    def __init__(self):
        self.Tracked_Objects = []

    def add_tracked_object(self, bbox, id):
        """
            This function expects the following three inputs in the said format:
                - bbox: [np.array](x-top, y-top, x-bottom, y-bottom format)
                - Id: [int]
        """

        ########    TYPE CHECKING TO FOR STANDARDIZATION    ########
        assert isinstance(bbox, np.ndarray), Exception("Bounding Box Must be a numpy array")
        assert isinstance(id, int), Exception("ID value must be an Integer")

        ########    VALUE SETTING IF TYPES VALID    ########
        tracked_obj = TrackedObject()
        tracked_obj.bbox = bbox
        tracked_obj.Id = id

        ########    APPENDING TO THE MAIN DETECTIONS LIST    ########
        self.Tracked_Objects.append(tracked_obj)
