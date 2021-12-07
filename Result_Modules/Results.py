from Result_Modules.ObjectDetectionResults import ObjectDetectionResults
from Result_Modules.ObjectTrackerResults import ObjectTrackerResults
from Result_Modules.VisualizerResults import VisualizerResults


class Results(object):
    """
    This is the Main Results class. The Purpose of this module is to
    provide with static functions which return every kind of Result
    modules as per the requirement.

    This Module will act as the hub to retrieve every other kind of
    Results module.

    Args:
        object (class): [description]
    """
    ########    GETTING AN OBJECT DETECTION RESULTS CLASS    ########
    @staticmethod
    def get_objectDetection_results():
        return ObjectDetectionResults()

    ########    GETTING A PROCESSOR RESULTS CLASS    ########
    @staticmethod
    def get_visualizer_results():
        return VisualizerResults()

    ########    GETTING AN OBJECT TRACKER RESULTS CLASS    ########
    @staticmethod
    def get_tracker_results():
        return ObjectTrackerResults()
