
class FrontendPosterResults():
    """
        This class is a intended to be used as a standard results class to store
    every results object to be needed in the Processor class.
    """

    def __init__(self):

        self.Frame = None
        self.Store_Map = None
        self.ObjectDetectionResults = None
        self.TrackerResults = None
        self.entry_exit_polygons = None

        self.entry_exit_counts = {
            'entries': None,
            'exits': None
        }

        self.bbox_age_polygon = None

        self.bbox_age_counts = {
            'child_count': None,
            'adult_count': None
        }

        self.HeatmapProjections = {
            'projections': None,
            'boundary': None
        }

        self.age_with_entry_exit_polygons = None
