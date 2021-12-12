from typing import Dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


CROP_SAVE_NUMBER = 0

######## COLORS FOR DIFFERENT POLYGONS ########

POLYGON_COLORS = {
    'ENTRY': (255, 255, 255),
    'EXIT': (0, 0, 125),
    'BBOX_AGE': (250, 150, 50),
    'ZONE': (35, 135, 235),
    'HEATMAP_PLOTS': (99, 71, 255),
    'NO-INTRUSION': (114, 238, 144),
    'INTRUSION': (184, 15, 10)

}

ESBAAR_COLORS = AttrDict({"orange": (30, 188, 243),
                          "red": (2, 109, 230),
                          "light_orange": (99, 175, 255),
                          "green": (96, 174, 39),
                          "light_green": (138, 239, 1),
                          "dark_red": (53, 67, 203)})

VIDEO_START_TIME = None
VIDEO_END_TIME = None

BLOCK_WEIGHT = 183

INTRUSION_CSV_DICT: Dict[str, list] = {
    'Intrusion Region': [],
    'Intrusion Date': [],
    'Start Time': [],
    'End Time': [],
    'Intrusion Duration': [],
    'Alarm Status': []
}

INTRUSION_GUI_FILE_PATH = './data_files/alarm.pickle'
INTRUSION_DATA_FILE_PATH = './data_files/intrusion_data/'
ACTIVITY_DATA_FILE_PATH = './data_files/activity.pickle'

ACTIVITY_DATA_DICT = {
    'activity': None,
    'color': None,
}

CODE_START_TIME = None

SOFTWARE_VERSION = 'v2.0.0-alpha'
# ! Save Locations
