import numpy as np
from collections import OrderedDict


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    def __init__(self, count_gen=None):
        self._count = 0
        self.count_gen = count_gen

        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New

        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0

        # multi-camera
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    def next_id(self):
        if self.count_gen:
            self._count = self.count_gen.__next__()
        else:
            self._count += 1
        return self._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed
