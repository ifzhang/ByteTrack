# import git
# import subprocess

# def clone_repository(repo_url, target_dir):
#     """Clones a Git repository to the specified target directory."""

#     try:
#         git.Repo.clone_from(repo_url, target_dir)
#         print(f"Repository cloned successfully to '{target_dir}'")
#     except git.GitCommandError as e:
#         print(f"Error cloning repository: {e}")
# def install_libraries(libraries):
#     """Installs a list of Python libraries using pip."""
#     for library in libraries:
#         try:
#             subprocess.run(["pip", "install", library], check=True)
#             print(f"Installed: {library}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error installing {library}: {e}")
# def install_requirements(requirements_path):
#     """Installs requirements from a requirements.txt file."""
#     try:
#         subprocess.run(["pip", "install", "-r", requirements_path], check=True)
#         print("Requirements installed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error installing requirements: {e}")

# # @title import libraries
# from ultralytics import YOLO
# from typing import List, Optional, Tuple, Dict, Generator
# from dataclasses import dataclass
# from onemetric.cv.utils.iou import box_iou_batch
# import cv2
# import numpy as np
# from ultralytics.engine.results import Results
# import json
# from pprint import pprint
# import sys
# from ultralytics.utils.plotting import Annotator
# import os
# import numpy as np
# from scipy.spatial.distance import cosine
# import math
# import re
# import shutil
# from torchreid.utils import FeatureExtractor
# from yolox.tracker.byte_tracker import STrack, BYTETracker
# import copy
# from yolox.tracker.kalman_filter import KalmanFilter
# from yolox.tracker import matching
# from yolox.tracker.basetrack import BaseTrack, TrackState

# @title import libraries
from ultralytics import YOLO
from typing import List, Optional, Tuple, Dict, Generator
from dataclasses import dataclass
from onemetric.cv.utils.iou import box_iou_batch
import cv2
import numpy as np
from ultralytics.engine.results import Results
import json
from pprint import pprint
import sys
from ultralytics.utils.plotting import Annotator
import os
import numpy as np
from scipy.spatial.distance import cosine
import math
import re
import shutil
from torchreid.utils import FeatureExtractor
from yolox.tracker.byte_tracker import STrack, BYTETracker
import copy
from yolox.tracker.kalman_filter import KalmanFilter
from yolox.tracker import matching
from yolox.tracker.basetrack import BaseTrack, TrackState
import random
# @title methods

def detections2boxes(results:Results):
  data = []

  for box,conf,cls in zip(results[0].boxes.xyxy.tolist(),results[0].boxes.conf.tolist(),results[0].boxes.cls.tolist()):
    if int(cls)!=0:
      continue
    box.append(conf)
    data.append(box)
  return np.array(data)

def frame_generator(cap):
  while True:
    success, frame = cap.read()
    if not success:
      break
    yield frame

def draw_rectangle(frame: np.ndarray, track_id: int, bbox: Tuple):
    """
    This method draws a rectangle on the frame with a color seeded from the track_id and returns the annotated frame.

    Args:
        frame: The frame to draw on. (numpy array)
        track_id: The track ID of the object. (int)
        bbox: The bounding box of the object in (x1, y1, x2, y2) format. (tuple)

    Returns:
        The frame with the rectangle drawn on it. (numpy array)
    """
    bbox = [int(val) for val in bbox]
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Seed the random number generator with the track ID
    random.seed(track_id)

    # Generate random color components in the range 0-255
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Draw the rectangle on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    return frame

def create_video(frames, video_info, output_path="/content/outputs/videoname0.mp4"):
    """
    Creates and stores a video from a list of NumPy frames.

    Args:
        frames (list): A list of NumPy arrays representing video frames.
        video_info (VideoInfo): An object containing FPS, height, and width information.
        output_path (str): The desired output path for the video file.
    """

    # Define the video codec (adjust if needed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create the video writer object
    print(video_info)
    out = cv2.VideoWriter(output_path, fourcc, int(video_info.fps), (int(video_info.width), int(video_info.height)))

    # Write each frame to the video
    for frame in frames:
        out.write(frame)
    # Release the video writer object
    out.release()

def annotate(frame:np.array, stracks:List[STrack], videoinfo:VideoInfo):
  for index, track in enumerate(stracks):
    frame = draw_rectangle(frame,track.track_id,track.tlbr)
  return frame

def create_detections_file(stracks_list, output_filename="detections.txt"):
    """Creates a detections.txt file in MOTChallenge format from a list of STrack objects.

    Args:
        stracks_list: A list of STrack objects (from ByteTrack).
        output_filename: The name of the output file to create.
    """

    with open(output_filename, "w") as f:
        for strack in stracks_list:
            # Assuming strack has attributes: frame_id, track_id, and tlwh (top-left x, y, width, height)
            frame_id = strack.frame_id
            track_id = strack.track_id
            tlwh = strack.tlwh
            score = strack.score  # Add confidence score here

            # Convert tlwh to MOTChallenge format (bb_left, bb_top, width, height)
            bb_left = int(tlwh[0])
            bb_top = int(tlwh[1])
            width = int(tlwh[2])
            height = int(tlwh[3])

            # Write the detection line in MOTChallenge format
            line = f"{frame_id},{track_id},{bb_left},{bb_top},{width},{height},{score},-1,-1,-1\n"
            f.write(line)
video_path = "/content/MOT20-01-raw.mp4"
# @title BYTETracker arguments & VideoInfo dataclass
@dataclass(frozen=True)
class ByteTrackArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.7
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

@dataclass(frozen=True)
class VideoInfo:
  fps:float
  height:int
  width:int

cap =  cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

model = YOLO('required_models\\yolov8l-oiv7.pt')
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='required_models\\osnet__x0_1_market1501.pth',
    device='cpu',
    verbose = True
)
args = ByteTrackArgs()
vid_data = VideoInfo(fps,height,width)
# tracker = BYTETrackerv2(args=args,frame_rate=vid_data.fps)
iterator = iter(frame_generator(cap=cap))