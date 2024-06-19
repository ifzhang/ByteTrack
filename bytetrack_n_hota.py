import sys
import random
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
from tqdm import tqdm
video_path = "input_videos\\MOT17-09-DPM-raw.webm"

# @title BYTETracker arguments & VideoInfo dataclass
@dataclass(frozen=True)
class ByteTrackArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.6
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = True

@dataclass(frozen=True)
class VideoInfo:
  fps:float
  height:int
  width:int

# @title Acquire video related data: FPS, height, width
#----------------GETTING INFORMATION ABOUT THE VIDEO---------------------#
cap =  cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#----------------DONE GETTING VIDEO INFORMATION--------------------------#

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

# @title necessary initializations
model = YOLO("yolov8x.pt")
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

import traceback
import copy
annotated_frames = []
stracks_list = []
temp_list = []

with open('results\\MOT17-09-DPM-190620241627.txt', 'w') as mot20_file:
    temp_tracker = BYTETracker(args,copy.deepcopy(extractor),vid_data.fps)
    frame_count = 0
    # Calculate total frames if possible (improves accuracy)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except:  # Some video sources might not provide frame count
        total_frames = None 
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
      while True:
          try:
              frame = next(iterator)
          except StopIteration:
              cap.release()
              print("Done. You have reached the end of the video")
              break

          try:
              results = model(frame)
              boxes = detections2boxes(results)
              if len(boxes)==0:
                continue
              stracks = temp_tracker.update(frame,boxes, [vid_data.height, vid_data.width], [vid_data.height, vid_data.width])
              for track in stracks:
                  print("LEN:",len(track.features))
                  x, y, w, h = track.tlwh  # Get bounding box coordinates
                  frame_id = track.frame_id
                  conf = track.score
                  # Write to MOT20 file (frame, ID, x, y, w, h, conf, -1, -1, -1)
                  mot20_file.write(f"{frame_id},{track.track_id},{x},{y},{w},{h},1,-1,-1,-1\n")
              annotation = annotate(frame, stracks, vid_data)
              annotated_frames.append(annotation)
              pbar.update(1) 
          except Exception as e:
              print(e)
              break

          frame_count += 1
create_video(annotated_frames, vid_data, "results\\output7.mp4")

def change_seventh_value(filename):
  """Loads data from a file, changes every 7th value to 1, and saves it back.

  Args:
    filename: The name of the text file to process.
  """

  # Load data from file
  with open(filename, "r") as f:
    data = [line.strip().split(",") for line in f]  # Split by comma

  # Change every 7th value (accounting for 0-based indexing)
  for row in data:
    for i in range(6, len(row), 7):  # Start at index 6, step by 7
      try:
        row[i] = 1.0 if float(row[i]) == -1 else float(row[i])  # Convert to float and change -1 to 1
      except ValueError:
        # Handle non-numeric values (if any)
        pass

  # Save modified data back to the original file (overwrites)
  with open(filename, "w") as f:
    for row in data:
      f.write(",".join(map(str, row)) + "\n")  # Join with commas for saving

# --- Main Execution ---
# filename = "detections.txt"
# change_seventh_value(filename)