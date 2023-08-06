#!/usr/bin/env python3
import math
import argparse
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


parser = argparse.ArgumentParser(
    description='convert mp4 to face meshed and blur video')
parser.add_argument(
    'filename',
    type=str,
    help='path to video file')

args = parser.parse_args()

VIDEO_PATH = args.filename
MODEL_ASSET_PATH = '../model/pose_landmarker.task'

# Height and width that will be used by the model
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920

BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white

# Performs resizing and showing the image


def resize_image(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

  return img


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
    running_mode=VisionRunningMode.VIDEO)

with PoseLandmarker.create_from_options(options) as landmarker:
  cap = cv2.VideoCapture(VIDEO_PATH)

  if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

  cv2.startWindowThread()
  cv2.namedWindow("Output")

  # Loop through demo image(s)
  while cap.isOpened():
    ret, image = cap.read()
    if not ret:
      break

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # The hande landmarker must be created with the image mode.
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    # print(face_landmarker_result)

    # Generate solid color images for showing the output segmentation mask.
    image_data = mp_image.numpy_view()
    # fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    # fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = (20, 20, 150)

    annotated_image = draw_landmarks_on_image(bg_image, pose_landmarker_result)

    image = resize_image(annotated_image)

    cv2.imshow('Output', image)
    cv2.waitKey(1)

  cv2.destroyAllWindows()
