#!/usr/bin/env python3
import sys
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
GREEN_BACK_COLOR = (0, 249, 0)  # green #00f900
BLACK = (0, 0, 0)  # white


def ret_square_image_size(image):
  h, w = image.shape[:2]
  if h < w:
    return h
  else:
    return w


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_ASSET_PATH),
    running_mode=VisionRunningMode.VIDEO)

with PoseLandmarker.create_from_options(options) as landmarker:
  cv2.startWindowThread()
  cv2.namedWindow("Output")

  cap = cv2.VideoCapture(VIDEO_PATH)

  # init
  total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame = 0
  center_x_old = int(DESIRED_WIDTH/2)
  center_y_old = int(DESIRED_HEIGHT/2)

  if cap.isOpened() == False:
    print("Error opening video stream or file")
    raise TypeError

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
    nose = pose_landmarker_result.pose_landmarks[0][0]

    center_x, center_y = int(nose.x * DESIRED_WIDTH), int(nose.y * DESIRED_HEIGHT)

    # filter
    FILTER = 0.1
    center_x = int(FILTER * center_x + (1-FILTER) * center_x_old)
    center_y = int(FILTER * center_y + (1-FILTER) * center_y_old)
    center_x_old = center_x
    center_y_old = center_y

    # square_size = ret_square_image_size(image)
    square_size = DESIRED_HEIGHT
    if center_x - square_size/2 < 0:
      pt1 = [0, 0]
      pt2 = [square_size, square_size]
    elif center_x + square_size/2 > DESIRED_WIDTH:
      pt1 = [DESIRED_WIDTH - square_size, 0]
      pt2 = [DESIRED_WIDTH, DESIRED_HEIGHT]
    else:
      pt1 = [int(center_x - square_size/2), 0]
      pt2 = [int(center_x + square_size/2), DESIRED_HEIGHT]

    # cv2.rectangle(image,
    #               pt1=pt1,
    #               pt2=pt2,
    #               color=(0, 255, 0),
    #               thickness=3,
    #               lineType=cv2.LINE_4,
    #               shift=0)

    # image crop
    cropped = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    # progress
    frame += 1
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[{}/{}] {}%".format(frame, total_frame, float(frame/total_frame)*100))
    sys.stdout.flush()

    # cv2.imshow('Output(org)', image)
    cv2.imshow('Output(cropped)', cropped)
    cv2.waitKey(1)

  cv2.destroyAllWindows()
