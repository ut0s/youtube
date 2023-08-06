#!/usr/bin/env python3
import math
import argparse
import cv2
import numpy as np

from ultralytics import YOLO
import torch

# MODEL_ASSET_PATH = '../model/yolov8n-seg.pt'
MODEL_ASSET_PATH = '../model/yolov8s-seg.pt'
# MODEL_ASSET_PATH = '../model/yolov8x-seg.pt'

parser = argparse.ArgumentParser(
    description='convert mp4 to face meshed and blur video')
parser.add_argument(
    'filename',
    type=str,
    help='path to video file')

args = parser.parse_args()

VIDEO_PATH = args.filename

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
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


# Load the YOLOv8 model
model = YOLO(MODEL_ASSET_PATH)

cap = cv2.VideoCapture(0)
if cap.isOpened() == False:
  print("Error opening video stream or file")
  raise TypeError

cv2.startWindowThread()
cv2.namedWindow("Output")

while cap.isOpened():
  ret, image = cap.read()
  if not ret:
    break

  results = model.predict(image, classes=[0], conf=0.65)

  # res_plotted = results[0].plot()
  fg_image = np.zeros(image.shape, dtype=np.uint8)
  fg_image[:] = MASK_COLOR
  masked = results[0].plot(img=fg_image, labels=False, boxes=False, mask=True)

  # cv2.imshow('Output', image)
  cv2.imshow('Output(masked)', masked)
  # cv2.imshow('Output(res)', fg_image)
  cv2.waitKey(1)

cv2.destroyAllWindows()
