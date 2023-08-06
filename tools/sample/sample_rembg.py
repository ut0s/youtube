#!/usr/bin/env python3
import math
import cv2
import numpy as np

from rembg import remove, new_session

# MODEL_NAME = "u2net"
MODEL_NAME = "u2net_human_seg"
session = new_session(MODEL_NAME)

# Height and width that will be used by the model
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920

# Performs resizing and showing the image


def resize_image(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

  return img


BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white


# Create the image segmenter
cap = cv2.VideoCapture(0)

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

  # Retrieve the masks for the segmented image
  segmentation_result = remove(image, session=session)

  cv2.imshow('Output', segmentation_result)
  cv2.waitKey(1)

cv2.destroyAllWindows()
