#!/usr/bin/env python3
import math
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white


# Create the options that will be used for ImageSegmenter
# base_options = python.BaseOptions(model_asset_path='deeplabv3.tflite')
# base_options = python.BaseOptions(model_asset_path='multiclass.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

# Create the image segmenter
with vision.ImageSegmenter.create_from_options(options) as segmenter:

  cap = cv2.VideoCapture(0)

  if cap.isOpened()== False:
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

    # Retrieve the masks for the segmented image
    segmentation_result = segmenter.segment(mp_image)
    category_mask = segmentation_result.category_mask

    # Generate solid color images for showing the output segmentation mask.
    image_data = mp_image.numpy_view()
    fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR

    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
    output_image = np.where(condition, fg_image, bg_image)
    output_image = np.where(condition, image_data, bg_image)

    # Blur the image background based on the segmentation mask.
    # Apply effects
    # blurred_image = cv2.GaussianBlur(image_data, (55,55), 0)
    # condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    # output_image = np.where(condition, image_data, blurred_image)

    # print(f'Segmentation mask of {name}:')
    image = resize_image(output_image)

    cv2.imshow('Output',image)
    cv2.waitKey(1)

  cv2.destroyAllWindows()
