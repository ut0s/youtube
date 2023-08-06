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


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_iris_connections_style())

  return annotated_image


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../model/face_landmarker.task'),
    # running_mode=VisionRunningMode.IMAGE)
    running_mode=VisionRunningMode.VIDEO,
)

# The landmarker is initialized. Use it here.
with FaceLandmarker.create_from_options(options) as landmarker:
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

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Perform face landmarking on the provided single image.
    # The face landmarker must be created with the image mode.
    # face_landmarker_result = landmarker.detect(mp_image)
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    # print(face_landmarker_result)

    # Generate solid color images for showing the output segmentation mask.
    image_data = mp_image.numpy_view()
    # fg_image = np.zeros(image_data.shape, dtype=np.uint8)
    # fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_data.shape, dtype=np.uint8)
    bg_image[:] = (20, 20, 150)

    annotated_image = draw_landmarks_on_image(bg_image, face_landmarker_result)

    image = resize_image(annotated_image)

    cv2.imshow('Output', image)
    cv2.waitKey(1)

  cv2.destroyAllWindows()
