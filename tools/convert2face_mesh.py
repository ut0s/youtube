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
from mediapipe.python.solutions import face_mesh_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec

parser = argparse.ArgumentParser(
    description='convert mp4 to face meshed and blur video')
parser.add_argument(
    'filename',
    type=str,
    help='path to video file')

args = parser.parse_args()

VIDEO_PATH = args.filename
AUTHOR_PATH = "/Users/ut0s/DEV/author.png"

# LPF
FILTER = 0.1


CAM_DIST = 4500

# Height and width that will be used by the model
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920

HUMAN_COLOR = (192, 192, 192)  # gray
WHITE = (255, 255, 255)  # white
GREEN_BACK_COLOR = (0, 249, 0)  # green #00f900
BLACK = (0, 0, 0)  #

# FaceMesh connections
THICKNESS_TESSELATION = 1
THICKNESS_CONTOURS = 3


def crop_square_image(center_x, center_y, image):
  center_x, center_y = int(nose.x * DESIRED_WIDTH), int(nose.y * DESIRED_HEIGHT)

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

  # image crop
  cropped = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
  return cropped


def draw_face_landmarks_on_image(rgb_image, detection_result):
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

    # FACEMESH_TESSELATION
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        # connection_drawing_spec=mp.solutions.drawing_styles
        # .get_default_face_mesh_tesselation_style()
        connection_drawing_spec=my_face_mesh_tesselation_style()
    )

    # FACEMESH_CONTOURS
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        landmark_drawing_spec=None,
        # connection_drawing_spec=mp.solutions.drawing_styles
        # .get_default_face_mesh_contours_style()
        connection_drawing_spec=my_face_mesh_contours_style()
    )

    # FACEMESH_IRISES
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #     landmark_drawing_spec=None,
    #     # connection_drawing_spec=mp.solutions.drawing_styles
    #     # .get_default_face_mesh_iris_connections_style()
    #     connection_drawing_spec=my_face_mesh_iris_connections_style()
    # )

  return annotated_image


def my_face_mesh_tesselation_style():
  return DrawingSpec(color=BLACK, thickness=THICKNESS_TESSELATION)


def my_face_mesh_contours_style():
  face_mesh_contours_connection_style = {}

  style = {
      face_mesh_connections.FACEMESH_LIPS:
      DrawingSpec(color=BLACK, thickness=THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_LEFT_EYE:
      DrawingSpec(color=BLACK, thickness=THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_LEFT_EYEBROW:
      DrawingSpec(color=BLACK, thickness=THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_RIGHT_EYE:
      DrawingSpec(color=BLACK, thickness=THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
      DrawingSpec(color=BLACK, thickness=THICKNESS_CONTOURS),
      face_mesh_connections.FACEMESH_FACE_OVAL:
      DrawingSpec(color=BLACK, thickness=THICKNESS_CONTOURS)
  }

  for k, v in style.items():
    for connection in k:
      face_mesh_contours_connection_style[connection] = v
  return face_mesh_contours_connection_style


def my_face_mesh_iris_connections_style():
  face_mesh_iris_connections_style = {}
  spec = DrawingSpec(color=(255, 255, 255), thickness=15)
  for connection in face_mesh_connections.FACEMESH_LEFT_IRIS:
    face_mesh_iris_connections_style[connection] = spec
  for connection in face_mesh_connections.FACEMESH_RIGHT_IRIS:
    face_mesh_iris_connections_style[connection] = spec
  return face_mesh_iris_connections_style


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/face_landmarker.task'),
    # running_mode=VisionRunningMode.IMAGE)
    running_mode=VisionRunningMode.VIDEO,
)

with FaceLandmarker.create_from_options(face_options) as landmarker:
  cap = cv2.VideoCapture(VIDEO_PATH)

  # print("FPS:{}".format(cap.get(cv2.CAP_PROP_FPS)))
  # fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # mp4
  fmt = cv2.VideoWriter_fourcc("H", "2", "6", "4")
  writer = cv2.VideoWriter('./result.mkv', fmt, cap.get(cv2.CAP_PROP_FPS), (DESIRED_HEIGHT, DESIRED_HEIGHT))

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

    # Perform face landmarking on the provided single image.
    # The face landmarker must be created with the image mode.
    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
    # print(face_landmarker_result)
    try:
      nose = face_landmarker_result.face_landmarks[0][0]
      center_x, center_y = int(nose.x * DESIRED_WIDTH), int(nose.y * DESIRED_HEIGHT)

      center_x = int(FILTER * center_x + (1-FILTER) * center_x_old)
      center_y = int(FILTER * center_y + (1-FILTER) * center_y_old)
      center_x_old = center_x
      center_y_old = center_y
    except IndexError:
      center_x = center_x_old
      center_y = center_y_old

    cropped = crop_square_image(center_x, center_y, image)

    # Blur the image background based on the segmentation mask.
    blurred_image = cv2.GaussianBlur(cropped, (85, 85), 0)

    try:
      # draw face mesh
      dummy = np.zeros(image.shape, dtype=np.uint8)
      dummy[:] = WHITE
      face_meshed = draw_face_landmarks_on_image(dummy, face_landmarker_result)
      face_meshed = crop_square_image(center_x, center_y, face_meshed)
      face_meshed_mask = face_meshed == BLACK
      merged_image = np.where(face_meshed_mask, face_meshed, blurred_image)

      writer.write(merged_image)
      # cv2.imshow('Output', merged_image)
      # cv2.waitKey(1)
    except IndexError:
      writer.write(blurred_image)

    # progress
    frame += 1
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[{}/{}] {}%".format(frame, total_frame, float(frame/total_frame)*100))
    sys.stdout.flush()

  writer.release()
