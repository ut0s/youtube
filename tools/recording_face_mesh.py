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
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.hands import HandLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

HUMAN_COLOR = (192, 192, 192)  # gray
WHITE = (255, 255, 255)  # white
GREEN_BACK_COLOR = (0, 249, 0)  # green #00f900
BLACK = (0, 0, 0)  #

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)
_CYAN = (192, 255, 48)
_MAGENTA = (192, 48, 255)

# FaceMesh connections
THICKNESS_TESSELATION = 1
THICKNESS_CONTOURS = 3

def crop_square_image(center_x, center_y, image, desired_width, desired_height):
  center_x, center_y = int(nose.x * desired_width), int(nose.y * desired_height)

  # square_size = ret_square_image_size(image)
  square_size = desired_height
  if center_x - square_size/2 < 0:
    pt1 = [0, 0]
    pt2 = [square_size, square_size]
  elif center_x + square_size/2 > desired_width:
    pt1 = [desired_width - square_size, 0]
    pt2 = [desired_width, desired_height]
  else:
    pt1 = [int(center_x - square_size/2), 0]
    pt2 = [int(center_x + square_size/2), desired_height]

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

    # fill right eye
    right_eye = [  33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246                   ]
    right_eye_points = []
    for id in right_eye:
      right_eye_points.append([
        int(face_landmarks[id].x*rgb_image.shape[1]),
        int(face_landmarks[id].y*rgb_image.shape[0])
        ])
    # print(right_eye_points)
    cv2.fillConvexPoly(annotated_image, np.array(right_eye_points), color=BLACK)

    # fill left eye
    left_eye = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362  ]
    left_eye_points = []
    for id in left_eye:
      left_eye_points.append([
        int(face_landmarks[id].x*rgb_image.shape[1]),
        int(face_landmarks[id].y*rgb_image.shape[0])
        ])
    cv2.fillConvexPoly(annotated_image, np.array(left_eye_points), color=BLACK)

    # fill mouse
    mouse = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78
       ]
    mouse_points = []
    for id in mouse:
      mouse_points.append([
        int(face_landmarks[id].x*rgb_image.shape[1]),
        int(face_landmarks[id].y*rgb_image.shape[0])
        ])
    cv2.fillConvexPoly(annotated_image, np.array(mouse_points), color=BLACK)



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


# Hands
_RADIUS_FINGERTIP = 15
_RADIUS_PALM = 0
_RADIUS = 10
_THICKNESS_WRIST_MCP = 15
_THICKNESS_FINGER = 15
_THICKNESS_DOT = -1

MARGIN = 10  # pixels

def draw_hand_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    palm = [    HandLandmark.WRIST,
                      # HandLandmark.THUMB_CMC,
                      HandLandmark.INDEX_FINGER_MCP,
                      HandLandmark.MIDDLE_FINGER_MCP,
                      HandLandmark.RING_FINGER_MCP,
                      HandLandmark.PINKY_MCP]
    palm_points = []
    for p in palm:
      palm_points.append([
        int(hand_landmarks[p].x*rgb_image.shape[1]),
        int(hand_landmarks[p].y*rgb_image.shape[0])
        ])
    cv2.fillConvexPoly(annotated_image, np.array(palm_points), color=_GRAY)


    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        my_hand_landmarks_style(),
        my_hand_connections_style())


  return annotated_image

def my_hand_landmarks_style():
  hand_landmark_style = {}

  # Hand landmarks
  _PALM_LANDMARKS = (HandLandmark.WRIST,
                      HandLandmark.THUMB_CMC,
                      HandLandmark.INDEX_FINGER_MCP,
                      HandLandmark.MIDDLE_FINGER_MCP,
                      HandLandmark.RING_FINGER_MCP,
                      HandLandmark.PINKY_MCP)
  _THUMP_LANDMARKS = (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP,
                      HandLandmark.THUMB_TIP)
  _INDEX_FINGER_LANDMARKS = (HandLandmark.INDEX_FINGER_PIP,
                            HandLandmark.INDEX_FINGER_DIP,
                            HandLandmark.INDEX_FINGER_TIP)
  _MIDDLE_FINGER_LANDMARKS = (HandLandmark.MIDDLE_FINGER_PIP,
                              HandLandmark.MIDDLE_FINGER_DIP,
                              HandLandmark.MIDDLE_FINGER_TIP)
  _RING_FINGER_LANDMARKS = (HandLandmark.RING_FINGER_PIP,
                            HandLandmark.RING_FINGER_DIP,
                            HandLandmark.RING_FINGER_TIP)
  _PINKY_FINGER_LANDMARKS = (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP,
                            HandLandmark.PINKY_TIP)

  _HAND_LANDMARK_STYLE = {
      _PALM_LANDMARKS:
          DrawingSpec(
              color=_GRAY, thickness=_THICKNESS_DOT, circle_radius=_RADIUS_PALM),
      _THUMP_LANDMARKS:
          DrawingSpec(
              color=_PEACH, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
      _INDEX_FINGER_LANDMARKS:
          DrawingSpec(
              color=_PURPLE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
      _MIDDLE_FINGER_LANDMARKS:
          DrawingSpec(
              color=_YELLOW, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
      _RING_FINGER_LANDMARKS:
          DrawingSpec(
              color=_GREEN, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
      _PINKY_FINGER_LANDMARKS:
          DrawingSpec(
              color=_BLUE, thickness=_THICKNESS_DOT, circle_radius=_RADIUS),
  }

  for k, v in _HAND_LANDMARK_STYLE.items():
    for landmark in k:
      hand_landmark_style[landmark] = v
  return hand_landmark_style


def my_hand_connections_style() :
  hand_connection_style = {}
  # Hands connections
  _HAND_CONNECTION_STYLE = {
      hands_connections.HAND_PALM_CONNECTIONS:
          DrawingSpec(color=_GRAY, thickness=_THICKNESS_WRIST_MCP),
      hands_connections.HAND_THUMB_CONNECTIONS:
          DrawingSpec(color=_PEACH, thickness=_THICKNESS_FINGER),
      hands_connections.HAND_INDEX_FINGER_CONNECTIONS:
          DrawingSpec(color=_PURPLE, thickness=_THICKNESS_FINGER),
      hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS:
          DrawingSpec(color=_YELLOW, thickness=_THICKNESS_FINGER),
      hands_connections.HAND_RING_FINGER_CONNECTIONS:
          DrawingSpec(color=_GREEN, thickness=_THICKNESS_FINGER),
      hands_connections.HAND_PINKY_FINGER_CONNECTIONS:
          DrawingSpec(color=_BLUE, thickness=_THICKNESS_FINGER)
  }

  for k, v in _HAND_CONNECTION_STYLE.items():
    for connection in k:
      hand_connection_style[connection] = v
  return hand_connection_style


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    )

with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
  with FaceLandmarker.create_from_options(face_options) as face_landmarker:
    cap = cv2.VideoCapture(0)

    # Height and width that will be used by the model
    DESIRED_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    DESIRED_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    center_x_old = int(DESIRED_WIDTH/2)
    center_y_old = int(DESIRED_HEIGHT/2)

    count=0

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
      face_landmarker_result = face_landmarker.detect(mp_image)
      # print(face_landmarker_result)
      try:
        nose = face_landmarker_result.face_landmarks[0][0]
        center_x, center_y = int(nose.x * DESIRED_WIDTH), int(nose.y * DESIRED_HEIGHT)
        center_x_old = center_x
        center_y_old = center_y
      except IndexError:
        center_x = center_x_old
        center_y = center_y_old

      cropped = crop_square_image(center_x, center_y, image, DESIRED_WIDTH, DESIRED_HEIGHT)

      # Blur the image background based on the segmentation mask.
      # blurred_image = cv2.GaussianBlur(cropped, (85, 85), 0)
      # blurred_image = cv2.GaussianBlur(cropped, (15, 15), 0)
      bg_image = np.zeros(cropped.shape, dtype=np.uint8)
      bg_image[:] = WHITE

      # draw face mesh
      dummy = np.zeros(image.shape, dtype=np.uint8)
      dummy[:] = WHITE
      face_meshed = draw_face_landmarks_on_image(dummy, face_landmarker_result)
      face_meshed = crop_square_image(center_x, center_y, face_meshed, DESIRED_WIDTH, DESIRED_HEIGHT)
      face_meshed_mask = face_meshed == BLACK
      # merged_image = np.where(face_meshed_mask, face_meshed, blurred_image)
      merged_image = np.where(face_meshed_mask, face_meshed, bg_image)

      # hand pose
      hand_landmarker_result = hand_landmarker.detect(mp_image)
      hand_image = draw_hand_landmarks_on_image(bg_image, hand_landmarker_result)

      cv2.imshow('Output_hand', hand_image)
      cv2.imshow('Output_face', merged_image)
      key = cv2.waitKey(1)
      if key == ord('q'):
        break
      elif key == ord('s'):
        filename="./saved_%d" % count
        print(filename)
        cv2.imwrite(filename+"_hand.png", hand_image)
        cv2.imwrite(filename+"_face.png", merged_image)
        count += 1

cv2.destroyAllWindows()