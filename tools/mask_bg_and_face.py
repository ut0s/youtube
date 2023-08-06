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

CAM_DIST = 4500

# Height and width that will be used by the model
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920

BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white
GREEN_BACK_COLOR = (0, 249, 0)  # green #00f900
BLACK = (0, 0, 0)  # white

# FaceMesh connections
THICKNESS_TESSELATION = 1
THICKNESS_CONTOURS = 2


# Performs resizing and showing the image
def resize_image(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

  return img


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
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
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

base_options = python.BaseOptions(model_asset_path='model/multiclass.tflite')
seg_options = vision.ImageSegmenterOptions(base_options=base_options,
                                           output_category_mask=True)


with vision.ImageSegmenter.create_from_options(seg_options) as segmenter:
  with FaceLandmarker.create_from_options(face_options) as landmarker:
    cap = cv2.VideoCapture(VIDEO_PATH)

    if cap.isOpened() == False:
      print("Error opening video stream or file")
      raise TypeError

    cv2.startWindowThread()
    cv2.namedWindow("Output")

    # auther
    author_img = cv2.imread(AUTHOR_PATH)

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

      img_h, img_w, img_c = image.shape

      face_3d = []
      face_2d = []

      for idx, lm in enumerate(face_landmarker_result.face_landmarks[0]):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
          if idx == 1:
            node_2d = (lm.x * img_w, lm.y * img_h)
            node_3d = (lm.x * img_w, lm.y * img_h, lm.z * CAM_DIST)

          x, y = int(lm.x * img_w), int(lm.y * img_h)

          # Get the 2D Coordinates
          face_2d.append([x, y])

          # Get the 3D Coordinates
          face_3d.append([x, y, lm.z])

      # Convert to numpy array
      face_2d = np.array(face_2d, dtype=np.float64)
      face_3d = np.array(face_3d, dtype=np.float64)

      # camera matrix
      focal_length = 1 * img_w

      cam_matrix = np.array([[focal_length, 0, img_h/2],
                             [0, focal_length, img_w/2],
                             [0, 0, 1]])

      # the distortion matrix
      dist_matrix = np.zeros((4, 1), dtype=np.float64)

      # Solve PnP
      success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

      # Get rotational matrix
      rmat, jac = cv2.Rodrigues(rot_vec)

      # Get angles
      angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

      # Get the y rotation degree
      x = angles[0]*360
      y = angles[1]*360
      z = angles[2]*360

      # Display nose dir
      node_3d_projection, jacobian = cv2.projectPoints(node_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

      p1 = (int(node_2d[0]), int(node_2d[1]))
      p2 = (int(node_2d[0]+y*10), int(node_2d[1]-x*10))

      # Generate solid color images for showing the output segmentation mask.
      image_data = mp_image.numpy_view()
      bg_image = np.zeros(image_data.shape, dtype=np.uint8)
      bg_image[:] = GREEN_BACK_COLOR

      annotated_image = draw_face_landmarks_on_image(bg_image, face_landmarker_result)

      # node line
      cv2.line(annotated_image, p1, p2, BLACK, 5)

      image = resize_image(annotated_image)

      dst = np.zeros(author_img.shape, dtype=np.uint8)
      R, Jacob = cv2.Rodrigues(np.array([math.pi/4,0,0]))
      rmat2 = np.dot(rmat,R)
      print(rmat)
      print(rmat2)
      perspective_img = cv2.warpPerspective(author_img, rmat2, (author_img.shape[0], author_img.shape[1]), borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
      # image[10:10+perspective_img.shape[0], 10:10+perspective_img.shape[1]] = perspective_img

      cv2.imshow('Output', image)
      cv2.imshow('Author', perspective_img)
      cv2.waitKey(1)

    cv2.destroyAllWindows()
