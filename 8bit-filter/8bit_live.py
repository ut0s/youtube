from ultralytics import YOLO
import cv2
import numpy as np
from skimage import io
from pyxelate import Pyx, Pal


def camera_for_loop():
  # Webカメラのキャプチャを開始
  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))

  if not cap.isOpened():
    print("Error: Could not open camera.")
    return

  # setup
  # Load a model
  model = YOLO("yolo11n-seg.pt")  # load an official model
  _, frame = cap.read()
  seg_image = apply_person_segmentation(model, frame)
  pyx = setup_pyxelate_filter(seg_image, scale=8, palette_num=24)


  screenshot_count = 0

  while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    if not ret:
      print("Error: Could not read frame.")
      break

    # fit() and transform() on image with alpha channel
    seg_image = apply_person_segmentation(model, frame)
    # scale image to 360x360
    seg_image = cv2.resize(seg_image, (360, 360))
    new_image = apply_pyxelate_filter(seg_image, pyx)

    # upscaled image
    # new_image = cv2.resize(new_image, (1080, 1080))

    # 結果を表示
    # cv2.imshow('Result', seg_image)
    cv2.imshow('Result', new_image)

    # 's'キーでスクリーンショットを保存
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
      screenshot_filename = f"screenshot_{screenshot_count}.png"
      cv2.imwrite(screenshot_filename, new_image)
      print(f"Screenshot saved as {screenshot_filename}")
      screenshot_count += 1

    # 'q'キーで終了
    # if key == ord('q'):
    #   break

  # リソースを解放
  cap.release()
  cv2.destroyAllWindows()


def setup_pyxelate_filter(frame, scale, palette_num) -> Pyx:
  # 1) Instantiate Pyx transformer
  pyx = Pyx(factor=scale, palette=palette_num, upscale=scale, dither="naive")
  # 2) fit an image, allow Pyxelate to learn the color palette
  pyx.fit(frame)
  return pyx


def apply_pyxelate_filter(frame, pyx) -> np.ndarray:
  # 3) transform image to pixel art using the learned color palette
  new_image = pyx.transform(frame)
  return new_image


def crop_to_square(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
  # マスクの重心を計算
  M = cv2.moments(mask)
  if M["m00"] == 0:
    return image  # マスクがない場合は元の画像を返す

  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])

  # 正方形のサイズを決定
  h, w = image.shape[:2]
  size = min(h, w)

  # 正方形の範囲を計算
  y1 = 0
  y2 = size
  if cX - size // 2 < 0:
    x1 = max(cX - size // 2, 0)
    x2 = size
  elif cX + size // 2 > w:
    x1 = w - size
    x2 = w
  else:
    x1 = cX - size // 2
    x2 = cX + size // 2

  # 画像を切り出し
  cropped_image = image[y1:y2, x1:x2]
  return cropped_image


def apply_person_segmentation(model, frame_img) -> np.ndarray:
  # Load a model
  # model = YOLO("yolo11n-seg.pt")  # load an official model

  # Predict with the model
  # predict on an image
  results = model(frame_img, verbose=False)

  # Generate a binary mask and draw contours
  img = np.copy(results[0].orig_img)
  b_mask = np.zeros(img.shape[:2], np.uint8)
  contour = results[0].masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
  cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

  # Isolate the object using the binary mask:
  mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
  isolated = cv2.bitwise_and(img, mask3ch)

  # 背景を緑に設定
  green_background = np.full_like(img, (0, 255, 0))
  green_background = cv2.bitwise_and(
      green_background, cv2.bitwise_not(mask3ch))
  isolated = cv2.add(isolated, green_background)

  cropped_image = crop_to_square(isolated, b_mask)
  return cropped_image


if __name__ == "__main__":
  camera_for_loop()
