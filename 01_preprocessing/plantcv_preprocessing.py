"""Color/scale and crop from 261122Colab_ImageAdjustment.ipynb, cells 14 and 17."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from plantcv import plantcv as pcv


def correct_and_scale(source, target_mm_per_px=0.1, chip_width_mm=40.0):
    """Use the recorded chip-area formula and the historical resize interpolation."""
    image, _, _ = pcv.readimage(filename=str(source))
    pcv.outputs.clear()
    corrected = pcv.transform.auto_correct_color(rgb_img=image, radius=15)
    chip_area = float(pcv.outputs.metadata["median_color_chip_size"]["value"][0])
    target_area = (chip_width_mm / target_mm_per_px) ** 2
    scale = np.sqrt(target_area / chip_area)
    width = int(round(corrected.shape[1] * scale))
    height = int(round(corrected.shape[0] * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(corrected, (width, height), interpolation=interp), chip_area, scale


def normalize_batch(source, output, target_mm_per_px, chip_width_mm):
    written = 0
    pcv.params.debug = None
    for path in sorted(source.glob("*.jpg")):
        if not path.is_file():
            continue
        dest = output / f"{path.stem}_c{path.suffix}"
        try:
            corrected, area, scale = correct_and_scale(path, target_mm_per_px, chip_width_mm)
            pcv.print_image(img=corrected, filename=str(dest))
            print(f"{path.name}: chip area={area:.0f}, scale={scale:.3f}, size={corrected.shape[1]}x{corrected.shape[0]}")
            written += 1
        except Exception as error:
            print(f"Skipped {path.name}: {error}")
    return written


VALID_MODES = ("leaf_on_white", "white_sheet", "remove_card_only")


def get_colorcard_left_x(rgb_img, color_chip_size=None):
  """
  Detects the ColorChecker via PlantCV and returns its leftmost x-coordinate
  (including the black card border).
  """
  labeled_mask = pcv.transform.detect_color_card(
      rgb_img=rgb_img, color_chip_size=color_chip_size
  )
  ys, xs = np.nonzero(labeled_mask)

  if len(xs) > 0:
    margin = max(15, int(0.025 * rgb_img.shape[1]))
    return max(1, int(np.min(xs)) - margin)

  h, w, _ = rgb_img.shape
  gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
  _, dark_mask = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
  dark_mask[:, : int(w * 0.35)] = 0

  contours, _ = cv2.findContours(
      dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if contours:
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) > (h * w * 0.05):
      x, _, _, _ = cv2.boundingRect(largest)
      return max(1, x - 5)

  return None


def get_white_sheet_bbox(rgb_no_card):
  """
  Finds the bounding box (x0, y0, x1, y1) of the white paper sheet.
  """
  gray = cv2.cvtColor(rgb_no_card, cv2.COLOR_RGB2GRAY)
  h, w = gray.shape

  _, white_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

  kernel = np.ones((21, 21), np.uint8)
  white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
  white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

  contours, _ = cv2.findContours(
      white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if not contours:
    return (0, 0, w, h)

  largest = max(contours, key=cv2.contourArea)
  if cv2.contourArea(largest) < (h * w * 0.05):
    return (0, 0, w, h)

  x, y, bw, bh = cv2.boundingRect(largest)
  return (x, y, x + bw, y + bh)


def crop_around_leaf_on_white(rgb_no_card, pad_ratio=0.12, min_pad_px=25):
  """
  Crops around the leaf inside the white sheet so the background stays white
  and handwritten labels near the paper margin are excluded.
  """
  sx0, sy0, sx1, sy1 = get_white_sheet_bbox(rgb_no_card)
  sheet_crop = rgb_no_card[sy0:sy1, sx0:sx1]
  sh, sw, _ = sheet_crop.shape

  lab = cv2.cvtColor(sheet_crop, cv2.COLOR_RGB2LAB)
  _, a_ch, b_ch = cv2.split(lab)
  hsv = cv2.cvtColor(sheet_crop, cv2.COLOR_RGB2HSV)
  _, s_ch, _ = cv2.split(hsv)

  chroma = np.sqrt(
      (a_ch.astype(np.float32) - 128.0) ** 2
      + (b_ch.astype(np.float32) - 128.0) ** 2
  )
  leaf_mask = ((chroma > 14) | (s_ch > 45)).astype(np.uint8) * 255

  border_x = max(10, int(sw * 0.04))
  border_y = max(10, int(sh * 0.04))
  leaf_mask[:border_y, :] = 0
  leaf_mask[sh - border_y :, :] = 0
  leaf_mask[:, :border_x] = 0
  leaf_mask[:, sw - border_x :] = 0

  kernel = np.ones((11, 11), np.uint8)
  leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
  leaf_mask = cv2.morphologyEx(
      leaf_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)
  )

  contours, _ = cv2.findContours(
      leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  if not contours:
    return sheet_crop

  leaf_contour = max(contours, key=cv2.contourArea)
  if cv2.contourArea(leaf_contour) < (sh * sw * 0.01):
    return sheet_crop

  lx, ly, lw, lh = cv2.boundingRect(leaf_contour)
  pad_x = max(min_pad_px, int(lw * pad_ratio))
  pad_y = max(min_pad_px, int(lh * pad_ratio))

  x0 = max(0, lx - pad_x)
  y0 = max(0, ly - pad_y)
  x1 = min(sw, lx + lw + pad_x)
  y1 = min(sh, ly + lh + pad_y)

  return sheet_crop[y0:y1, x0:x1]


def process_images_for_leaf_analysis(
    input_dir, output_dir, crop_mode,
    pad_ratio=0.12,
    color_chip_size=None,
    extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff"),
):

  pcv.params.debug = None

  files = sorted([
      f
      for f in os.listdir(input_dir)
      if os.path.isfile(os.path.join(input_dir, f))
      and f.lower().endswith(extensions)
  ])

  if not files:
    print(f"No supported image files found in '{input_dir}'.")
    return

  print(f"\nSaving '{crop_mode}' images to: {output_dir}\n")

  written = 0
  for fname in files:
    in_path = os.path.join(input_dir, fname)
    rgb_img, _, _ = pcv.readimage(filename=in_path, mode="rgb")

    left_x = get_colorcard_left_x(rgb_img, color_chip_size=color_chip_size)
    if left_x is None:
      print(f"[SKIP] {fname}: ColorChecker not detected.")
      continue

    rgb_no_card = rgb_img[:, :left_x]

    if crop_mode == "leaf_on_white":
      result_img = crop_around_leaf_on_white(rgb_no_card, pad_ratio=pad_ratio)
    elif crop_mode == "white_sheet":
      sx0, sy0, sx1, sy1 = get_white_sheet_bbox(rgb_no_card)
      result_img = rgb_no_card[sy0:sy1, sx0:sx1]
    elif crop_mode == "remove_card_only":
      result_img = rgb_no_card

    out_path = os.path.join(output_dir, fname)
    pcv.print_image(img=result_img, filename=out_path)
    written += 1
    print(
        f"[OK] {fname}: {rgb_img.shape[1]}x{rgb_img.shape[0]} ->"
        f" {result_img.shape[1]}x{result_img.shape[0]} -> {out_path}"
    )
  return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("normalize", "crop"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-mm-per-px", type=float, default=0.1)
    p.add_argument("--chip-width-mm", type=float, default=40.0)
    p.add_argument("--crop-mode", choices=VALID_MODES, default="leaf_on_white")
    a = p.parse_args()
    if not a.input.is_dir():
        p.error("Input must be a directory")
    if not all(np.isfinite(x) and x > 0 for x in (a.target_mm_per_px, a.chip_width_mm)):
        p.error("Scale and chip width must be finite and positive")
    if a.output.resolve().is_relative_to(a.input.resolve()):
        p.error("Choose a new directory outside the input")
    a.output.mkdir(parents=True)
    if a.stage == "normalize":
        written = normalize_batch(a.input, a.output, a.target_mm_per_px, a.chip_width_mm)
    else:
        written = process_images_for_leaf_analysis(a.input, a.output, a.crop_mode)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
