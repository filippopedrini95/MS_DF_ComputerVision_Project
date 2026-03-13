#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np


def image_preprocess(img):
  if type(img) != np.ndarray:
    print("provide an image")
    return 0
  
  #resize and padding
  height, width, ch = img.shape

  w_longest =  width > height
  max_side = np.max([height, width])
  min_side = np.min([height, width])

  resized_max_side = 32
  resized_min_side = int(32 * min_side / max_side) if height != width else 32

  dsize = (resized_max_side, resized_min_side) if w_longest else (resized_min_side, resized_max_side)
  img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

  padd_tuple = (int(np.ceil((32 - resized_min_side)/2)), int((32 - resized_min_side)/2), 0, 0)

  if w_longest:
    top, bottom, left, right = padd_tuple
  else:
    left, right, top, bottom = padd_tuple

  final_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

  return final_img
