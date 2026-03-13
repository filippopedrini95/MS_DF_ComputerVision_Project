import cv2
import numpy as np

def image_preprocess(img):
  
  # Extract image dimensions (height, width, channels)
  # This information is used to determine how to resize while preserving aspect ratio
  height, width, ch = img.shape

  # Determine whether the longest side of the image is the width
  w_longest =  width > height
  
  # Identify the longest and shortest sides of the image
  max_side = np.max([height, width])
  min_side = np.min([height, width])

  # Compute the new resized dimensions while preserving aspect ratio
  # The longest side is scaled to 32 pixels (CIFAR-10 input size)
  resized_max_side = 32
  resized_min_side = int(32 * min_side / max_side) if height != width else 32

  # Define the target resize dimensions depending on which side is longer
  dsize = (resized_max_side, resized_min_side) if w_longest else (resized_min_side, resized_max_side)
  img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

  # Resize the image
  padd_tuple = (int(np.ceil((32 - resized_min_side)/2)), int((32 - resized_min_side)/2), 0, 0)

  # If the width is the longest side, padding is applied to top and bottom
  if w_longest:
    top, bottom, left, right = padd_tuple
  else:
    # Otherwise padding is applied to left and right
    left, right, top, bottom = padd_tuple

  # Apply constant padding (black pixels) to obtain the final 32x32 image
  final_img = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

  return final_img