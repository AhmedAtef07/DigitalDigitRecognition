from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import operator

import image_processing as imgp

def sym_x(mask):
  row, col = mask.shape
  midpoint_y = row / 2
  ratios = []

  for i in range(midpoint_y):
    ratios += [(mask[i] ^ mask[-i - 1]).tolist().count(False) / float(col)]
    continue
    count_and = (mask[i] & mask[-i - 1]).tolist().count(True)
    count_xor = (mask[i] ^ mask[-i - 1]).tolist().count(True)
    if count_and + count_xor != 0:
      ratios += [count_and / float((count_and + count_xor))]
  return sum(ratios) / len(ratios)

def sym_y(mask):
  org_mask = mask
  mask = np.rot90(mask);
  row, col = mask.shape
  midpoint_y = row / 2
  ratios = []

  for i in range(midpoint_y):
    ratios += [(mask[i] ^ mask[-i - 1]).tolist().count(False) / float(col)]
    continue
    count_and = (mask[i] & mask[-i - 1]).tolist().count(True)
    count_xor = (mask[i] ^ mask[-i - 1]).tolist().count(True)
    if count_and + count_xor != 0:
      ratios += [count_and / float((count_and + count_xor))]
  if len(ratios) == 0:
    return 0.0
  return sum(ratios) / len(ratios)

def mean(mask):
  on_pixels = np.where(mask == True)
  on_pixels_x = sum(on_pixels[0]) / float(len(on_pixels[0])) / mask.shape[0]
  on_pixels_y = sum(on_pixels[1]) / float(len(on_pixels[1])) / mask.shape[1]
  return on_pixels_x, on_pixels_y

def on_pixels_in_h_half(mask):
  return mask[:mask.shape[0] / 2, :].flatten().tolist().count(True) / float(mask.flatten().tolist().count(True));

def on_pixels_in_v_half(mask):
  return mask[:, :mask.shape[1] / 2].flatten().tolist().count(True) / float(mask.flatten().tolist().count(True));

def color_change_count_h(mask, neglection_rate = .10):
  row, col = mask.shape
  row, col = float(row), float(col)

  color_change_count = []

  for r in mask:
    color_change_count.append([0, 0])
    current_segment = 0
    for i in range(len(r) - 1):
      if r[i] == r[i - 1]:
        current_segment += 1
      else:
        if current_segment / col >= col * neglection_rate:
          if r[i - 1]: # If the previous cell was black, so that's black_to_white
            color_change_count[-1][1] += 1
          else: # Change from white_to_black
            color_change_count[-1][0] += 1
        current_segment = 0
    # Check for the last unchecked segment
    if current_segment / col >= col * neglection_rate:
      if r[i - 1]: # If the previous cell was black, so that's black_to_white
        color_change_count[-1][1] += 1
      else: # Change from white_to_black
        color_change_count[-1][0] += 1

  print color_change_count
  return np.array(color_change_count).mean(axis=0).tolist()
  color_change_count_avg = np.array(color_change_count).mean(axis=0).tolist()
  return sum(color_change_count_avg)

def featurize(mask, value = -1):
  mean_y, mean_x = mean(mask)
  h_white_to_black, h_black_to_white = color_change_count_h(mask)
  v_white_to_black, v_black_to_white = color_change_count_h(np.rot90(mask))

  features = {
    "mean_y":              mean_y,
    "mean_x":              mean_x,
    "sym_x" :              sym_x(mask),
    "sym_y":               sym_y(mask),
    "on_pixels_in_h_half": on_pixels_in_h_half(mask),
    "on_pixels_in_v_half": on_pixels_in_v_half(mask),
    "h_white_to_black":    h_white_to_black,
    "h_black_to_white":    h_black_to_white,
    "v_white_to_black":    v_white_to_black,
    "v_black_to_white":    v_black_to_white,
  }

  if value == -1:
    return features
  return {"features": features, "value": value}
