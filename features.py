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

def color_change_count_h2(mask, neglection_rate = .05):
  row, col = mask.shape
  row, col = float(row), float(col)

  # black_to_white = 0
  # white_to_black = 0
  color_change_count = 0

  for r in mask:
    current_segment = 0
    for i in range(len(r) - 1):
      if r[i] == r[i + 1]:
        current_segment += 1
      else:
        if current_segment / col >= col * neglection_rate:
          color_change_count += 1
        current_segment = 0
    # Check for the last unchecked segment
    if current_segment / col >= col * neglection_rate:
      color_change_count += 1
    current_segment = 0

  # print black_to_white, white_to_black
  # if black_to_white > 5 or white_to_black > 5:
  #   imgp.plot_img(mask)  
  # print color_change_count, row, color_change_count / row
  return color_change_count / row * neglection_rate 

def color_change_count_h(mask):
  row, col = mask.shape
  row, col = float(row), float(col)

  color_change_count = 0

  for r in mask:
    for i in range(len(r) - 1):
      if r[i] != r[i + 1]:
        color_change_count += 1

  return color_change_count / row / col

def standard_diviation(mask):
  return np.array([i.std() for i in mask]).mean() 


def featurize(mask, value = -1):
  mean_y, mean_x = mean(mask)
  # h_white_to_black, h_black_to_white = color_change_count_h(mask)
  # v_white_to_black, v_black_to_white = color_change_count_h(np.rot90(mask))

  features = {
    "mean_y":               mean_y,
    "mean_x":               mean_x,
    "sym_x" :               sym_x(mask),
    "sym_y":                sym_y(mask),
    "on_pixels_in_h_half":  on_pixels_in_h_half(mask),
    "on_pixels_in_v_half":  on_pixels_in_v_half(mask),
    "color_change_count_h": color_change_count_h(mask),
    "color_change_count_v": color_change_count_h(np.rot90(mask)),
    "standard_diviation_h": standard_diviation(mask),
    "standard_diviation_v": standard_diviation(np.rot90(mask)),
  }

  if value == -1:
    return features
  return {"features": features, "value": value}
