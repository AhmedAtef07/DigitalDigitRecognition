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

def featurize(mask, value = -1):
  mean_y, mean_x = mean(mask)
  features = {
    "mean_y":              mean_y,
    "mean_x":              mean_x,
    "sym_x" :              sym_x(mask),
    "sym_y":               sym_y(mask),
    "on_pixels_in_h_half": on_pixels_in_h_half(mask),
    "on_pixels_in_v_half": on_pixels_in_v_half(mask)
  }
  if value == -1:
    return features
  return {"features": features, "value": value}
