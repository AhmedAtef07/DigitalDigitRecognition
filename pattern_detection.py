from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import operator

import features

train_set = []

def analyze_raw_set(raw_train_set):
  for img in raw_train_set:
    train_set.append(features.featurize(img["data"], img["value"]))

def eclidean_distance(f1, f2):
  f1_values = f1.values()
  f2_values = f2.values()
  dis = 0.0
  for i in range(len(f1_values)):
    dis += (f1_values[i] - f2_values[i]) ** 2
  return math.sqrt(dis)

def nearest_of_k_neighbors(distances, k):
  pass

def detect(mask):
  mask_set = features.featurize(mask);
  distances = []
  for s in train_set:
    distances.append({
      "distance": eclidean_distance(mask_set, s["features"]), 
      "value": s["value"]
      })
  distances.sort(key=operator.itemgetter("distance"))
  return distances
