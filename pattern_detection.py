from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import operator
from threading import Thread
import Queue

import features

train_set = []

q = Queue.Queue()

def analyze_raw_set(raw_train_set):
  ts = []
  for img in raw_train_set:
    ts.append(Thread(None, train_one_set, None, (img, )))
    ts[-1].start()
  for t in ts:  
    t.join()
  global train_set
  train_set = [i for i in q.queue]
    
    
def train_one_set(img):
  q.put(features.featurize(img["data"], img["value"]))

def eclidean_distance(f1, f2):
  f1_values = f1.values()
  f2_values = f2.values()
  dis = 0.0
  for i in range(len(f1_values)):
    dis += (f1_values[i] - f2_values[i]) ** 2
  return math.sqrt(dis)

def nearest_of_k_neighbors(distances, k):
  pass

q = Queue.Queue()

def detect2(mask):
  mask_set = features.featurize(mask);
  ts = []
  for s in train_set:
    ts.append(Thread(None, detect_one_mask, None, (mask_set, s, )))
    ts[-1].start()
  for t in ts:
    t.join()

  distances = [i for i in q.queue]
  print type(distances[0])
  distances.sort(key=operator.itemgetter("distance"))
  return distances

  q = Queue.Queue()

def detect(mask):
  mask_set = features.featurize(mask);
  distances = []
  for s in train_set:
    distances.append({
    "distance": eclidean_distance(mask_set, s["features"]), 
    "value":    s["value"]
  })

  distances.sort(key=operator.itemgetter("distance"))
  return distances


def detect_one_mask(mask_set, s):
  q.put({
    "distance": eclidean_distance(mask_set, s["features"]), 
    "value":    s["value"]
  })