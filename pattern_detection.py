from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import operator
from threading import Thread
import Queue
import cPickle

import features

train_set = []

q = Queue.Queue()

def analyze_raw_set(raw_train_set, cache_file_name = "NONE"):
  ts = []
  for img in raw_train_set:
    ts.append(Thread(None, train_one_set, None, (img, )))
    ts[-1].start()
  for t in ts:  
    t.join()
  global train_set
  train_set = [i for i in q.queue]
  cache_train_set(cache_file_name)
    
    
def train_one_set(img):
  q.put(features.featurize(img["data"], img["value"]))

def cache_train_set(cache_file_name):
  file_name = r"trainset_%d_%d.cache" % (len(train_set), int(time.time()))
  if cache_file_name != "NONE":
    file_name = "%s_%s" % (cache_file_name, len(train_set))
  with open(file_name, "wb") as output_file:
    cPickle.dump(train_set, output_file)

def load_from_cache(file):
  with open(file, "rb") as input_file:
    global train_set
    train_set = cPickle.load(input_file)

def eclidean_distance(f1, f2):
  f1_values = f1.values()
  f2_values = f2.values()
  dis = 0.0
  for i in range(len(f1_values)):
    dis += (f1_values[i] - f2_values[i]) ** 2
  return math.sqrt(dis)

def nearest_of_k_neighbors(distances, k):
  classVotes = defaultdict(lambda: 0)
  for d in distances[:k]:    
      classVotes[d['value']] += 1
  sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
  return sortedVotes[0][0]

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