from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import operator
import sys

import image_processing as imgp
import pattern_detection as pd

lines_count = int(sys.argv[1])
train_percent = .90

second_arg = sys.argv[2]
if "." in second_arg:
  train_percent = float(second_arg)
else:
  train_percent = int(second_arg) / float(lines_count)




t = time.time()
raw_train_set = imgp.get_train_set("train.csv", lines_count)
print len(raw_train_set), "samples are in the raw_train_set."
print "Parsing images completed in %d seconds." % (time.time() - t)
# 18 sec

train_raw_set, test_raw_set = np.split(raw_train_set, [len(raw_train_set) * train_percent])
print "Number of samples in train_raw_set = %d" % len(train_raw_set)
print "Number of samples in test_raw_set = %d" % len(test_raw_set)

t = time.time()
pd.analyze_raw_set(train_raw_set)
print "Calculating train set features completed in %d seconds." % (time.time() - t)


print "=" * 20
t = time.time()

miss_count = 0

for test_img in test_raw_set:
  # pd.detect(test_img["data"])[:10]
  nearest_neighbours = [i["value"] for i in pd.detect(test_img["data"])[:10]]
  

  if test_img["value"] != nearest_neighbours[0]:
    miss_count += 1
    # print test_img["value"], nearest_neighbours
  # if test_img["value"] != nearest_neighbours[0]:
  #   imgp.plot_img(test_img["data"])

print "=" * 20

print "Train Count: %d" % len(train_raw_set)
print "Test  Count: %d" % len(test_raw_set)
print "Miss  Count: %d" % miss_count
print "Accuracy on K=1: %.2f%%" % (100 - miss_count / float(len(test_raw_set)) * 100)
print "Detecting completed in %d seconds." % (time.time() - t)
