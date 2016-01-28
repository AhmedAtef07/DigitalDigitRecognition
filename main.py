from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import math
import operator
import sys

# Local import
import image_processing as imgp
import pattern_detection as pd

train_raw_set, test_raw_set = [], []

################################## TESTING FROM CACHED FILE ##################################
if sys.argv[1] == "lc":
  pd.load_from_cache(sys.argv[2])
  test_raw_set = imgp.get_train_set("test/1")

  for test_img in test_raw_set:
    x = pd.detect(test_img["data"])

    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 200]:
      # print pd.nearest_of_k_neighbors(x, k)
      print "K = %2d: Expected %s" % (k, pd.nearest_of_k_neighbors(x, k))
    imgp.plot_img(test_img["data"])
    print "=" * 20

   
   

elif sys.argv[1] == "lcsv":
  pd.load_from_cache(sys.argv[2])
  test_raw_set = imgp.get_csv_train_set("../mnist_test.csv", 10000)

  print "=" * 20
  t = time.time()

  k_misses = defaultdict(lambda: 0)

  count = 0 
  ne = [1, 3, 20, 30, 50, 70, 100, 200, 500, 1000, 1500, 4000, 6000, 7000, 10000]
  for test_img in test_raw_set:
    x = pd.detect(test_img["data"])
    # nearest_neighbours = [i["value"] for i in pd.detect(test_img["data"])[:10]]
    

    for k in ne:
      if test_img["value"] != pd.nearest_of_k_neighbors(x, k):
        k_misses[k] += 1
    # if test_img["value"] != nearest_neighbours[0]:
    #   imgp.plot_img(test_img["data"])

    count += 1
    # print "#%d, %d -> %d; Accuracy: %.2f%%, Correct: %d, Wrong: %d" % (count, test_img["value"], x[0]["value"], 100 - k_misses[1] / float(count) * 100, count - k_misses[1], k_misses[1])
    for k in ne:
      print "#%d, %d -> %d; K = %4d, Correct: %d, Wrong: %d, Accuracy: %.2f%%" % (
        count, 
        test_img["value"],
        pd.nearest_of_k_neighbors(x, k), 
        k,
        count - k_misses[k], 
        k_misses[k],
        100 - k_misses[k] / float(count) * 100)
    print "-" * 17 
    
  print "=" * 20

  print "Test  Count: %d" % len(test_raw_set)
  for k in range(len(k_misses)):
    print "K = %2d: Accuracy: %.2f%%, Miss Count: %d" % (k_misses.keys()[k], 100 - k_misses.values()[k] / float(len(test_raw_set)) * 100, k_misses.values()[k])
  print "Detecting completed in %d seconds." % (time.time() - t)




################################## FEATURZING THE DATA SET ##################################
else:
  lines_count = int(sys.argv[1])
  train_percent = .90

  second_arg = sys.argv[2]
  if "." in second_arg:
    train_percent = float(second_arg)
  else:
    train_percent = int(second_arg) / float(lines_count)

  cache_file_name = "NONE"  
  if len(sys.argv) > 3: cache_file_name = sys.argv[3]

  t = time.time()
  # raw_train_set = imgp.get_csv_train_set("../mnist_train.csv", lines_count)
  raw_train_set = imgp.get_train_set("../data_digital_digits", lines_count)
  print len(raw_train_set), "samples are in the raw_train_set."
  print "Parsing images completed in %d seconds." % (time.time() - t)
  # 18 sec

  train_raw_set, test_raw_set = np.split(raw_train_set, [len(raw_train_set) * train_percent])
  print "Number of samples in train_raw_set = %d" % len(train_raw_set)
  print "Number of samples in test_raw_set = %d" % len(test_raw_set)

  t = time.time()
  pd.analyze_raw_set(train_raw_set, cache_file_name)
  print "Calculating train set features completed in %d seconds." % (time.time() - t)

  print "=" * 20
  t = time.time()

  k_misses = defaultdict(lambda: 0)

  for test_img in test_raw_set:
    x = pd.detect(test_img["data"])
    # nearest_neighbours = [i["value"] for i in pd.detect(test_img["data"])[:10]]
    

    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
      if test_img["value"] != pd.nearest_of_k_neighbors(x, k):
        k_misses[k] += 1
    # if test_img["value"] != nearest_neighbours[0]:
    #   imgp.plot_img(test_img["data"])

  print "=" * 20

  print "Train Count: %d" % len(train_raw_set)
  print "Test  Count: %d" % len(test_raw_set)
  for k in range(len(k_misses)):
    print "K = %2d: Accuracy: %.2f%%, Miss Count: %d" % (k_misses.keys()[k], 100 - k_misses.values()[k] / float(len(test_raw_set)) * 100, k_misses.values()[k])
  print "Detecting completed in %d seconds." % (time.time() - t)

