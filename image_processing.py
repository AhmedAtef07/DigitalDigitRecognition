import numpy as np
import matplotlib.pyplot as plt
import random

WHITE = np.array((255, 255, 255), dtype=np.uint8)
BLACK = np.array(( 0,  0,  0), dtype=np.uint8)

# Trim passed image and convert it into binary image.
# grayscale_img is 2d array of values range from 0 to 255 inclusive.
# width is the width of the grayscale_img.
# returns boolean mask depends on the mean of the image pixels.
def threshold_grayscale_img(grayscale_img):
  mask = grayscale_img > grayscale_img.mean()
  return mask
  
def get_bin_img_from_mask(mask):
  bin_img = np.empty(mask.shape + (3,), dtype=np.uint8)
  bin_img[mask] = BLACK
  bin_img[~mask] = WHITE
  return bin_img

def trim_mask(mask):
  row, col = mask.shape

  mask_flatten_true = np.where(mask.flatten() == True)[0]
  mask_rot_flatten_true = np.where(np.rot90(mask, 3).flatten() == True)[0]

  top    = mask_flatten_true[0] / col
  bottom = mask_flatten_true[-1] / col
  left   = mask_rot_flatten_true[0] / row
  right  = mask_rot_flatten_true[-1] / row

  return mask[top:bottom, left:right]

def read_images(file_path, lines_count, randomize):
  lines = [line.rstrip('\n').rstrip('\r') for line in open(file_path)][1:]
  if randomize: random.shuffle(lines)
  lines_count = min(len(lines), lines_count)
  return get_img_raw(lines, lines_count)
  
def get_img_raw(lines, lines_count):
  imgs = []
  for line in lines[:lines_count]:
  # for line in lines[-2000:]:
  # import random
  # for line in random.sample(lines, 5):
    tokens = np.array(line.split(",")).astype(int)
    imgs.append([tokens[0], tokens[1:]])
  return imgs
    
def plot_img(mask):
  plt.imshow(get_bin_img_from_mask(mask))
  plt.show()  
  pass

def get_train_set(file_path, lines_count, randomize = True):
  raw_imgs = read_images(file_path, lines_count, randomize)
  imgs = []
  for row in raw_imgs:
    img_value = row[0]
    img1d = row[1]
    img2d = img1d.reshape(28, 28)
    img_thresholded = threshold_grayscale_img(img2d)
    img_trimmed = trim_mask(img_thresholded)

    imgs.append({"value": img_value, "data": img_trimmed})
    # plot_img(imgs[-1]["data"])

  return imgs
