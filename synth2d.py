import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage.color import rgb2gray
import argparse
import json
import os
from os import listdir
from os.path import isfile, join
import random

# Extract and set command line arguments for parameters
parser = argparse.ArgumentParser()
parser.add_argument("--blocksize", type=float, help="enter some quality limit",
                    nargs='?', default=0.035)
parser.add_argument("--multi", type=float, help="enter some quality limit",
                    nargs='?', default=1.95)
parser.add_argument("--texthresh", type=float, help="enter some quality limit",
                    nargs='?', default=0.1)
parser.add_argument("--colthresh", type=float, help="enter some quality limit",
                    nargs='?', default=0.4)
parser.add_argument("--file", help="enter some quality limit",
                    nargs='?', default='miro')
parser.add_argument("--detail", help="enter some quality limit", action='store_true', default=False)
parser.add_argument("--sbsize", type=float, default=512)
args = parser.parse_args()
BLOCK_SIZE = args.blocksize 
COLTHRESH = args.colthresh
TEXTHRESH = args.texthresh
FILE = args.file
DETAIL = args.detail
MULTI = args.multi
SBSIZE = args.sbsize

# Creates output directory
o_dir = "output//{0}-{1}-{2}-{3}-{4}//".format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, MULTI)
if not os.path.exists(o_dir):
    os.makedirs(o_dir)



background_windows = cv2.imread('{0}bg-synth.png'.format(o_dir),1)
print(background_windows.shape)
h,w,c = background_windows.shape


onlyfiles = [f for f in listdir(o_dir) if isfile(join(o_dir, f)) and 'conponent' in f]
for f in onlyfiles:
    print(f)
    min_sf = 0.1
    max_sf = 0.8
    comp = cv2.imread("{0}{1}".format(o_dir, f),1)

    scaling_factor = np.random.normal(1, 0)
    scaling_factor = min(scaling_factor, max_sf)
    scaling_factor = max(scaling_factor, min_sf)
    print(scaling_factor)
    rw = comp.shape[1]
    rh = comp.shape[0]
    dash_split = f.split('-')
    if len(dash_split) > 2:
        sf_orig = float(dash_split[4][0:-4])
        rw = rw /sf_orig
        rh = rh /sf_orig
    scaled = cv2.resize(comp, (int(rw * scaling_factor), int(rh*scaling_factor)))

    l = math.floor(random.random() * (w + scaled.shape[1]/2) - (scaled.shape[1]/2))
    t = math.floor(random.random() * (h + scaled.shape[0]/2) - (scaled.shape[0]/2))
    r = l+scaled.shape[1]
    b = t+scaled.shape[0]
    visible_l = max(0, l)
    visible_t  = max(0, t)
    visible_r = min(w, r)
    visible_b = min(h, b)
    for ri in range(0, visible_b - visible_t):
        for ci in range(0, visible_r - visible_l):
            pixel = scaled[ri, ci]
            #print(pixel)
            if not np.all(pixel > 100):
                background_windows[visible_t+ri, visible_l+ci] = pixel
cv2.imshow("bg", background_windows)
cv2.imwrite("output//{0}-{1}-{2}-{3}-{4}-synth.png".format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, MULTI), background_windows)

cv2.waitKey(0)
cv2.destroyAllWindows()


