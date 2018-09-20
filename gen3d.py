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
import random
from os import listdir
from os.path import isfile, join

def make_cube(l):
    output_str = """v 0 0 0
v 0 1 0
v 1 0 0
v 1 1 0
v 0 0 1
v 0 1 1
v 1 0 1
v 1 1 1
vt 0 0
vt 0 1
vt 1 0
vt 1 1
f 1/1 2/2 3/3
f 2/2 4/4 3/3
f 2/1 6/2 8/4
f 2/1 8/4 4/3
f 4/2 8/4 3/1
f 8/4 7/3 3/1
f 6/2 5/1 8/4
f 5/1 7/3 8/4
f 5/1 1/2 7/3
f 1/2 3/1 7/3
f 6/1 2/2 1/3
f 6/1 1/2 5/4"""
    return output_str

def gen_3d(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI, SBSIZE):
    # Creates output directory
    o_dir = "output//{0}-{1}-{2}-{3}-{4}//".format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, MULTI)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    onlyfiles = [f for f in listdir(o_dir) if isfile(join(o_dir, f)) and 'conponent' in f]
    for f in onlyfiles:
        print(f)
        component_id = int(f.split('-')[1])
        print(component_id)
        obj_str = make_cube(1)
        with open('{0}component-{1}.obj'.format(o_dir, component_id), 'w+') as wf:
            wf.write(obj_str)



if __name__ == "__main__":
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
    gen_3d(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI, SBSIZE)