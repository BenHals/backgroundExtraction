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


def create_bg(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI, SBSIZE):
    # Creates output directory
    o_dir = "output//{0}-{1}-{2}-{3}-{4}//".format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, MULTI)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)



    background_windows = cv2.imread('{0}background_windows.png'.format(o_dir),1)
    print(background_windows.shape)
    BLOCK_SIZE = int(background_windows.shape[0])
    num_windows = int(background_windows.shape[1] / BLOCK_SIZE)
    windows = []
    for w in range(0,num_windows):
        window = background_windows[:, int(w*BLOCK_SIZE):int((w+1)*BLOCK_SIZE)]
        windows.append(window)


    base = np.zeros((SBSIZE, SBSIZE, 3), np.uint8 )
    ri = 0

    overlap = math.floor(BLOCK_SIZE * 0.5)
    ol_thresh = 9000
    while ri < SBSIZE-1:
        print('row')
        start_row = max(0, ri - overlap) 
        end_row = min(start_row+BLOCK_SIZE, SBSIZE-1)
        take_rows = end_row - start_row
        ci = 0
        while ci < SBSIZE-1:
            start_col = max(0, ci - overlap)
            end_col = min(start_col+BLOCK_SIZE, SBSIZE-1)
            take_cols = end_col - start_col
            #print("{0}:{1}, {2}:{3}".format(start_row, end_row, start_col, end_col))
            
            min_diff = None
            min_wind = None
            mins_winds = []
            for wind in windows:
                base_ol_hor = base[0:int(take_rows), start_col:min(end_col, start_col + overlap)]
                wind_ol_hor = wind[0:int(take_rows), 0:min(take_cols, overlap)]
                ol_hor_dif = np.linalg.norm(base_ol_hor - wind_ol_hor)
                base_ol_ver = base[start_row:min(end_row, start_row + overlap), 0:int(take_cols)]
                wind_ol_ver = wind[0:min(take_rows, overlap), 0:int(take_cols)]
                ol_ver_dif = np.linalg.norm(base_ol_ver - wind_ol_ver)
                
                
                if min_diff == None or ol_hor_dif + ol_ver_dif < min_diff:
                    min_diff = ol_hor_dif + ol_ver_dif
                    min_wind = wind

                if ol_hor_dif + ol_ver_dif < ol_thresh:
                    mins_winds.append(wind)

            print(len(mins_winds))
            if(len(mins_winds) < 1):
                mins_winds = windows
            reroll = random.random()
            picked_wind_index = math.floor(random.random() * len(mins_winds))
            while reroll < picked_wind_index/len(min_wind):
                reroll = random.random()
                picked_wind_index = math.floor(random.random() * len(mins_winds))

            picked_wind = mins_winds[picked_wind_index]
            paste_window = picked_wind[0:int(take_rows), 0:int(take_cols)]
            base[start_row:end_row, start_col:end_col] = paste_window
            ci = end_col
        ri = end_row




    cv2.imshow("bg", base)
    cv2.imwrite("{0}bg-synth.png".format(o_dir), base)

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
    create_bg(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI, SBSIZE)