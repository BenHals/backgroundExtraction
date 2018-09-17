import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage.color import rgb2gray
import argparse
# import mahotas as mt
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--blocksize", type=int, help="enter some quality limit",
                    nargs='?', default=10)
parser.add_argument("--multi", type=float, help="enter some quality limit",
                    nargs='?', default=2)
parser.add_argument("--texthresh", type=int, help="enter some quality limit",
                    nargs='?', default=100)
parser.add_argument("--colthresh", type=float, help="enter some quality limit",
                    nargs='?', default=0.7)
parser.add_argument("--file", help="enter some quality limit",
                    nargs='?', default='miro')
parser.add_argument("--detail", help="enter some quality limit", action='store_true', default=False)
args = parser.parse_args()
print(args)
BLOCK_SIZE = args.blocksize
COLTHRESH = args.colthresh
TEXTHRESH = args.texthresh
FILE = args.file
DETAIL = args.detail
MULTI = args.multi

o_dir = "output//{0}-{1}-{2}-{3}-{4}//".format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, MULTI)
if not os.path.exists(o_dir):
    os.makedirs(o_dir)

img_orig = cv2.imread('input//{0}.jpg'.format(FILE),1)
img = cv2.imread('input//{0}.jpg'.format(FILE),1)
img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,w,chn = img.shape
mask_f = np.zeros((h+2,w+2),np.uint8)

print("w: {0}, h: {1}".format(w, h))
# Number of blocks which fit within image width/height, with margins at edge
inner_blocks_x = math.floor(w/BLOCK_SIZE)
inner_blocks_y = math.floor(h/BLOCK_SIZE)
print("ibx: {0}, iby:{1}".format(inner_blocks_x, inner_blocks_y))

# Add two to either side, will partially be out of image
blocks_x = inner_blocks_x if inner_blocks_x * BLOCK_SIZE == w else inner_blocks_x + 2
blocks_y = inner_blocks_y if inner_blocks_y * BLOCK_SIZE == h else inner_blocks_y + 2
# blocks_y = inner_blocks_y + 2

print("bx: {0}, by: {1}".format(blocks_x, blocks_y))


# To stop this overlap, grow image out to match exactly

extra_cols = (blocks_x * BLOCK_SIZE) - w
extra_rows = (blocks_y * BLOCK_SIZE) - h

top = math.floor(extra_rows / 2)
bottom = math.ceil(extra_rows / 2)
left = math.floor(extra_cols / 2)
right = math.ceil(extra_cols / 2)

print(img.shape)
img_bigger = cv2.copyMakeBorder(img_orig, top, bottom, left, right, cv2.BORDER_REPLICATE)

# img_grey = cv2.cvtColor(img_bigger, cv2.COLOR_RGB2GRAY)
img_grey =img_bigger[..., 2]
print(img_bigger.shape)

windows = []
bins = []

if DETAIL:
    init_rects = img_bigger.copy()
    for row in range(0, blocks_y):
        for col in range(0, blocks_x):
            t = row * BLOCK_SIZE
            l = col * BLOCK_SIZE
            b = (row+1) * BLOCK_SIZE
            r = (col+1) * BLOCK_SIZE
            cv2.rectangle(init_rects, (l+1, t+1), (r-1, b-1), (100, 255, 255), 1)
    cv2.imshow('window segmentation', cv2.cvtColor(init_rects, cv2.COLOR_HSV2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for row in range(0, blocks_y):
# for row in range(0, 1):
    for col in range(0, blocks_x):
        print(img_grey.shape)
        print("Row {0}/{2} Col {1}/{3}".format(row, col, blocks_y, blocks_x))
        block_id = row * blocks_x + col
        t = row * BLOCK_SIZE
        l = col * BLOCK_SIZE
        b = (row+1) * BLOCK_SIZE
        r = (col+1) * BLOCK_SIZE
        #print("({0},{1}), ({2},{3})".format(l,t, r, b))
        window = img_bigger[t:b, l:r]
        window_grey = img_grey[t:b, l:r]
        window_grey = window_grey.astype(int)
        #print(window_grey.shape)
        glcm = greycomatrix(window_grey, [5], [0], 256, symmetric=True, normed=True)
        dis = greycoprops(glcm, 'dissimilarity')[0, 0]
        con = greycoprops(glcm, 'contrast')[0, 0]
        cor = greycoprops(glcm, 'correlation')[0, 0]
        texture = np.array([dis, con, cor])
        # textures = mt.features.haralick(window_grey)
        
        # # take the mean of it and return it
        # texture = textures.mean(axis=0)
        #print(texture)
        # hist = cv2.calcHist([window], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        hist = cv2.calcHist([window], [0, 1, 2], None, [18, 25, 25],[0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        
        bin_id = len(bins)
        for w in windows:
            #print(texture)
            #print(w[1])
            text_dist = np.linalg.norm(texture - np.array(w[1]))
            #print(texture - w[1])
            #print("texture dist {0}".format(text_dist))
            # hist = hist.tolist()
            # hist = np.array(hist).flatten()
            # print(type(hist))
            # print(hist.shape)
            # hist2 = np.array(w[2]).flatten()
            hist2 = w[2]
            # print(type(hist2))
            # print(hist2.shape)
            col_dist = (1 - cv2.compareHist(hist, hist2, 0))
            #print("color dist {0}".format(col_dist))
            # total_dist = math.sqrt(math.pow(text_dist, 2) + math.pow(col_dist, 2))
            # col_weighted_dist = col_dist * text_dist
            # if total_dist < thresh:
            if col_dist < COLTHRESH and text_dist < TEXTHRESH:
                bin_id = w[4]
                #print(bin_id)
                #print("total dist {0}".format(total_dist))
                # cv2.imshow('comp1', window)
                # cv2.imshow('comp2', w[3])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                break



        windows.append([block_id, texture, hist, window, bin_id, [l, t, r, b]])
        if bin_id == len(bins):
            bins.append([])
        bins[bin_id].append([block_id, texture, hist, window, bin_id, [l, t, r, b]])
        
print(img_bigger[5:10, 5:10])


img_rect_pre_merge = img_bigger.copy()
num_bins = len(bins)
print(num_bins)
hue_step = max(math.floor(180/num_bins), 1)
print(hue_step)
hues = list(range(0, 180, hue_step))
print(hues)
for bin_set in bins:
    for wind in bin_set:
        l,t,r,b = wind[5]
        bin_id = wind[4]
        cv2.rectangle(img_rect_pre_merge, (l+1, t+1), (r-1, b-1), (hues[bin_id % 180], 255, 255), 1)

if DETAIL:
    cv2.imshow('post init segmentation', cv2.cvtColor(img_rect_pre_merge, cv2.COLOR_HSV2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bins_to_merge = {}
for w in windows:
    print("window {0} / {1}".format(w[0], len(windows)))
    a_bin_id = w[4]
    a_texture = w[1]
    a_hist = w[2]
    if a_bin_id not in bins_to_merge:
        bins_to_merge[a_bin_id] = set()

    for match_w in windows:
        b_bin_id = match_w[4]
        if b_bin_id in bins_to_merge[a_bin_id] or b_bin_id == a_bin_id:
            continue
        b_texture = match_w[1]
        b_hist = match_w[2]
        if b_bin_id not in bins_to_merge:
            bins_to_merge[b_bin_id] = set()
        
        text_dist = np.linalg.norm(a_texture - b_texture)
        col_dist = (1 - cv2.compareHist(a_hist, b_hist, 0))
        if col_dist < COLTHRESH *0.1 and text_dist < TEXTHRESH * 0.1:
                bins_to_merge[a_bin_id].add(b_bin_id)
                bins_to_merge[b_bin_id].add(a_bin_id)

visited = []
new_bins = []


for b in bins_to_merge.keys():
    print('merging {0}'.format(b))
    if b in visited:
        continue
    visited.append(b)
    new_bins.append([])
    for wind in bins[b]:
        wind[4] = len(new_bins) - 1
        windows[wind[0]][4] = len(new_bins) - 1
        new_bins[len(new_bins) - 1].append(wind)
    dfs_stack = [] + list(bins_to_merge[b])
    while len(dfs_stack) > 0:
        visiting = dfs_stack.pop()
        if visiting in visited:
            continue
        visited.append(visiting)
        for wind in bins[visiting]:
            wind[4] = len(new_bins) - 1
            windows[wind[0]][4] = len(new_bins) - 1
            new_bins[len(new_bins) - 1].append(wind)
        
        dfs_stack = dfs_stack + list(bins_to_merge[visiting])

bins = new_bins
img_rect_post_merge = img_bigger.copy()
num_bins = len(bins)
print(num_bins)
hue_step = max(math.floor(180/num_bins), 1)
print(hue_step)
hues = list(range(0, 180, hue_step))
print(hues)
for bin_set in bins:
    for wind in bin_set:
        l,t,r,b = wind[5]
        bin_id = wind[4]
        cv2.rectangle(img_rect_post_merge, (l+1, t+1), (r-1, b-1), (hues[bin_id % 180], 255, 255), 1)

if DETAIL:
    cv2.imshow('segs pre', cv2.cvtColor(img_rect_pre_merge, cv2.COLOR_HSV2BGR))
    cv2.imshow('segs post merge', cv2.cvtColor(img_rect_post_merge, cv2.COLOR_HSV2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_rect = img_bigger.copy()
print(img_bigger[5:10, 5:10])
bins.sort(key=len, reverse=True)
for curr_bin in bins[0]:
    l,t,r,b = curr_bin[5]
    cv2.rectangle(img_rect, (l, t), (r, b), (100, 255, 255), 1)
    b = curr_bin[3][..., 1]
    g = curr_bin[3][..., 1]
    r = curr_bin[3][..., 2]
    print('window')
    print(b)
    print(g)
    print(r)
# # for b in bins:
# #     print(len(b))
# #     cv2.imshow('comp2', b[0][3])
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()  
# print(img_bigger[5:10, 5:10])
if DETAIL:
    cv2.imshow('image',cv2.cvtColor(img_orig, cv2.COLOR_HSV2BGR))
    cv2.imshow('image background segments',cv2.cvtColor(img_rect, cv2.COLOR_HSV2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# cv2.imwrite('save-test.png',cv2.cvtColor(img_rect, cv2.COLOR_HSV2BGR))

print(img_bigger[5:10, 5:10])
floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)
mask = np.zeros((img_bigger.shape[0]+2, img_bigger.shape[1]+2),np.uint8)
mask2 = np.zeros((img_bigger.shape[0]+2, img_bigger.shape[1]+2),np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

ov_c1_std = 0
ov_c2_std = 0
ov_c3_std = 0
av_c1_std = 0
av_c2_std = 0
av_c3_std = 0


for curr_bin in bins[0]:
    c1 = curr_bin[3][..., 0]
    c2 = curr_bin[3][..., 1]
    c3 = curr_bin[3][..., 2]

    ov_c1_std = max(ov_c1_std, np.std(c1))
    ov_c2_std = max(ov_c2_std, np.std(c2))
    ov_c3_std = max(ov_c3_std, np.std(c3))

    av_c1_std += np.std(c1)
    av_c2_std += np.std(c2)
    av_c3_std += np.std(c3)

av_c1_std = av_c1_std / len(bins[0])
av_c2_std = av_c2_std / len(bins[0])
av_c3_std = av_c3_std / len(bins[0])

for curr_bin in bins[0]:
    l,t,r,b = curr_bin[5]
    print(l)
    print(t)
    print(r)
    print(b)
    point = (int((l+r)/2), int((t+b)/2))
    print(img_bigger[t, l])
    c1 = curr_bin[3][..., 0]
    c2 = curr_bin[3][..., 1]
    c3 = curr_bin[3][..., 2]
    std_multi = MULTI
    c1_std = np.std(c1)
    c2_std = np.std(c2)
    c3_std = np.std(c3)
    c1_std = (ov_c1_std + c1_std * 3) / 4
    c2_std = (ov_c2_std + c2_std * 3) / 4
    c3_std = (ov_c3_std + c3_std * 3) / 4
    c1_std = (av_c1_std + c1_std * 2) / 3
    c2_std = (av_c2_std + c2_std * 2) / 3
    c3_std = (av_c3_std + c3_std * 2) / 3
    # #curr_bin_hsv = cv2.cvt
    # print('window')
    # print(b)
    # print(g)
    # print(r)
    # print(img_bigger[point[1], point[0]])
    # cv2.imshow('img', cv2.cvtColor(curr_bin[3], cv2.COLOR_HSV2BGR))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.floodFill(img_bigger ,mask, point, 255, (c1_std * std_multi * 0.5, c2_std * std_multi, c3_std * std_multi), (np.std(c1) * std_multi, c2_std * std_multi, c3_std * std_multi * 0.5), floodflags)     # line 27
    if DETAIL:
        img_show_rect = img_rect.copy()
        cv2.rectangle(img_show_rect, (l, t), (r, b), (50, 255, 255), 1)
        
        mask2 = np.where((mask==0), 0, 1).astype('uint8')
        final = img_bigger*mask2[1:-1,1:-1,np.newaxis]
        cv2.rectangle(final, (l, t), (r, b), (50, 255, 255), 1)
        cv2.imshow('fill', cv2.cvtColor(final, cv2.COLOR_HSV2BGR))
        cv2.imshow('rect', cv2.cvtColor(img_show_rect, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
mask2 = np.where((mask==0), 0, 1).astype('uint8')
mask_inv = np.where((mask==255), 0, 1).astype('uint8')

kernel = np.ones((5,5),np.uint8)
    
opening = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

print(mask2.shape)
print(img_bigger.shape)

final = img_bigger*mask2[1:-1,1:-1,np.newaxis]
final_comp = img_bigger*mask_inv[1:-1,1:-1,np.newaxis]
cv2.line(img_orig,(point[0] - 5,point[1]),(point[0] + 5,point[1]),(255,0,0),1)
cv2.line(img_orig,(point[0],point[1] - 5),(point[0],point[1] + 5),(255,0,0),1)

background = img_bigger - final_comp

#Change all pixels in the background that are not black to white
background[np.where((background > [0,0,0]).all(axis = 2))] =[255,0,255]
final_comp = background + final_comp

im2, contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
approxes = []
for c in contours:
    # epsilon = 0.005*cv2.arcLength(c,True)
    if cv2.arcLength(c,True) > 50:
        epsilon = 2
        approx = cv2.approxPolyDP(c,epsilon,True)
        approxes.append(approx)

img_contour = img_bigger.copy()
img_contour2 = img_bigger.copy()
img_contour_indi = img_bigger.copy()
cv2.drawContours(img_contour, contours, -1, (0,0,255), 3)
cv2.drawContours(img_contour2, approxes, -1, (0,0,255), 3)


re_x = 250
re_y = 250
for i,a in enumerate(approxes):
    img_contour_indi = img_bigger.copy()
    cnt = a
    cv2.drawContours(img_contour_indi, [cnt], 0, (0,255,255), 3)
    
    approxed_mask = np.zeros((img_bigger.shape[0]+2, img_bigger.shape[1]+2),np.uint8)
    cv2.fillPoly(approxed_mask, pts=[a], color = (1))
    approxed_mask_inv = np.where((approxed_mask==1), 0, 1).astype('uint8')
    final_a = img_bigger*approxed_mask[1:-1,1:-1,np.newaxis]
    blank = np.zeros((img_bigger.shape[0], img_bigger.shape[1], 3),np.uint8)
    blank[:, :] = (180, 0, 255)
    approxed_comp = blank*approxed_mask_inv[1:-1,1:-1,np.newaxis] + final_a
    x,y,w,h = cv2.boundingRect(approxed_mask)
    print("width {0}, height {1}".format(w, h))
    scaling_factor = math.floor(min(re_x/w, re_y/h) * 100)/100
    print("sfw: {0}, sfh: {1}, sf {2}".format(re_x/w, re_y/h, scaling_factor))
    resized =cv2.resize(approxed_comp[y:y+h, x:x+w], (int(w*scaling_factor), int(h*scaling_factor)), interpolation=cv2.INTER_AREA)
    resized_x = resized.shape[1]
    resized_y = resized.shape[0]
    blank_64 = np.zeros((re_x, re_y, 3),np.uint8)
    blank_64[:, :] = (180, 0, 255)
    x_offset = (re_x - resized_x)/2
    y_offset = (re_y - resized_y)/2
    print("xoff: {0}, yoff: {1}".format(x_offset, y_offset))
    print("rw: {0}, rh: {1}".format(resized_x, resized_y))
    print("{0}:{1}, {2}:{3}".format(math.floor(y_offset),math.floor(y_offset)+resized_y, math.floor(x_offset),math.floor(x_offset)+resized_x))
    blank_64[math.floor(y_offset):math.floor(y_offset)+resized_y, math.floor(x_offset):math.floor(x_offset)+resized_x] = resized
    cv2.rectangle(img_contour_indi,(x,y),(x+w,y+h),(75,160,255),2)
    if cv2.arcLength(a,True) > 5:

        cv2.imwrite('{0}conponent-{1}.png'.format(o_dir, i), cv2.cvtColor(blank_64, cv2.COLOR_HSV2BGR))
        if DETAIL:
            cv2.imshow('con-approx', cv2.cvtColor(img_contour_indi, cv2.COLOR_HSV2BGR))
            cv2.imshow('components', cv2.cvtColor(blank_64, cv2.COLOR_HSV2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()




# cv2.imshow('components',cv2.cvtColor(final_comp, cv2.COLOR_HSV2BGR))
# cv2.imshow('components_a',cv2.cvtColor(approxed_comp, cv2.COLOR_HSV2BGR))
# cv2.imshow('img', cv2.cvtColor(img_bigger, cv2.COLOR_HSV2BGR))
# cv2.imshow('image-rects',cv2.cvtColor(img_rect, cv2.COLOR_HSV2BGR))
# cv2.imshow('background', cv2.cvtColor(final, cv2.COLOR_HSV2BGR))
# cv2.imshow('finala', cv2.cvtColor(final_a, cv2.COLOR_HSV2BGR))
# cv2.imshow('con', cv2.cvtColor(img_contour, cv2.COLOR_HSV2BGR))
# cv2.imshow('con-approx', cv2.cvtColor(img_contour2, cv2.COLOR_HSV2BGR))
# cv2.imshow('mask', mask2)
# cv2.imshow('im2', im2)
cv2.imwrite('{4}seg-components-{0}-{1}-{2}-{3}.png'.format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, o_dir),cv2.cvtColor(final_comp, cv2.COLOR_HSV2BGR))
cv2.imwrite('{4}seg-image-rects-{0}-{1}-{2}-{3}.png'.format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, o_dir),cv2.cvtColor(img_rect, cv2.COLOR_HSV2BGR))
cv2.imwrite('{4}seg-background-{0}-{1}-{2}-{3}.png'.format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, o_dir), cv2.cvtColor(final, cv2.COLOR_HSV2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(approxed_mask)
# plt.colorbar()
# plt.show()

with open('{0}run.txt'.format(o_dir), 'w+') as f:
    f.write("python seg-auto.py --file {0} --block {1} --texthresh {2} --colthresh {3} --multi {4}".format(FILE, BLOCK_SIZE, TEXTHRESH, COLTHRESH, MULTI))
