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
import contour_obj

def round_color(col, num_bins):
    #print(col)
    col_0 = math.floor(col[0] / num_bins) * num_bins
    col_1 = math.floor(col[1] / num_bins) * num_bins
    col_2 = math.floor(col[2] / num_bins) * num_bins
    ret_col = np.array((col_0, col_1, col_2), np.uint8)
    #print(ret_col)
    return ret_col

def seg(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI):
    # Creates output directory
    o_dir = "output//{0}-{1}-{2}-{3}-{4}//".format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, MULTI)
    if not os.path.exists(o_dir):
        os.makedirs(o_dir)

    # Load in image, convert to HSV
    img_orig = cv2.imread('input//{0}.jpg'.format(FILE),1)
    img = cv2.imread('input//{0}.jpg'.format(FILE),1)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,w,chn = img.shape
    mask_f = np.zeros((h+2,w+2),np.uint8)

    BLOCK_SIZE = int(w * BLOCK_SIZE)

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

    img_bigger = cv2.copyMakeBorder(img_orig, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # Get the grey values (only works with HSV)
    img_grey =img_bigger[..., 2]

    windows = []
    bins = []

    # If asked, prints out image with windows overlayed
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
        cv2.imwrite('windows.png', cv2.cvtColor(init_rects, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Creates windows, gets their texture and color histograms
    for row in range(0, blocks_y):
        for col in range(0, blocks_x):
            print("Row {0}/{2} Col {1}/{3}".format(row, col, blocks_y, blocks_x))
            block_id = row * blocks_x + col
            t = row * BLOCK_SIZE
            l = col * BLOCK_SIZE
            b = (row+1) * BLOCK_SIZE
            r = (col+1) * BLOCK_SIZE

            window = img_bigger[t:b, l:r]
            window_grey = img_grey[t:b, l:r]
            window_grey = window_grey.astype(int)

            # Texture metrics are Harlick features
            glcm = greycomatrix(window_grey, [5], [0], 256, symmetric=True, normed=True)
            dis = greycoprops(glcm, 'dissimilarity')[0, 0]
            con = greycoprops(glcm, 'contrast')[0, 0]
            cor = greycoprops(glcm, 'correlation')[0, 0]
            texture = np.array([dis, con, cor])

            # Color metric is the image histogram (In HSV)
            hist = cv2.calcHist([window], [0, 1, 2], None, [18, 25, 25],[0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, None).flatten()

            windows.append([block_id, texture, hist, window, None, [l, t, r, b]])

    # Normalizes texture features
    dis_vals = list(map(lambda w: w[1][0], windows))
    dis_max = max(dis_vals)
    dis_min = min(dis_vals)
    con_vals = list(map(lambda w: w[1][1], windows))
    con_max = max(con_vals)
    con_min = min(con_vals)
    cor_vals = list(map(lambda w: w[1][2], windows))
    cor_max = max(cor_vals)
    cor_min = min(cor_vals)
    for a in windows:
        a[1][0] = ((a[1][0] - dis_min) / (dis_max - dis_min))
        a[1][1] = ((a[1][1] - con_min) / (con_max - con_min))
        a[1][2] = ((a[1][2] - cor_min) / (cor_max - cor_min))
        

    # Compares windows to all other windows, if similar add to the same bin
    for ia,a in enumerate(windows):            
        bin_id = len(bins)
        texture = np.array(a[1])
        hist = a[2]
        
        for b in windows[:ia]:
            if a == b:
                continue
            
            text_dist = np.linalg.norm(texture - np.array(b[1]))
            hist2 = b[2]
            col_dist = (1 - cv2.compareHist(hist, hist2, 0))
            if col_dist < COLTHRESH and text_dist < TEXTHRESH:
                bin_id = b[4]
                break
        if bin_id == len(bins):
            bins.append([])
        a[4] = bin_id
        bins[bin_id].append(a)
        
            

    # Draw bins onto image in different hues
    img_rect_pre_merge = img_bigger.copy()
    num_bins = len(bins)
    hue_step = max(math.floor(180/num_bins), 1)
    hues = list(range(0, 180, hue_step))
    for bin_set in bins:
        for wind in bin_set:
            l,t,r,b = wind[5]
            bin_id = wind[4]
            cv2.rectangle(img_rect_pre_merge, (l+1, t+1), (r-1, b-1), (hues[bin_id % 180], 255, 255), 1)

    if DETAIL:
        cv2.imshow('post init segmentation', cv2.cvtColor(img_rect_pre_merge, cv2.COLOR_HSV2BGR))
        cv2.imwrite('windowGroups.png', cv2.cvtColor(img_rect_pre_merge, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Calculate if we can merge bins
    # We merge if any items in two bins are close enough
    # This thresh is smaller to avoid chaining
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

    # Actually merge bins with DFS
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

    # Draw post merge bins
    bins = new_bins
    img_rect_post_merge = img_bigger.copy()
    num_bins = len(bins)
    hue_step = max(math.floor(180/num_bins), 1)
    hues = list(range(0, 180, hue_step))
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

    # We select the largest bin as the background
    # Here we draw these windows and save them
    bins.sort(key=len, reverse=True)
    img_rect = img_bigger.copy()
    windows_img = np.zeros((BLOCK_SIZE, len(bins[0])*BLOCK_SIZE, 3), np.uint8)
    print(windows_img.shape)

    for icb,curr_bin in enumerate(bins[0]):
        l,t,r,b = curr_bin[5]
        cv2.rectangle(img_rect, (l, t), (r, b), (100, 255, 255), 1)
        b = curr_bin[3][..., 1]
        g = curr_bin[3][..., 1]
        r = curr_bin[3][..., 2]
        print("{0}:{1}".format(icb*BLOCK_SIZE, (icb+1)*BLOCK_SIZE))
        print(windows_img[:, icb*BLOCK_SIZE:(icb+1)*BLOCK_SIZE].shape)
        print(curr_bin[3].shape)
        windows_img[:, icb*BLOCK_SIZE:(icb+1)*BLOCK_SIZE] = curr_bin[3]

    cv2.imwrite("{0}background_windows.png".format(o_dir), cv2.cvtColor(windows_img, cv2.COLOR_HSV2BGR))


    if DETAIL:
        cv2.imshow('image',cv2.cvtColor(img_orig, cv2.COLOR_HSV2BGR))
        cv2.imshow('image background segments',cv2.cvtColor(img_rect, cv2.COLOR_HSV2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



    # With these windows, we can calculate the two parameters for floodfill
    # We get starting points as the center of each window
    # We get upper/lower thresh as a function of standard deviation in window
    floodflags = 4
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)
    mask = np.zeros((img_bigger.shape[0]+2, img_bigger.shape[1]+2),np.uint8)
    #mask2 = np.zeros((img_bigger.shape[0]+2, img_bigger.shape[1]+2),np.uint8)

    # Calculate channel standard deviation over all windows
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


    std_multi = (1 + (1 - ((BLOCK_SIZE * 21)/ img_bigger.shape[1]))) * MULTI
    shown_windows = 0
    for c_1_m in range(10, 11):
        for c_2_m in range(8, 9):  
            for c_3_m in range(9, 10):
                mask = np.zeros((img_bigger.shape[0]+2, img_bigger.shape[1]+2),np.uint8)
                mask[0:top, :] = 255
                mask[:, 0:left] = 255
                if(bottom > 0):
                    mask[-bottom:, :] = 255
                if right > 0 :
                    mask[:, -right:] = 255
                for curr_bin in bins[0]:
                    l,t,r,b = curr_bin[5]
                    point = (int((l+r)/2), int((t+b)/2))
                    c1 = curr_bin[3][..., 0]
                    c2 = curr_bin[3][..., 1]
                    c3 = curr_bin[3][..., 2]
                    c1_std = np.std(c1)
                    c2_std = np.std(c2)
                    c3_std = np.std(c3)

                    # We weight this windows standard deviation with total
                    c1_std = (av_c1_std * 2 + c1_std ) / 3
                    c2_std = (av_c2_std * 2 + c2_std) / 3
                    c3_std = (av_c3_std * 2 + c3_std) / 3
                    # c1_std = (av_c1_std)
                    # c2_std = (av_c2_std)
                    # c3_std = (av_c3_std)

                    # Each channel gets an empirically calced multiplyer
                    c1_thresh = std_multi * c_1_m / 10
                    c2_thresh = std_multi * c_2_m / 10
                    c3_thresh = std_multi * c_3_m / 10
                    print(std_multi)
                    # Floodfill sets mask to 255 where background is
                    cv2.floodFill(img_bigger ,mask, point, 255, (c1_std * c1_thresh, c2_std * c2_thresh, c3_std * c3_thresh), (c1_std * c1_thresh, c2_std * c2_thresh, c3_std * c3_thresh), floodflags)     # line 27
                    if DETAIL and shown_windows < 5:
                        img_show_rect = img_rect.copy()
                        cv2.rectangle(img_show_rect, (l, t), (r, b), (50, 255, 255), 1)
                        
                        mask2 = np.where((mask==0), 0, 1).astype('uint8')
                        final = img_bigger*mask2[1:-1,1:-1,np.newaxis]
                        cv2.rectangle(final, (l, t), (r, b), (50, 255, 255), 1)
                        cv2.imshow('fill', cv2.cvtColor(final, cv2.COLOR_HSV2BGR))
                        cv2.imshow('rect', cv2.cvtColor(img_show_rect, cv2.COLOR_HSV2BGR))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        shown_windows += 1
                    cv2.line(img_rect,(point[0] - 5,point[1]),(point[0] + 5,point[1]),(255,0,0),1)
                    cv2.line(img_rect,(point[0],point[1] - 5),(point[0],point[1] + 5),(255,0,0),1)
                
                # Mask is 0 for foreground, 1 in background
                mask2 = np.where((mask==0), 0, 1).astype('uint8')
                mask_inv = np.where((mask==255), 0, 1).astype('uint8')

                # Open to remove noise
                kernel = np.ones((5,5),np.uint8)
                kernel3 = np.ones((3,3),np.uint8)
                    
                opening = mask2
                #opening = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
                opening = cv2.dilate(opening,kernel3,iterations = 1)
                opening = cv2.dilate(opening,kernel3,iterations = 1)
                opening = cv2.erode(opening,kernel3,iterations = 1)
                #opening = cv2.dilate(opening,kernel3,iterations = 1)

                mask_inv = np.where((opening==1), 0, 1).astype('uint8')
                # Final comp is the components
                final = img_bigger*opening[1:-1,1:-1,np.newaxis]
                final_comp = img_bigger*mask_inv[1:-1,1:-1,np.newaxis]
                background = img_bigger - final_comp

                #Change all pixels in the background that are not black to white
                background[np.where((background > [0,0,0]).all(axis = 2))] =[255,0,255]
                final_comp = background + final_comp
                cv2.imwrite('{0}//test3-{1}-{2}-{3}.png'.format(o_dir, c_1_m, c_2_m, c_3_m),cv2.cvtColor(final_comp, cv2.COLOR_HSV2BGR))


    # Get contours of the mask
    im2, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Simplify these and cull small objects
    approxes = []
    max_area = img_bigger.shape[0] * img_bigger.shape[1]
    for c in contours:
        cx,cy,cw,ch = cv2.boundingRect(c)
        if cv2.arcLength(c,True) > 50 and cv2.contourArea(c) < 0.75 * max_area:
            epsilon = 6
            approx = cv2.approxPolyDP(c,epsilon,True)
            approxes.append(approx)


    # Crop components to seperate images
    img_contour_indi = img_bigger.copy()

    re_x = 256
    re_y = 256

    for i,a in enumerate(approxes):
        print("component {0}/{1}".format(i, len(approxes)))
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

        textured_bg = np.zeros((img_bigger.shape[0], img_bigger.shape[1], 3),np.uint8)
        color_pairs = {}
        num_bins = 4
        for ri in range(y, min(y+h, final_a[1:-1, 1:-1].shape[0])):
            for ci in range(x, min(x+w, final_a[1:-1, 1:-1].shape[1])):
                #print("{0}:{1} {2}".format(x, min(x+w, final_a[1:-1, 1:-1].shape[1]), final_a.shape[1]))
                print("finding neighbors {0}\{1} \r".format(y*x + x, min(y+h, final_a[1:-1, 1:-1].shape[0]) * min(x+w, final_a[1:-1, 1:-1].shape[1])), end="", flush=True)
                if np.all(final_a[ri, ci] == 0):
                    continue
                rounded_pix = round_color(final_a[1:-1, 1:-1][ri, ci], num_bins)
                pix_str = str(rounded_pix.tolist())
                if pix_str not in color_pairs:
                    color_pairs[pix_str] = {}
                    color_pairs[pix_str]['all'] = []
                for n_y_offset in range(-1, 2):
                    for n_x_offset in range(-1, 2):
                        if n_y_offset == 0 and n_x_offset == 0:
                            continue
                        
                        neighbor_row = ri + n_y_offset
                        neighbor_col = ci + n_x_offset

                        if neighbor_row < 0 or neighbor_row >= min(y+h, final_a[1:-1, 1:-1].shape[0]):
                            continue
                        if neighbor_col < 0 or neighbor_col >= min(x+w, final_a[1:-1, 1:-1].shape[1]):
                            continue
                        if np.all(final_a[neighbor_row, neighbor_col] == 0):
                            continue
                        
                        pix_from_neighbor_offset_y = n_y_offset * -1
                        pix_from_neighbor_offset_x = n_x_offset * -1

                        rounded_neighbor = round_color(final_a[neighbor_row, neighbor_col], num_bins)

                        neighbor_str = str(rounded_neighbor.tolist())
                        direction_str = str([pix_from_neighbor_offset_y, pix_from_neighbor_offset_x])

                        if direction_str not in color_pairs[pix_str]:
                            color_pairs[pix_str][direction_str] = []
                        
                        color_pairs[pix_str][direction_str].append(rounded_neighbor)
                        color_pairs[pix_str]['all'].append(rounded_neighbor)
               
        
        scaling_factor = math.floor(min(re_x/w*0.6, re_y/h*0.6) * 100)/100
        resized =cv2.resize(approxed_comp[y:y+h, x:x+w], (int(w*scaling_factor), int(h*scaling_factor)), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(approxed_mask[1:-1, 1:-1][y:y+h, x:x+w], (int(w*scaling_factor), int(h*scaling_factor)), interpolation=cv2.INTER_AREA)
        #resized_tex = cv2.resize(textured_bg[y:y+h, x:x+w], (int(w*scaling_factor), int(h*scaling_factor)), interpolation=cv2.INTER_AREA)
        resized_x = resized.shape[1]
        resized_y = resized.shape[0]
        blank_64 = np.zeros((re_x, re_y, 3),np.uint8)
        blank_64[:, :] = (180, 0, 255)
        x_offset = (re_x - resized_x)/2
        y_offset = (re_y - resized_y)/2



        blank_64[math.floor(y_offset):math.floor(y_offset)+resized_y, math.floor(x_offset):math.floor(x_offset)+resized_x] = resized
        
        textured_bg = textured_bg[0:re_y, 0:re_x]
        textured_bg[0, 0] = np.array(json.loads(list(color_pairs.keys())[0]))
        for ri in range(0, re_y):
            for ci in range(0, re_x):
                print("making tex {0}\{1} \r".format(ri * re_x + ci, re_y * re_x), end="", flush=True)
                if ri >= math.floor(y_offset) and ci >= math.floor(x_offset) and ri < math.floor(y_offset)+resized_y and ci < math.floor(x_offset)+resized_x and not np.all(resized_mask[:, :, np.newaxis][int(ri - y_offset), int(ci - x_offset)] == 0):
                    continue
                #possible_colors = [ c for c in list(color_pairs.keys())]
                possible_colors = []
                for n_y_offset in range(-1, 2):
                    for n_x_offset in range(-1, 2):
                        if n_y_offset == 0 and n_x_offset == 0:
                            continue
                        
                        neighbor_row = ri + n_y_offset
                        neighbor_col = ci + n_x_offset

                        if neighbor_row < 0 or neighbor_row >= re_y:
                            continue
                        if neighbor_col < 0 or neighbor_col >= re_x:
                            continue
                        if neighbor_row >= math.floor(y_offset) and neighbor_col >= math.floor(x_offset) and neighbor_row < math.floor(y_offset)+resized_y and neighbor_col < math.floor(x_offset)+resized_x and not np.all(resized_mask[:, :, np.newaxis][int(neighbor_row - y_offset), int(neighbor_col - x_offset)] == 0):
                            continue
                        pix_from_neighbor_offset_y = n_y_offset * -1
                        pix_from_neighbor_offset_x = n_x_offset * -1
                        try:
                            rounded_neighbor = round_color(textured_bg[neighbor_row, neighbor_col], num_bins)
                        except:
                            continue
                        if np.all(rounded_neighbor == 0):
                            continue
                        neighbor_str = str(rounded_neighbor.tolist())
                        direction_str = str([pix_from_neighbor_offset_y, pix_from_neighbor_offset_x])
                        if neighbor_str not in color_pairs:
                            #print(neighbor_row, neighbor_col)
                            test_pix = blank_64.copy()
                            test_pix[neighbor_row, neighbor_col] = [75, 255, 255]
                        elif direction_str in color_pairs[neighbor_str]:
                            possibilities = [ c for c in color_pairs[neighbor_str][direction_str]]
                            possible_colors += possibilities

                if len(possible_colors) == 0:
                    #print('oh no')
                    possible_colors += [ c for c in list(color_pairs.keys())]
                rand_col_index = np.random.randint(0, len(possible_colors))
                rand_col = possible_colors[rand_col_index]
                #print(rand_col)
                if isinstance(rand_col, str):  
                    rand_col = np.array(json.loads(rand_col))
                try:
                    textured_bg[ri, ci] = rand_col
                except:
                    continue
            


        #cv2.destroyAllWindows()
        #cv2.imshow('test', cv2.cvtColor(textured_bg, cv2.COLOR_HSV2BGR))
        #cv2.imshow('testpix', cv2.cvtColor(test_pix, cv2.COLOR_HSV2BGR))
        #cv2.waitKey(0)
        # cv2.destroyAllWindows()
        try:
            textured_bg[math.floor(y_offset):math.floor(y_offset)+resized_y, math.floor(x_offset):math.floor(x_offset)+resized_x] = textured_bg[math.floor(y_offset):math.floor(y_offset)+resized_y, math.floor(x_offset):math.floor(x_offset)+resized_x] + resized*resized_mask[:, :, np.newaxis]
        except:
            continue
        print('making 3d')
        contour_points = []
        last_point = None
        for point in a:
            new_point = [int(((point[0][0] - x) * scaling_factor + x_offset)*1), int((((point[0][1]- y)*scaling_factor + y_offset))*1)*-1]
            if new_point == last_point or (last_point != None and np.linalg.norm(np.array(new_point) - np.array(last_point)) < 2):
                continue
            else:
                last_point = new_point
            contour_points.append(new_point)
        if(len(contour_points) < 3):
            
            cv2.imshow('con-approx', cv2.cvtColor(img_contour_indi, cv2.COLOR_HSV2BGR))
            cv2.imshow('components', cv2.cvtColor(blank_64, cv2.COLOR_HSV2BGR))
            cv2.imshow('tex', cv2.cvtColor(textured_bg, cv2.COLOR_HSV2BGR))
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue
        contour_points.reverse()
        try:
            obj_str = contour_obj.obj_triang(contour_points, re_x, re_y)

            with open('{0}component-{1}.obj'.format(o_dir, i), 'w+') as wf:
                wf.write(obj_str)

            cv2.rectangle(img_contour_indi,(x,y),(x+w,y+h),(75,160,255),2)
            if cv2.arcLength(a,True) > 5:

                blank_alpha = cv2.cvtColor(blank_64, cv2.COLOR_HSV2BGR)
                blank_alpha = cv2.cvtColor(blank_alpha, cv2.COLOR_BGR2BGRA)
                for r,row in enumerate(blank_alpha):
                    for c,col in enumerate(blank_alpha):
                        if blank_alpha[r][c][0] == 180 and blank_alpha[r][c][1] == 0 and blank_alpha[r][c][2] == 255:
                            blank_alpha[r][c][3] = 0
                        else:
                            print(blank_alpha[r][c])
                            blank_alpha[r][c][3] = 255
                            print(blank_alpha[r][c])

                print(blank_alpha.shape)
                cv2.imwrite('{0}conponent-{1}-{2}-{3}-{4}-img.png'.format(o_dir, i, w, h, scaling_factor), blank_alpha)
                cv2.imwrite('{0}conponent-{1}-{2}-{3}-{4}-mask.png'.format(o_dir, i, w, h, scaling_factor), resized_mask*255)
                cv2.imwrite('{0}conponent-{1}-{2}-{3}-{4}-tex.png'.format(o_dir, i, w, h, scaling_factor), cv2.cvtColor(textured_bg, cv2.COLOR_HSV2BGR))
                if DETAIL:
                    cv2.imshow('con-approx', cv2.cvtColor(img_contour_indi, cv2.COLOR_HSV2BGR))
                    cv2.imshow('components', cv2.cvtColor(blank_64, cv2.COLOR_HSV2BGR))
                    cv2.imshow('tex', cv2.cvtColor(textured_bg, cv2.COLOR_HSV2BGR))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        except:
            print("making 3d failed")





    cv2.imwrite('{4}seg-components-{0}-{1}-{2}-{3}.png'.format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, o_dir),cv2.cvtColor(final_comp, cv2.COLOR_HSV2BGR))
    cv2.imwrite('{4}seg-image-rects-{0}-{1}-{2}-{3}.png'.format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, o_dir),cv2.cvtColor(img_rect, cv2.COLOR_HSV2BGR))
    cv2.imwrite('{4}seg-background-{0}-{1}-{2}-{3}.png'.format(FILE, BLOCK_SIZE, COLTHRESH, TEXTHRESH, o_dir), cv2.cvtColor(final, cv2.COLOR_HSV2BGR))

    with open('{0}run.txt'.format(o_dir), 'w+') as f:
        f.write("python seg-auto.py --file {0} --block {1} --texthresh {2} --colthresh {3} --multi {4}".format(FILE, BLOCK_SIZE, TEXTHRESH, COLTHRESH, MULTI))


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
    args = parser.parse_args()
    BLOCK_SIZE = args.blocksize 
    COLTHRESH = args.colthresh
    TEXTHRESH = args.texthresh
    FILE = args.file
    DETAIL = args.detail
    MULTI = args.multi
    seg(BLOCK_SIZE, COLTHRESH, TEXTHRESH, FILE, DETAIL, MULTI)
