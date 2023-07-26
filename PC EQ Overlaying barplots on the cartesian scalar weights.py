# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:22:56 2023

@author: jahn39
"""


from nd2reader import ND2Reader
import cv2
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib as mtl



def cropBar(bar_img,width):
    crop = bar_img[25:155,20:145,:]
    br_bord = cv2.copyMakeBorder(crop, width, width, width, width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return br_bord

def place(select_row,select_col,background,img):
    rows,cols,lays = img.shape
    combo = background
    
    s_r = select_row + rows
    s_c = select_col + cols
    
    combo[select_row:s_r, select_col:s_c, :] = img
    
    return combo

def overBarCart(bgrd, br1, br2, br3, br4, br5, br6, br7, br8):
    add_col = np.full((720, 100, 3), 255)
    add_col = add_col.astype(np.uint8)
    ext_bgrd = np.concatenate((bgrd, add_col), axis=1)
    
    #PC 1
    background = ext_bgrd
    select_row = 85
    select_col = 822
    img = br1
    combo = place(select_row,select_col,background,img)
    #PC 2
    background = combo
    select_row = select_row + 135
    select_col = 822
    img = br2
    combo = place(select_row,select_col,background,img)
    #PC 3
    background = combo
    select_row = select_row + 135
    select_col = 822
    img = br3
    combo = place(select_row,select_col,background,img)
    #PC 4
    background = combo
    select_row = select_row + 136
    select_col = 822
    img = br4
    combo = place(select_row,select_col,background,img)
    
    #PC 5
    background = combo
    select_row = 85
    select_col = 952
    img = br5
    combo = place(select_row,select_col,background,img)
    #PC 6
    background = combo
    select_row = select_row + 135
    select_col = 952
    img = br6
    combo = place(select_row,select_col,background,img)
    #PC 7
    background = combo
    select_row = select_row + 135
    select_col = 952
    img = br7
    combo = place(select_row,select_col,background,img)
    #PC 8
    background = combo
    select_row = select_row + 136
    select_col = 952
    img = br8
    combo = place(select_row,select_col,background,img)

    return combo    
    
    

#Add directory of the cartesian plot representations of the scalar weights
directory_cartesian = ""
#Add a save directory folder
save_fold = ""

#Add directories for each of the barplots
bar1 = ""
bar2 = ""
bar3 = ""
bar4 = ""

bar5 = ""
bar6 = ""
bar7 = ""
bar8 = ""



dir_cart = directory_cartesian
li_dir_cart = os.listdir(dir_cart)
srt_cart = natsorted(li_dir_cart)

save_dir = save_fold

width = 5
for i in range(len(srt_cart)):
    print(i)
    cart_path = os.path.join(dir_cart,srt_cart[i])
    
    bgrd = cv2.imread(cart_path)
    
    br1 = cv2.imread(bar1 + "/frame " + str(i+1) + ".png")
    br2 = cv2.imread(bar2 + "/frame " + str(i+1) + ".png")
    br3 = cv2.imread(bar3 + "/frame " + str(i+1) + ".png")
    br4 = cv2.imread(bar4 + "/frame " + str(i+1) + ".png")
    
    br5 = cv2.imread(bar5 + "/frame " + str(i+1) + ".png")
    br6 = cv2.imread(bar6 + "/frame " + str(i+1) + ".png")
    br7 = cv2.imread(bar7 + "/frame " + str(i+1) + ".png")
    br8 = cv2.imread(bar8 + "/frame " + str(i+1) + ".png")
    
    br1 = cropBar(br1, width)
    br2 = cropBar(br2, width)
    br3 = cropBar(br3, width)
    br4 = cropBar(br4, width)
    
    br5 = cropBar(br5, width)
    br6 = cropBar(br6, width)
    br7 = cropBar(br7, width)
    br8 = cropBar(br8, width)

    combo = overBarCart(bgrd, br1, br2, br3, br4, br5, br6, br7, br8)
    plt.imshow(combo)
    cv2.imwrite(save_dir + "/frame " + str(i+1) + ".png",combo)


