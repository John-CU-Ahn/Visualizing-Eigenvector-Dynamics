# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:23:14 2023

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

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def polFig2Cart(degs,radii,cx,cy):    
    xlist = []
    ylist = []
    count = -1
    #print(len(degs))
    #print(len(radii))
    for i,j in zip(degs,radii):
        count = count + 1
        x,y = pol2cart(i,j)
        #print("count is: " + str(count),"deg is: " + str(i),"rad is: " + str(j))
        xlist.append(int(x)+cx)
        ylist.append(int(y)+cy)
    
    return xlist,ylist


def addWei(df_wts,df_PC):
    rows,cols = df_wts.shape
    
    applied = []
    for i in range(cols):
        t = df_wts.iloc[:,i]
        
        scaled_PC = []
        for j in range(8):
            row = df_PC.iloc[j,:]*t[j]
            scaled_PC.append(row)
            
        total = [0]*360
        for k in scaled_PC:
            total = total + k
    
        applied.append(total)
    
    return applied


def roLine(series, window):
    le = len(series)
    
    sli = int((window-1)/2)
    beg_window = pd.Series(series[0:sli])
    
    sli_e = le-sli + 1
    end_window = pd.Series(series[sli_e:])
    
    a = pd.Series(series)
    
    win_list = pd.concat([end_window,a,beg_window])
    
    
    new_list = []
    norm_i = sli
    for i in range(le):
        norm_i = norm_i + 1
        
        b = int(norm_i-(window-1)/2)
        e = int(norm_i+(window-1)/2)
        
        avrg = win_list[b:e].mean()
        new_list.append(avrg)
    
    if window == 0:
        new_list = series
    
    return new_list

def repRoLine(series, window, repeat):
    y = series
    for i in range(repeat):
        y = roLine(y,window)
    new_list = y
    
    if window == 0:
        new_list = series
    
    return new_list


def repRoDf(df_wts,window,repeat):
    rows,cols = df_wts.shape
    
    rep_l = []
    for i in range(rows):
        new = repRoLine(df_wts.iloc[i,:],window,repeat)
        rep_l.append(new)
    
    rep_l = pd.DataFrame(rep_l)
    
    return rep_l

def kernel(size,row,col):
    number = row
    rows=[]
    pos_neg = -1
    for i in range(2):
        pos_neg = pos_neg*-1
        for j in range(size):
            add = (j + 1)*pos_neg
            z = number + add
            rows.append(z)
            #print(rows)
    rows.append(number)
    rows.sort()
    mult_r = []
    for i in rows:
        y = [i]*(size*2+1)
        mult_r.append(y)
    row_pts = (np.array(mult_r)).flatten()
    
    number = col
    cols=[]
    pos_neg = -1
    for i in range(2):
        pos_neg = pos_neg*-1
        for j in range(size):
            add = (j + 1)*pos_neg
            z = number + add
            cols.append(z)
            #print(cols)
    cols.append(number)
    cols.sort()
    #print(cols)
    mult_c = []
    for i in cols:
        y = [i]*(size*2+1)
        mult_c.append(y)
    array = np.array(mult_c)
    t_array = array.T
    mult_c_t = t_array.tolist()
    col_pts = (np.array(mult_c_t)).flatten()
    
    return row_pts,col_pts

def kernCol(img,size, color, x_list, y_list):
    r,g,b = mtl.colors.to_rgb(color)
    for i,j in zip(x_list,y_list):
        row_pts, col_pts = kernel(size,j,i)
        for k,l in zip(row_pts,col_pts):
            #print(k,l)
            img[k,l] = [b*255,g*255,r*255]
    return img

def kernColTest(img,size, x_list, y_list):    
    for i,j in zip(x_list,y_list):
        row_pts, col_pts = kernel(size,j,i)
        for k,l in zip(row_pts,col_pts):
            #print(k,l)
            img[k,l] = 255
    return img

def addWei(df_wts,df_PC):
    rows,cols = df_wts.shape
    
    applied = []
    for i in range(cols):
        t = df_wts.iloc[:,i]
        
        scaled_PC = []
        for j in range(8):
            row = df_PC.iloc[j,:]*t[j]
            scaled_PC.append(row)
            
        total = [0]*360
        for k in scaled_PC:
            total = total + k
    
        applied.append(total)
    
    return applied


#Add directory for the principal components
dir_PC = ""
#Add directory for the principal component weights
dir_wts = ""
#Add a save directory for the barplots
save_dir = ""

df_UniPC = pd.read_csv(dir_PC,index_col=0)
df_wts = pd.read_csv(dir_wts,index_col=0)

window = 10
repeat = 3
df_wts_rep = repRoDf(df_wts,window,repeat)
df_wts_rep = abs(df_wts_rep)

plt.rcParams['font.size'] = '4'
my_dpi = 141
mx = (df_wts_rep.max(axis=1)).max()

rows,cols = df_wts_rep.shape
colors = ["tab:red","tab:orange","gold","tab:green",
          "tab:blue","tab:purple","tab:pink","tab:gray"]
base_dir = save_dir
count = 0
for i in range(cols):
    count = count + 1
    pc_num = 0
    for j in range(rows):
        pc_num = pc_num + 1
        
        wt = df_wts_rep.iloc[j,i]
        
        lab = "PC " + str(pc_num)
        
        f_dir = base_dir + "/fit " + lab
        
        fig, ax = plt.subplots(figsize=(25/my_dpi, 50/my_dpi), dpi=my_dpi)
        ax.bar(lab,wt,color=colors[j])
        ax.set_ylim((0,mx))
        #plt.tight_layout()
        plt.savefig(f_dir + "/frame " + str(count),bbox_inches='tight',dpi=300)














