# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:11:45 2023

@author: jahn39
"""

from nd2reader import ND2Reader
import cv2
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import matplotlib as mtl
import matplotlib.pyplot as plt

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


def kernColGrad(img,size, color_codes, x_list, y_list):
    count = -1
    for i,j in zip(x_list,y_list):
        #print(j,i)
        count = count + 1
        
        b = color_codes[0][count]
        g = color_codes[1][count]
        r = color_codes[2][count]
        
        row_pts, col_pts = kernel(size,j,i)
        #print(b,g,r)
        for k,l in zip(row_pts,col_pts):
            #print(k,l)
            img[k,l] = [r,g,b]
        #print(img[k,l])
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


def makeColor(pix_number):
    r = 0
    g = 150
    b = 255
    alpha = 0.2
    pix_rl = []
    pix_gl = []
    pix_bl = []
    count = -1
    for i in range(pix_number):
        count = count + 1
        if count <256:
            r = r + 1
            if r == 256:
                r = 255
            pix_rl.append(r)
            pix_gl.append(g)
            pix_bl.append(b)
        else:
            r = r-1
            g = (g + 1)%255
            pix_rl.append(r)
            pix_gl.append(g)
            pix_bl.append(b)

    return pix_bl,pix_gl,pix_rl

def kernPosNeg(img, size, neg, pos, x_list, y_list):
    r1,g1,b1 = mtl.colors.to_rgb("yellow")
    r2,g2,b2 = mtl.colors.to_rgb("deepskyblue")
    
    count = -1
    
    for i in neg:
        row_pts, col_pts = kernel(size,y_list[i],x_list[i])
        for k,l in zip(row_pts,col_pts):
            img[k,l] = [r2*255,g2*255,b2*255]
        #print(img[k,l])

    for j in pos:
        row_pts, col_pts = kernel(size,y_list[j],x_list[j])
        for k,l in zip(row_pts,col_pts):
            img[k,l] = [r1*255,g1*255,b1*255]
        #print(img[k,l])
        
    return img


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

apply = addWei(df_wts,df_UniPC)

new_fold_path = save_dir
    
    

degs = np.arange(0,np.pi*2,np.pi*2/360)
size = 1
cx = 200
cy = 200
img = np.zeros((400,400,3))
pix_number = 360
color_codes = makeColor(pix_number)

c_radii = [24]*360
c_cx = 350
c_cy = 50
cx_list, cy_list = polFig2Cart(degs, c_radii, c_cx, c_cy)
for i in range(len(apply)):
    name = i + 1
    
    img = np.zeros((400,400,3))
    spec_rad = apply[i]
    
    neg = [index for index,value in enumerate(spec_rad) if value<0]
    pos = [index for index,value in enumerate(spec_rad) if value>0]
    
    c2_cx = 200
    c2_cy = 200
    c2_radii = [100]*360
    c2x_list, c2y_list = polFig2Cart(degs, c2_radii, c2_cx, c2_cy)
    kern = kernPosNeg(img, size, neg, pos, c2x_list, c2y_list)
    
            
    x2_list, y2_list = polFig2Cart(degs, spec_rad, cx, cy)
    color_lines = kernColGrad(kern, size, color_codes, x2_list, y2_list)
    
    circ = kernColGrad(color_lines, size, color_codes, cx_list, cy_list)
    
    plt.imshow(circ.astype(np.uint8))
    plt.text(25,25,"Positive",color="yellow",fontsize=12)
    plt.text(25,50,"Negative",color="deepskyblue",fontsize=12)
    plt.text(332,52,"Degrees",color="white",fontsize=5)
    plt.text(380,50,"0°",color="white",fontsize=5)
    plt.text(343,20,"90°",color="white",fontsize=5)
    plt.text(300,50,"180°",color="white",fontsize=5)
    plt.text(340,87,"270°",color="white",fontsize=5)
    plt.title("Frame " + str(name))
    plt.savefig(new_fold_path + "/frame " + str(name) + ".jpg")
    plt.show()
    