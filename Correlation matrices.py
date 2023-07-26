# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:26:30 2023

@author: jahn39
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from natsort import natsorted
from scipy import stats

def iterDir(directory,common_name):
    listdir = os.listdir(directory)
    ord_dir = natsorted(listdir)
    sel_dir = [i for i in ord_dir if common_name in i]
    
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    
    p5 = []
    p6 = []
    p7 = []
    p8 = []
    
    for filename in sel_dir:
        print(filename)
        new_path = os.path.join(directory,filename)
        
        df = pd.read_csv(new_path,index_col=0)
        
        w1 = df.iloc[0,:]
        w2 = df.iloc[1,:]
        w3 = df.iloc[2,:]
        w4 = df.iloc[3,:]
        
        w5 = df.iloc[4,:]
        w6 = df.iloc[5,:]
        w7 = df.iloc[6,:]
        w8 = df.iloc[7,:]
        
        p1.append(w1.tolist())
        p2.append(w2.tolist())
        p3.append(w3.tolist())
        p4.append(w4.tolist())
        
        p5.append(w5.tolist())
        p6.append(w6.tolist())
        p7.append(w7.tolist())
        p8.append(w8.tolist())
    
    return p1, p2, p3, p4, p5, p6, p7, p8


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

def roCorr(ser1, ser2, window):
    le = len(ser1)
    sli = int((window-1)/2)
    
    beg_window1 = pd.Series(ser1[0:sli])
    sli_e = le-sli + 1
    end_window1 = pd.Series(ser1[sli_e:])
    a1 = pd.Series(ser1)
    win_list1 = pd.concat([end_window1,a1,beg_window1])
    
    beg_window2 = pd.Series(ser2[0:sli])
    sli_e = le-sli + 1
    end_window2 = pd.Series(ser2[sli_e:])
    a2 = pd.Series(ser2)
    win_list2 = pd.concat([end_window2,a2,beg_window2])
    
    corr_list = []
    p_list = []
    norm_i = sli
    for i in range(le):
        norm_i = norm_i + 1
        
        b = int(norm_i-(window-1)/2)
        e = int(norm_i+(window-1)/2)
        
        sec1 = win_list1[b:e]
        sec2 = win_list2[b:e]
        
        corr,pval = stats.pearsonr(sec1,sec2)
        
        corr_list.append(corr)
        p_list.append(pval)
    
    return corr_list, p_list

def consec(time_pos_pts,total_time):
    le = len(time_pos_pts)
    
    #index of time_pos_pts the times with high correlations
    ind = [0]
    for i in range(le-1):     
        p1 = time_pos_pts[i]
        p2 = time_pos_pts[i+1]
    
        diff = p2-p1
        if diff > 1:
            ind.append(i)
            ind.append(i+1)
    ind.append(le-1)
    
    print(ind)
    ind_tpp = [time_pos_pts[i] for i in ind]
    print(ind_tpp)
    
    
    sec_list = []
    flip = -1
    for j in range(len(ind)-1):
        flip = flip*(-1)
        if flip == 1:
            beg = ind[j]
            end = ind[j+1]
            
            t_beg = time_pos_pts[beg]
            t_end = time_pos_pts[end]
            
            section = total_time[t_beg:t_end]
            sec_list.append(section)
    
    return sec_list

def extMinMax(series):
    l_mima = []
    for i in series:
        mini = min(i)
        maxi = max(i)
        
        l_mima.append([mini,maxi])
    return l_mima

def extractMatrix(matrix):
    time = len(matrix[0][0])
    heatmap = []
    for t in range(time):
        print(t)
        time_matrix = []
        for i in range(8):
            row = []
            for j in range(8):
                timepoint = matrix[i][j][t]
                row.append(timepoint)
            time_matrix.append(row)
        heatmap.append(time_matrix)
    return heatmap

def corrMatrix(cell_index, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8):
    y1 = repRoLine(PC1[cell_index],10,3)
    y2 = repRoLine(PC2[cell_index],10,3)
    y3 = repRoLine(PC3[cell_index],10,3)
    y4 = repRoLine(PC4[cell_index],10,3)
    
    y5 = repRoLine(PC5[cell_index],10,3)
    y6 = repRoLine(PC6[cell_index],10,3)
    y7 = repRoLine(PC7[cell_index],10,3)
    y8 = repRoLine(PC8[cell_index],10,3)
    
    ylist = [y1,y2,y3,y4,y5,y6,y7,y8]
    
    time = np.arange(len(y1))
    
    overall = []
    for i in ylist:
        row_list = []
        for j in ylist:
            corr_list,p_list = roCorr(i,j,30)
            row_list.append(corr_list)
        overall.append(row_list)
    
    return overall
    

#Add appropropriate directories and common names
'''3T3 30um'''
com_30 = ""

dir_30_k2 = ""
dir_30_k3 = ""
dir_30_k4 = ""
dir_30_k5 = ""


'''3T3 40um'''
com_40 = ""

dir_40_k2 = ""
dir_40_k3 = ""
dir_40_k4 = ""
dir_40_k5 = ""


'''3T3 50um'''
com_50 = ""

dir_50_k2 = ""
dir_50_k3 = ""
dir_50_k4 = ""
dir_50_k5 = ""


'''3T3 60um'''
com_60 = ""

dir_60_k2 = ""
dir_60_k3 = ""
dir_60_k4 = ""
dir_60_k5 = ""





'''3T3 30um'''
w30_k2_PC1, w30_k2_PC2, w30_k2_PC3, w30_k2_PC4, w30_k2_PC5, w30_k2_PC6, w30_k2_PC7, w30_k2_PC8 = iterDir(dir_30_k2,com_30)
w30_k3_PC1, w30_k3_PC2, w30_k3_PC3, w30_k3_PC4, w30_k3_PC5, w30_k3_PC6, w30_k3_PC7, w30_k3_PC8 = iterDir(dir_30_k3,com_30)
w30_k4_PC1, w30_k4_PC2, w30_k4_PC3, w30_k4_PC4, w30_k4_PC5, w30_k4_PC6, w30_k4_PC7, w30_k4_PC8 = iterDir(dir_30_k4,com_30)
w30_k5_PC1, w30_k5_PC2, w30_k5_PC3, w30_k5_PC4, w30_k5_PC5, w30_k5_PC6, w30_k5_PC7, w30_k5_PC8 = iterDir(dir_30_k5,com_30)

'''3T3 40um'''
w40_k2_PC1, w40_k2_PC2, w40_k2_PC3, w40_k2_PC4, w40_k2_PC5, w40_k2_PC6, w40_k2_PC7, w40_k2_PC8 = iterDir(dir_40_k2,com_40)
w40_k3_PC1, w40_k3_PC2, w40_k3_PC3, w40_k3_PC4, w40_k3_PC5, w40_k3_PC6, w40_k3_PC7, w40_k3_PC8 = iterDir(dir_40_k3,com_40)
w40_k4_PC1, w40_k4_PC2, w40_k4_PC3, w40_k4_PC4, w40_k4_PC5, w40_k4_PC6, w40_k4_PC7, w40_k4_PC8 = iterDir(dir_40_k4,com_40)
w40_k5_PC1, w40_k5_PC2, w40_k5_PC3, w40_k5_PC4, w40_k5_PC5, w40_k5_PC6, w40_k5_PC7, w40_k5_PC8 = iterDir(dir_40_k5,com_40)

'''3T3 50um'''
w50_k2_PC1, w50_k2_PC2, w50_k2_PC3, w50_k2_PC4, w50_k2_PC5, w50_k2_PC6, w50_k2_PC7, w50_k2_PC8 = iterDir(dir_50_k2,com_50)
w50_k3_PC1, w50_k3_PC2, w50_k3_PC3, w50_k3_PC4, w50_k3_PC5, w50_k3_PC6, w50_k3_PC7, w50_k3_PC8 = iterDir(dir_50_k3,com_50)
w50_k4_PC1, w50_k4_PC2, w50_k4_PC3, w50_k4_PC4, w50_k4_PC5, w50_k4_PC6, w50_k4_PC7, w50_k4_PC8 = iterDir(dir_50_k4,com_50)
w50_k5_PC1, w50_k5_PC2, w50_k5_PC3, w50_k5_PC4, w50_k5_PC5, w50_k5_PC6, w50_k5_PC7, w50_k5_PC8 = iterDir(dir_50_k5,com_50)

'''3T3 60um'''
w60_k2_PC1, w60_k2_PC2, w30_k2_PC3, w30_k2_PC4, w30_k2_PC5, w30_k2_PC6, w30_k2_PC7, w30_k2_PC8 = iterDir(dir_30_k2,com_30)
w60_k3_PC1, w60_k3_PC2, w30_k3_PC3, w30_k3_PC4, w30_k3_PC5, w30_k3_PC6, w30_k3_PC7, w30_k3_PC8 = iterDir(dir_30_k3,com_30)
w60_k4_PC1, w60_k4_PC2, w30_k4_PC3, w30_k4_PC4, w30_k4_PC5, w30_k4_PC6, w30_k4_PC7, w30_k4_PC8 = iterDir(dir_30_k4,com_30)
w60_k5_PC1, w60_k5_PC2, w30_k5_PC3, w30_k5_PC4, w30_k5_PC5, w30_k5_PC6, w30_k5_PC7, w30_k5_PC8 = iterDir(dir_30_k5,com_30)


#Middle blue green
mid_bg = (140/255, 223/255, 214/255)
#Middle blue
mid_b = (109/255,192/255,213/255)
#Hookers green
hoo_g =  (90/255,113/255,106/255)
#Opera mauve
op_ma = (173/255,122/255,153/255)

#Carolina blue
ca_b = (0/255,165/255,224/255)
#Vivid sky blue
vi_s = (50/255,203/255,255/255)
#Orchid crayola
or_c =  (239/255,156/255,218/255)
#Pink lace
pi_l = (254/255,206/255,241/255)



y1 = w30_k2_PC1[0]
y2 = w30_k2_PC2[0]
time = np.arange(len(y1))

cor = stats.pearsonr(y1,y2)


yy1 = repRoLine(y1,10,3)
yy2 = repRoLine(y2,10,3)



overall = corrMatrix(0, w30_k2_PC1, w30_k2_PC2, w30_k2_PC3, w30_k2_PC4,
                     w30_k2_PC5, w30_k2_PC6, w30_k2_PC7, w30_k2_PC8)



heatmap = extractMatrix(overall)
heatmap_np = np.asarray(heatmap)

length,rows,cols = heatmap_np.shape

PC_labels = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8"]
for i in range(length):
    mat = heatmap_np[i]
    plt.imshow(mat)
    plt.xticks(np.arange(len(PC_labels)), PC_labels)
    plt.yticks(np.arange(len(PC_labels)), PC_labels)
    plt.colorbar()
    plt.title("Frame " + str(i))
    plt.savefig("Z:/General/John Ahn/Manuscript/Figure 3/scripts/Correlation matrix movie frames/frame_"+str(i)+".png")
    plt.show()
    



