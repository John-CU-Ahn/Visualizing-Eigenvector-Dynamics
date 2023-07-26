# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:59:15 2023

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
        #print(filename)
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
    if len(time_pos_pts) >0:
        le = len(time_pos_pts)
        #print(time_pos_pts)
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
        
        #print(ind)
        ind_tpp = [time_pos_pts[i] for i in ind]
        #print(ind_tpp)
        
        
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
    else:
        sec_list = []
    return sec_list

def extMinMax(series):
    l_mima = []
    for i in series:
        if len(i)>0:
            mini = min(i)
            maxi = max(i)
            
            l_mima.append([mini,maxi])
    return l_mima



def corrMatrix(cell_index,window, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8):
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
            corr_list,p_list = roCorr(i,j,window)
            row_list.append(corr_list)
        overall.append(row_list)
    
    return overall

def pvalMatrix(cell_index,window, PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8):
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
            corr_list,p_list = roCorr(i,j,window)
            row_list.append(p_list)
        overall.append(row_list)
    
    return overall
    

def extractMatrix(matrix):
    time = len(matrix[0][0])
    heatmap = []
    for t in range(time):
        #print(t)
        time_matrix = []
        for i in range(8):
            row = []
            for j in range(8):
                timepoint = matrix[i][j][t]
                row.append(timepoint)
            time_matrix.append(row)
        heatmap.append(time_matrix)
    return heatmap




#y1, y2, are the scalar weight values over time 
def posNegCorr(corr_list, p_list, time, y1, y2, label_y1,label_y2,cell_label):
    yy1 = repRoLine(y1, 10, 3)
    yy2 = repRoLine(y2, 10, 3)
    
    
    ind_sig_pos = [iv1[0] for iv1,iv2 in zip(enumerate(corr_list),enumerate(p_list)) if iv1[1]>0.95 and iv2[1]<0.05]
    ind_sig_neg = [iv1[0] for iv1,iv2 in zip(enumerate(corr_list),enumerate(p_list)) if iv1[1]<-0.95 and iv2[1]<0.05]
    
    print("Positive correlation")
    print(ind_sig_pos)
    print("Negative correlation")
    print(ind_sig_neg)
    
    time_pos_pts = [time[i] for i in ind_sig_pos]
    time_neg_pts = [time[i] for i in ind_sig_neg]
    
    
    sec_pos = consec(time_pos_pts,time)
    mima_pos = extMinMax(sec_pos)

    sec_neg = consec(time_neg_pts,time)
    print(sec_neg)
    mima_neg = extMinMax(sec_neg)
    
    fig = plt.figure(figsize=(9,5))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[1, 0])
    
    ax1.plot(time,corr_list,color='tab:purple')
    ax1.set_ylabel("Correlation coefficient")
    ax1.axhline(0,color='black',linestyle='dashed')
    ax1.axhspan(0.95,1,alpha=0.3,color='green')
    ax1.axhspan(-0.95,-1,alpha=0.3,color='red')
    
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(time,p_list,color='tab:cyan')
    ax2.set_ylabel("p-value")
    ax2.axhspan(0,0.05,alpha=0.3,color='tab:cyan')
    
    ax3 = fig.add_subplot(gs[0, :])
    ax3.plot(time,yy1,color='tab:blue', label=label_y1)
    ax3.plot(time,yy2,color='tab:orange', label=label_y2)
    ax3.set_ylabel("Scalar weight")
    
    
    tot_mima_p = len(mima_pos)
    count = 0
    if len(mima_pos)>0:
        for i in mima_pos:
            count = count + 1
            if count < tot_mima_p:
                ax3.axvspan(i[0], i[1], alpha=0.3, color='green')
            else:
                ax3.axvspan(i[0], i[1], alpha=0.3, color='green',label="Positive correlation")
            
    tot_mima_n = len(mima_neg)
    count = 0
    if len(mima_neg)>0:
        for i in mima_neg:
            count = count + 1
            if count < tot_mima_n:
                ax3.axvspan(i[0], i[1], alpha=0.3, color='red')
            else:
                ax3.axvspan(i[0], i[1], alpha=0.3, color='red',label ="Negative correlation")
    
    
    ax3.legend(fontsize=5)
    #plt.savefig(cell_label + "_Positive and Negative correlations of " + label_y1 + " and " + label_y2 + " example.svg")
    plt.show()



#y1, y2, are the scalar weight values over time 
def posNegCorrFrames(corr_list, p_list, time, y1, y2, label_y1,label_y2,cell_label,directory):
    yy1 = repRoLine(y1, 10, 3)
    yy2 = repRoLine(y2, 10, 3)
    
    
    ind_sig_pos = [iv1[0] for iv1,iv2 in zip(enumerate(corr_list),enumerate(p_list)) if iv1[1]>0.95 and iv2[1]<0.05]
    ind_sig_neg = [iv1[0] for iv1,iv2 in zip(enumerate(corr_list),enumerate(p_list)) if iv1[1]<-0.95 and iv2[1]<0.05]
    
    #print("Positive correlation")
    #print(ind_sig_pos)
    #print("Negative correlation")
    #print(ind_sig_neg)
    
    time_pos_pts = [time[i] for i in ind_sig_pos]
    time_neg_pts = [time[i] for i in ind_sig_neg]
    
    
    sec_pos = consec(time_pos_pts,time)
    mima_pos = extMinMax(sec_pos)

    sec_neg = consec(time_neg_pts,time)
    #print(sec_neg)
    mima_neg = extMinMax(sec_neg)
    
    
    for vert_line in time:
        print(vert_line)
        fig = plt.figure(figsize=(9,5))
        gs = fig.add_gridspec(2,2)
        ax1 = fig.add_subplot(gs[1, 0])
        
        ax1.plot(time,corr_list,color='tab:purple')
        ax1.set_ylabel("Correlation coefficient")
        ax1.axhline(0,color='black',linestyle='dashed')
        #ax1.axhspan(0.95,1,alpha=0.3,color='green')
        #ax1.axhspan(-0.95,-1,alpha=0.3,color='red')
        ax1.axhspan(0.95,1,alpha=0.3,color='gold')
        ax1.axhspan(-0.95,-1,alpha=0.3,color='darkblue')
        ax1.axvline(vert_line,color='black',linestyle='-')
        
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(time,p_list,color='tab:cyan')
        ax2.set_ylabel("p-value")
        ax2.axhspan(0,0.05,alpha=0.3,color='tab:cyan')
        ax2.axvline(vert_line,color='black',linestyle='-')
        
        ax3 = fig.add_subplot(gs[0, :])
        ax3.plot(time,yy1,color='tab:blue', label=label_y1)
        ax3.plot(time,yy2,color='tab:orange', label=label_y2)
        ax3.set_ylabel("Scalar weight")
        
        
        tot_mima_p = len(mima_pos)
        count = 0
        if len(mima_pos)>0:
            for i in mima_pos:
                count = count + 1
                if count < tot_mima_p:
                    #ax3.axvspan(i[0], i[1], alpha=0.3, color='green')
                    ax3.axvspan(i[0], i[1], alpha=0.3, color='gold')
                else:
                    #ax3.axvspan(i[0], i[1], alpha=0.3, color='green',label="Positive correlation")
                    ax3.axvspan(i[0], i[1], alpha=0.3, color='gold',label="Positive correlation")
                
        tot_mima_n = len(mima_neg)
        count = 0
        if len(mima_neg)>0:
            for i in mima_neg:
                count = count + 1
                if count < tot_mima_n:
                    #ax3.axvspan(i[0], i[1], alpha=0.3, color='red')
                    ax3.axvspan(i[0], i[1], alpha=0.3, color='darkblue')
                else:
                    #ax3.axvspan(i[0], i[1], alpha=0.3, color='red',label ="Negative correlation")
                    ax3.axvspan(i[0], i[1], alpha=0.3, color='darkblue',label ="Negative correlation")
        ax3.axvline(vert_line,color='black',linestyle='-')
        ax3.legend(fontsize=5)
        plt.savefig(directory + "/Frame_goldblue" + str(vert_line) + "_" + cell_label + "_Positive and Negative correlations of " + label_y1 + " and " + label_y2 + " example.png")

#Add a common name between the different files
com_30 = ""
#Add directories for the different k values
dir_30_k2 = ""
dir_30_k3 = ""
dir_30_k4 = ""
dir_30_k5 = ""
#Add a save directory for the frames
save_dir = ""


w30_k2_PC1, w30_k2_PC2, w30_k2_PC3, w30_k2_PC4, w30_k2_PC5, w30_k2_PC6, w30_k2_PC7, w30_k2_PC8 = iterDir(dir_30_k2,com_30)
w30_k3_PC1, w30_k3_PC2, w30_k3_PC3, w30_k3_PC4, w30_k3_PC5, w30_k3_PC6, w30_k3_PC7, w30_k3_PC8 = iterDir(dir_30_k3,com_30)
w30_k4_PC1, w30_k4_PC2, w30_k4_PC3, w30_k4_PC4, w30_k4_PC5, w30_k4_PC6, w30_k4_PC7, w30_k4_PC8 = iterDir(dir_30_k4,com_30)
w30_k5_PC1, w30_k5_PC2, w30_k5_PC3, w30_k5_PC4, w30_k5_PC5, w30_k5_PC6, w30_k5_PC7, w30_k5_PC8 = iterDir(dir_30_k5,com_30)


window=30
overall = corrMatrix(0,window, w30_k2_PC1, w30_k2_PC2, w30_k2_PC3, w30_k2_PC4,
                     w30_k2_PC5, w30_k2_PC6, w30_k2_PC7, w30_k2_PC8)

pval = pvalMatrix(0,window, w30_k2_PC1, w30_k2_PC2, w30_k2_PC3, w30_k2_PC4,
                     w30_k2_PC5, w30_k2_PC6, w30_k2_PC7, w30_k2_PC8)

heatmap = extractMatrix(overall)
heatmap_np = np.asarray(heatmap)

heatmap_pval = extractMatrix(pval)
heatmap_pval_np = np.asarray(heatmap_pval)

length,rows,cols = heatmap_np.shape

time = np.arange(0,length,1)


PC1_PC2_pval = heatmap_pval_np[:,1,0]
PC1_PC3_pval = heatmap_pval_np[:,2,0]
PC1_PC4_pval = heatmap_pval_np[:,3,0]
PC1_PC5_pval = heatmap_pval_np[:,4,0]
PC1_PC6_pval = heatmap_pval_np[:,5,0]
PC1_PC7_pval = heatmap_pval_np[:,6,0]
PC1_PC8_pval = heatmap_pval_np[:,7,0]


#plt.legend(bbox_to_anchor=(1.2,0.5))


PC1_PC2_corr = heatmap_np[:,1,0]
PC1_PC3_corr = heatmap_np[:,2,0]
PC1_PC4_corr = heatmap_np[:,3,0]
PC1_PC5_corr = heatmap_np[:,4,0]
PC1_PC6_corr = heatmap_np[:,5,0]
PC1_PC7_corr = heatmap_np[:,6,0]
PC1_PC8_corr = heatmap_np[:,7,0]


corr_list = PC1_PC2_corr
p_list = PC1_PC2_pval
y1 = w30_k2_PC1[0]
y2 = w30_k2_PC2[0]
label_y1 = "PC1"
label_y2 = "PC2"
cell_label = "Cell0_window" + str(window)
posNegCorr(corr_list, p_list, time, y1, y2, label_y1,label_y2,cell_label)

directory = save_dir
corr_list = PC1_PC2_corr
p_list = PC1_PC2_pval
y1 = w30_k2_PC1[0]
y2 = w30_k2_PC2[0]
label_y1 = "PC1"
label_y2 = "PC2"
cell_label = "Cell0_window" + str(window)
posNegCorrFrames(corr_list, p_list, time, y1, y2, label_y1,label_y2,cell_label,directory)








