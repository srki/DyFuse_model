import csv
#import pandas as pd
import numpy as np
#import os
#import heapq
#import functools
#import subprocess


def minimum(a, n):

    # inbuilt function to find the position of minimum
    minpos = a.index(min(a))
    #print("min(a):",min(a), minpos)
    # inbuilt function to find the position of maximum
    #maxpos = a.index(max(a))

    # printing the position
    #print "The maximum is at position", maxpos + 1,max(a)
    #print("The minimum is at position", minpos, min(a))

    return min(a), minpos+1


def maximum(a, n):

    # inbuilt function to find the position of minimum
    maxpos = a.index(max(a))
    #print("min(a):",min(a), minpos)
    # inbuilt function to find the position of maximum
    #maxpos = a.index(max(a))

    # printing the position
    #print "The maximum is at position", maxpos + 1,max(a)
    #print("The minimum is at position", minpos, min(a))

    return max(a), maxpos+1


bytime=1
app="spgemm_mask_comp" #tricnt; spgemm_mask_comp
thres="faster_base64"
if bytime == 1:
    if app=="spgemm_mask":
        name="data/train_raw_data_spgemm_maskA/spgemm_maskA_train_merge.csv"
    #name="data/train_raw_data_mxm/tmp"
    #thres="minarea"
    #name="data/train_raw_data_tri_mask/fortest.csv"
    if app=="tricnt":
        name="data/train_raw_data_tri_mask/merge_model_train_tri_mask.csv"

    if app=="spgemm_mask_comp":
        name="data/train_raw_data_spgemm_maskA_comp/spgemm_maskA_comp_train_merge.csv"


with open(name) as csvDataFile:
        csvReader = csv.reader(csvDataFile,delimiter=",")
        #next(csvReader) # if have header
        # all data
        #included_cols=[2,4,9,11,13,15,21,27,33,39,45,17,23,29,35,41,47,19,25,31,37,43,49] #start from 0
        # mxm timing
        if app=="spgemm_mask" or app=="spgemm_mask_comp":
            included_cols=[14,17,20,23,26] #start from 0
            included_cols_feature=[4,5,6,14,17,20,23,26] #start from 0
        if app=="tricnt":
            included_cols=[13,15,17,19,21] #start from 0
            included_cols_feature=[7,9,11,13,15,17,19,21] #start from 0
        threads_num=[64,32,16,8,1]
        for row in csvReader:
            myrow=list(row)
            data = np.array(list(row[i] for i in included_cols)).astype(float)
            featureanddata = np.array(list(row[i] for i in included_cols_feature)).astype(float)
            #print(featureanddata[2])
            if float(featureanddata[2] != 0.0):
                #print(data)
                #base=data[0]
                #new_area=np.delete(data, 0)
                #print(new_area)
                #area = np.flip((np.multiply(data,threads_num)),0)
                #area = np.multiply(data,threads_num)
                if thres=="faster_base64":
                    area=data[0]/data
                    #print(area)
                    #print(threads_num)
                    #normalize_area=base/new_area
                    #print(normalize_area)
                    max_area, myclass=maximum(list(area), len(list(area)))
                    #min_area, myclass=minimum(list(area), len(list(area)))
                    #print(max_area,myclass,",threads=",threads_num[myclass-1])
                if thres=="minarea":
                    print(data)
                    concurrentNo=np.divide(64,threads_num)
                    area=np.multiply(data,concurrentNo)
                    print(concurrentNo)
                    print(area)

                #if max_area < 1:
                #if min_area > flipdata[myclass]*(base/1.25):
                #    myclass = 0 ## 64 threads by default
                    #print(max_area,myclass,"adjust to 5: 64 threads")

                class_name="data/ancilla_classification_"+str(len(threads_num))+"/1D_thres_"+str(thres)+"_all_class_"+app+".csv"
                #class_name="data/ancilla_classification_"+str(len(threads_num))+"/1D_thres_"+str(thres)+"_class_"+str(threads_num[myclass-1])+"_train_data_"+app+".csv"

                featureanddata=np.append(featureanddata,threads_num[myclass-1])
                with open(class_name,'a') as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerow(featureanddata)

