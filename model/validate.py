import csv
from csv import reader
import math
import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import heapq
import functools
import subprocess

from kernel import process_train_data
from kernel import reorg_data,reorg_data_nonlinear,reorg_data_nonlinear3
from kernel import model_select

machine="ancilla"
apps="mxm"
qnum=4
threads_num_in_q=[1,16,32,64]
#1D: 1 2 4 8
#2D: 2 3 4
blocks1D=[1,4,16,64]
blocks2D=[8,27,64]
num_config=len(blocks1D)+len(blocks2D)
buf=sum(blocks1D)+sum(blocks2D)
print(num_config,buf)
config_name=['1x1','1x2','1x4','1x8','2x2','3x3','4x4']
elem=[1,4,16,64,8,27,64]
###########################################################
# load models
###########################################################
def sublist_creator(lst, n,idx):
    lists = [[] for _ in range(n)]
    lists_idx = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    c=0
    for value in lst:
        total, index = heapq.heappop(totals)
        lists[index].append(value)
        lists_idx[index].append(idx[c])
        heapq.heappush(totals, (total + value, index))
        c=c+1
    res = functools.reduce(lambda i, j: i if sum(i) > sum(j) else j, lists)
    return lists_idx, sum(res)

def format_func(value, tick_number):
    N = value
    if N==0:
        return 0
    else:
        return r'10^{}'.format(N)

def minimum(a, n):

    # inbuilt function to find the position of minimum
    minpos = a.index(min(a))
    #print("min(a):",min(a), minpos)
    # inbuilt function to find the position of maximum
    #maxpos = a.index(max(a))

    # printing the position
    #print "The maximum is at position", maxpos + 1,max(a)
    #print("The minimum is at position", minpos, min(a))

    return minpos

def updatefile(updatedlist,fname):
    with open(fname,"a",newline="") as f:
        Writer=csv.writer(f)
        Writer.writerows(updatedlist)
        print("File, ",fname,",has been updated")
#testidex=[]
#test=[4,1,1,2,1]
#gidx=[0,2,3,6,8]
#testidex,testtotals=sublist_creator(test,3,gidx)
#print("TEST heap idx:",testidex)
#print("TEST heap total:",testtotals)
#quit()


threshold=[]
model=[]
classes=[]
threshold,model,classes=model_select(machine,apps,qnum)
print("out",threshold)
print("out",model)
print("out",classes)
loadmodel={}
for m in range(1,qnum+1):
    print(m)
    pkl_filename = "kernel/"+machine+"/"+model[m-1]+"_class_"+str(m)+"_train_data_"+apps+".csv.pkl"
    print(pkl_filename)
    if os.path.exists(pkl_filename):
        print("-- Model exists. Loading from file: ",pkl_filename)
        with open(pkl_filename, 'rb') as file:
            loadmodel[m] = pickle.load(file)


#print(time)
#print(time[0])
#print(time[1])
flag=1
has_head=0
head=["name", "maskTime","nomaskTime","Aidx","Anrows","Ancols","Annz","Bidx","Bnrows","Bncols","Bnnz","Cidx","Cnrows","Cncols","Cnnz","Cnnz_log10","Cnrows_log10","Cncols_log10"]
matrix_name=[]
input_filename= "test/big_"+apps+".csv"
with open(input_filename,'r') as csvfile:
    plots=csv.reader(csvfile,delimiter=",")
    for row in plots:
        if len(row) == 15:
            if row[0] not in matrix_name:
                matrix_name.append(row[0])
            if flag == 0:
                log_name="test/"+machine+"/"+row[0]+".csv"
                if os.path.exists(log_name):
                    has_head=1
                else:
                    has_head=0
                if has_head==0:
                    with open(log_name,'a') as f:
                        writer = csv.writer(f, delimiter=',')
                        writer.writerow(head)
                if row[14] !='':
                    with open(log_name,'a') as f:
                        writer = csv.writer(f, delimiter=',')
                        #print(len(row),row)
                        if float(row[14]) > 0.0:
                            row.append(math.log10(float(row[14])))
                            row.append(math.log10(float(row[12])))
                            row.append(math.log10(float(row[13])))
                        else:
                            row.append("-1")
                            row.append("-1")
                            row.append("-1")
                        writer.writerow(row)

#print(matrix_name)
#matrix_name=['BenElechi1.mtx']
matrix_name=['HV15R.mtx']
for name in matrix_name:
    print("Name:",name)
    modeled_time={}
    predict_filename= "test/"+machine+"/"+name+".csv"
    feature_set1=['Cnnz_log10',]
    feature_set2=['Cnnz_log10','Cnrows_log10']
    feature_set3=['Cnnz_log10','Cnrows_log10','Cncols_log10']

    if 'linear' in model:
        indexes = [i for i, j in enumerate(model) if j == 'linear']
        print(f"linear found at index {indexes}")
        nnz1 = pd.read_csv(predict_filename,usecols=feature_set1)
        for my_indexes in indexes:
            modeled_time[my_indexes]= loadmodel[my_indexes+1].predict(nnz1)
            #print("linear...",my_indexes,modeled_time[my_indexes])

    if 'nonlinear' in model:
        indexes = [i for i, j in enumerate(model) if j == 'nonlinear']
        print(f"nonlinear found at index {indexes}")
        nnz2 = pd.read_csv(predict_filename,usecols=feature_set2)
        for my_indexes in indexes:
            modeled_time[my_indexes]= loadmodel[my_indexes+1].predict(nnz2)
            #print("nonlinear...",my_indexes,nnz2,modeled_time[my_indexes])
    if 'nonlinear3' in model:
        indexes = [i for i, j in enumerate(model) if j == 'nonlinear3']
        print(f"nonlinear3 found at index {indexes}")
        nnz3 = pd.read_csv(predict_filename,usecols=feature_set3)
        for my_indexes in indexes:
            modeled_time[my_indexes]= loadmodel[my_indexes+1].predict(nnz3)
            #print("nonlinear3...",my_indexes,modeled_time[my_indexes])
    #print("modeled_time:",modeled_time)
    #print("modeled_time[0][1]:",modeled_time[0][1],modeled_time[0][0])
    myrow=[]
    mytime_mask=[]
    mytime_nomask=[]
    iszero=float(0.0)
    with open(predict_filename,'r') as csvfile:
        csv_reader = reader(csvfile)
        header = next(csv_reader)
        plots=csv.reader(csvfile,delimiter=",")
        for row in plots:
            if len(row) == 18:
                myrow.append(float(row[15]))
                mytime_mask.append(float(row[1]))
                if not row[2]:
                    #print("There's no NomaskTime record")
                    mytime_nomask.append(float(iszero))
                else:
                    mytime_nomask.append(float(row[2]))
    #print(myrow)
    final_model_time=[]
    plot_final_model_time=[]
    final_threads=[]
    plot_realmasktime=[]
    plot_realnomasktime=[]

    for i in range(len(myrow)):
        #print(myrow[i],i,len(myrow))
        if myrow[i] == -1.0:
            plot_final_model_time.append(float(iszero))
            final_model_time.append(float(iszero))
            final_threads.append(float(iszero))
            mytime_mask[i]=float(0.0)
            mytime_nomask[i]=float(0.0)
            plot_realmasktime.append(float(mytime_mask[i]))
            plot_realnomasktime.append(float(mytime_nomask[i]))
        elif myrow[i] <= float(threshold[1]):
            myclass=1
            final_model_time.append(float(math.pow(10,modeled_time[myclass-1][i])))
            plot_final_model_time.append(float(modeled_time[myclass-1][i]))
            final_threads.append(float(threads_num_in_q[myclass-1]))
            plot_realmasktime.append(float(mytime_mask[i]))
            plot_realnomasktime.append(float(mytime_nomask[i]))
            #final_model_time[i]=modeled_time[myclass-1][i]
            #final_threads[i]=threads_num_in_q[myclass-1]
        elif myrow[i] <= float(threshold[2]):
            myclass=2
            final_model_time.append(float(math.pow(10,modeled_time[myclass-1][i])))
            plot_final_model_time.append(float(modeled_time[myclass-1][i]))
            final_threads.append(float(threads_num_in_q[myclass-1]))
            plot_realmasktime.append(float(mytime_mask[i]))
            plot_realnomasktime.append(float(mytime_nomask[i]))
            #final_model_time[i]=modeled_time[myclass-1][i]
            #final_threads[i]=threads_num_in_q[myclass-1]
        elif myrow[i] <= float(threshold[2]):
            myclass=3
            final_model_time.append(float(math.pow(10,modeled_time[myclass-1][i])))
            plot_final_model_time.append(float(modeled_time[myclass-1][i]))
            final_threads.append(float(threads_num_in_q[myclass-1]))
            plot_realmasktime.append(float(mytime_mask[i]))
            plot_realnomasktime.append(float(mytime_nomask[i]))
        elif myrow[i] > float(threshold[3]):
            myclass=4
            final_model_time.append(float(math.pow(10,modeled_time[myclass-1][i])))
            plot_final_model_time.append(float(modeled_time[myclass-1][i]))
            final_threads.append(float(threads_num_in_q[myclass-1]))
            plot_realmasktime.append(float(mytime_mask[i]))
            plot_realnomasktime.append(float(mytime_nomask[i]))
            #final_model_time[i]=modeled_time[myclass-1][i]
            #final_threads[i]=threads_num_in_q[myclass-1]

        #print("HaHa...",len(final_model_time),len(plot_realmasktime))

    apps_time=[]
    real_masktime=[]
    real_nomasktime=[]
    #elem=[1,4,16,64,8,27,64]
    apps_time.append(final_model_time[0])
    real_masktime.append(plot_realmasktime[0])
    real_nomasktime.append(plot_realnomasktime[0])
    start=1
    class1_blockidx=[]
    class2_blockidx=[]
    class3_blockidx=[]
    class4_blockidx=[]
    adjust_num_config=num_config
    if len(final_threads) < 184:
        adjust_num_config=adjust_num_config-1

    for i in range(1,adjust_num_config):
        print("Config:",i)
        print("---start:",start,", end:",start+elem[i])

        class1_real_nomasktime=[]
        class2_real_nomasktime=[]
        class3_real_nomasktime=[]
        class4_real_nomasktime=[]


        class1_real_masktime=[]
        class2_real_masktime=[]
        class3_real_masktime=[]
        class4_real_masktime=[]

        class1_time=[]
        class2_time=[]
        class3_time=[]
        class4_time=[]
        class1_gidx=[]
        class2_gidx=[]
        class3_gidx=[]
        class4_gidx=[]
        time=0
        mymasktime=0
        mynomasktime=0

        g_idx=0
        for each_block_time in range(start,start+elem[i]):
            print("------each_block_time:",each_block_time,len(plot_realmasktime))
            if final_threads[each_block_time]==1:
                class1_time.append(float(final_model_time[each_block_time]))
                class1_gidx.append(g_idx)
                class1_real_masktime.append(float(plot_realmasktime[each_block_time]))
                class1_real_nomasktime.append(float(plot_realnomasktime[each_block_time]))
            if final_threads[each_block_time]==16:
                class2_time.append(float(final_model_time[each_block_time]))
                class2_gidx.append(g_idx)
                class2_real_masktime.append(float(plot_realmasktime[each_block_time]))
                class2_real_nomasktime.append(float(plot_norealmasktime[each_block_time]))
            if final_threads[each_block_time]==32:
                class3_time.append(float(final_model_time[each_block_time]))
                class3_gidx.append(g_idx)
                class3_real_masktime.append(float(plot_realmasktime[each_block_time]))
                class3_real_nomasktime.append(float(plot_norealmasktime[each_block_time]))
            if final_threads[each_block_time]==64:
                class4_time.append(float(final_model_time[each_block_time]))
                class4_gidx.append(g_idx)
                class4_real_masktime.append(float(plot_realmasktime[each_block_time]))
                class4_real_nomasktime.append(float(plot_norealmasktime[each_block_time]))
            g_idx=g_idx+1
        if not class1_time:
            print("class1 is empty")
        else:
            class1_blockidx,tmp_mymasktime=sublist_creator(class1_real_masktime,128,class1_gidx)
            class1_blockidx,tmp_mynomasktime=sublist_creator(class1_real_nomasktime,128,class1_gidx)
            class1_blockidx=[]
            class1_blockidx,tmp_time=sublist_creator(class1_time,128,class1_gidx)
            time=time+tmp_time
            mymasktime=mymasktime+tmp_mymasktime
            mynomasktime=mynomasktime+tmp_mynomasktime

        if not class2_time:
            print("class2 is empty")
        else:
            class2_blockidx,tmp_mymasktime=sublist_creator(class2_real_masktime,8,class2_gidx)
            class2_blockidx,tmp_mynomasktime=sublist_creator(class2_real_nomasktime,8,class2_gidx)
            class2_blockidx=[]
            class2_blockidx,tmp_time=sublist_creator(class2_time,8,class2_gidx)
            time=time+tmp_time
            mymasktime=mymasktime+tmp_mymasktime
            mynomasktime=mynomasktime+tmp_mynomasktime

        if not class3_time:
            print("class3 is empty")
        else:
            class3_blockidx,tmp_mymasktime=sublist_creator(class3_real_masktime,4,class3_gidx)
            class3_blockidx,tmp_mynomasktime=sublist_creator(class3_real_nomasktime,4,class3_gidx)
            class3_blockidx=[]
            class3_blockidx,tmp_time=sublist_creator(class3_time,4,class3_gidx)
            time=time+tmp_time
            mymasktime=mymasktime+tmp_mymasktime
            mynomasktime=mynomasktime+tmp_mynomasktime

        if not class4_time:
            print("class4 is empty")
        else:
            class4_blockidx,tmp_mymasktime=sublist_creator(class4_real_masktime,2,class4_gidx)
            class4_blockidx,tmp_mynomasktime=sublist_creator(class4_real_nomasktime,2,class4_gidx)
            class4_blockidx=[]
            class4_blockidx,tmp_time=sublist_creator(class4_time,2,class4_gidx)
            time=time+tmp_time
            mymasktime=mymasktime+tmp_mymasktime
            mynomasktime=mynomasktime+tmp_mynomasktime

        start=start+elem[i]
        apps_time.append(float(time))
        real_masktime.append(float(mymasktime))
        real_nomasktime.append(float(mynomasktime))
    best_config=minimum(apps_time,len(apps_time))
    best_real_mask=minimum(real_masktime,len(real_masktime))
    best_real_nomask=minimum(real_nomasktime,len(real_nomasktime))
    print("Best reak mask:",config_name[best_real_mask],"time:", real_masktime[best_real_mask])
    print("Best config:",config_name[best_config],"time:", real_masktime[best_config],real_masktime[best_real_mask])
    print("Best reak nomask:",config_name[best_real_nomask],"time:", real_nomasktime[best_real_nomask])
    tmprow=[]
    tmprow.append(str(name))
    tmprow.append(float(math.pow(10,myrow[0])))

    tmprow.append(str(config_name[best_real_mask]))
    tmprow.append(float(real_masktime[best_real_mask]))

    tmprow.append(str(config_name[best_config]))
    tmprow.append(float(real_masktime[best_config]))

    res_file="tempory_validate_result_using_graphblas_individual_blocks.csv"

    with open(res_file,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(tmprow)





    #plt.show()


#if time_1:
#    my_time_1, schedule_idx_1=sublist_creator(time_1, 16)
#    print(schedule_idx_1)
#    q1.append([schedule_idx_1])
#    res = functools.reduce(lambda i, j: i if sum(i) > sum(j) else j, my_time_1)
#    total_time+=sum(res)
#    print ("Maximum sum sublist is : " + str(res),sum(res))
