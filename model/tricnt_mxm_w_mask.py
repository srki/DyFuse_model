import csv
import math
import numpy as np
import os
import pandas as pd
import pickle
import subprocess
import heapq
import functools
import sys
import timeit

from kernel import reorg_data,reorg_data_linear2,reorg_data_linear3
from kernel import model_select

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D



#import matplotlib.cm as cm
#from sklearn.model_selection import train_test_split
#from matplotlib.ticker import MultipleLocator
#from kernel import process_train_data

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

def minimum_index(a, n):

    # inbuilt function to find the position of minimum
    minpos = a.index(min(a))
    #print("min(a):",min(a), minpos)
    # inbuilt function to find the position of maximum
    #maxpos = a.index(max(a))

    # printing the position
    #print "The maximum is at position", maxpos + 1,max(a)
    #print("The minimum is at position", minpos, min(a))

    return min(a), minpos

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

machine="ancilla"
apps="tri"
threads_num_in_q=[0,1,8,16,32,64]
time_col=[0,7,6,5,4,3]
#threads_num_in_q=[0,64,32,16,8,1]
#time_col=[0,3,4,5,6,7]
qnum=len(threads_num_in_q)
#print("****** LOADING DATA ********")
#process_train_data(machine,apps,qnum)

print("****** BUILDING MODEL ********")
folder_prefix="data/"+machine+"_classification_"+str(qnum-1)+"/"



logrow=[]
#if machine=="ancilla" and apps=="mxm":
#    threshold=[0.0, 10000, 100000,1.0e+7]
#logrow.append(threshold[1])
#logrow.append(threshold[2])
#logrow.append(threshold[3])


# 0: /home/nanding/big_matrix/thermal2.mtx,
# 1,2: readLU,8.90838,
# 3,4: createblocks,1.40262,
# 5,6,7: 1D, blocks,2,
# 8,9,10,11,12,13: nrows,614023, ncols,614023, nnz,1.7508e+06,
# 14,15,16,17,18,19: 1-128,0.000441882,S,0.0270028,G,0.0145828,
# 20,21,22,23,24,25: 128-64,1.8869e-05,S,0.0128327,G,0.0283992,
# 26,27,28,29,30,31: 64-32,1.02871e-308,S,0.0113088,G,0.0134205,
# 32,33,34,35,36,37: 32-16,1.398e-05,S,0.0125428,G,0.01354,
# 38,39,40,41,42,43: 16-8,2.435e-05,S,0.0111806,G,0.0369299,
# 44,45,46,47,48,49:  8-1,6.5458e-05,S,0.0201883,G,0.02872

time_regression=0
knn_class=1
time_predict=1
scheduler=1
plot_classes=0
#thres="1.5"
#thres="1.8"
#thres="1.2x_faster"
#thres="minarea_base64"
thres="faster_base64"
#thres="bytime"

#class files:
#nrows,ncols,nnz,64,32,16,8,1
if time_regression==1:
    for q in range (1,qnum):
        print(q,threads_num_in_q[q])
        time=[]
        realtime64=[]
        realtime=[]
        nrows=[]
        ncols=[]
        nnzs=[]
        nnzs_per_row=[]
        mythread=[]
        max_nnz=0
        min_nnz=0
        max_nrows=0
        min_nrows=0
        max_ncols=0
        min_ncols=0
        filename=folder_prefix+"1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"
        print(filename)
        model_fname="1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"
        if os.path.exists(filename):
            print("Find it!")
            with open(filename,'r') as csvfile:
                plots=csv.reader(csvfile,delimiter=",")
                for row in plots:
                    if float(row[2]) > 0.0 : # and float(row[9]) > 1.0 and float(row[11]) > 1.0:
                        #if q<=2 and float(row[13])<=float(q_nnz_max[q]) and float(row[13])>=float(q_nnz_max[q-1]) and float(row[9])>=float(q_nrows_max[q-1]) and float(row[9])<=float(q_nrows_max[q]):
                        realtime.append((float(row[time_col[q]])))
                        realtime64.append((float(row[7])))
                        time.append(math.log10(float(row[time_col[q]])))
                        #print(q,threads_num_in_q[q],"nrows:",row[0])
                        nrows.append(math.log10(float(row[0])))
                        #print(q,threads_num_in_q[q],"ncols:",row[1])
                        ncols.append(math.log10(float(row[1])))
                        #print(q,threads_num_in_q[q],"nnz:",row[2])
                        nnzs.append(math.log10(float(row[2])))
                        nnzs_per_row.append(math.log10(float(row[2])/float(row[0])))
                        mythread.append(threads_num_in_q[q])
                        #time.append((float(row[time_col[q]])))
                        #nnzs.append((float(row[13])))
                        #nrows.append((float(row[9])))
                        #ncols.append((float(row[11])))
                    #if q>2 and float(row[13])>=float(q_nnz_max[2]) and float(row[9])>=float(q_nrows_max[2]):
                    #    time.append(math.log10(float(row[time_col])))
                    #    nnzs.append(math.log10(float(row[13])))
                    #    nrows.append(math.log10(float(row[9])))
                    #    ncols.append(math.log10(float(row[11])))

                    #else:
                    #    print("nnz=0, skip.....")
        else:
            #print("No class of q,",q, threads_num_in_q[q])
            continue
        print("Q:",q,"#th:",threads_num_in_q[q],",samples:",len(time),",nnz:",min(nnzs),",", max(nnzs),"nnz_per_row", min(nnzs_per_row),max(nnzs_per_row),"nrows:",min(nrows),",", max(nrows),",ncols:",min(ncols),",",max(ncols),",sum=",sum(realtime),",64sum=",sum(realtime64) )


        rmsd1, r2_value1 = reorg_data(model_fname,machine,time,nrows,ncols,nnzs)
        rmsd2, r2_value2 = reorg_data_linear2(model_fname,machine,time,nrows,ncols,nnzs)
        rmsd3, r2_value3 = reorg_data_linear3(model_fname,machine,time,nrows,ncols,nnzs)

        if math.isnan(r2_value1):
            r2_value1=float(-1.0)
        if math.isnan(r2_value2):
            r2_value2=float(-1.0)
        if math.isnan(r2_value3):
            r2_value3=float(-1.0)
        best=max(r2_value1,r2_value2,r2_value3)
        print("class ",q," nnz range:",min(nnzs),max(nnzs),'sample:',len(nnzs),'r2 score:', r2_value1,r2_value2,r2_value3);
        print("best score is ",best)
        flag=0
        if best == r2_value1 and flag==0:
            del_model_name="kernel/"+machine+"/linear2_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"
            del_model_name1="kernel/"+machine+"/linear2_1D_thres_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"+".pkl"
            if os.path.exists(del_model_name):
                os.remove(del_model_name)
                print("deleting:",del_model_name)
            if os.path.exists(del_model_name1):
                os.remove(del_model_name1)
                print("deleting:",del_model_name1)

            del_model_name="kernel/"+machine+"/linear3_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"
            del_model_name1="kernel/"+machine+"/linear3_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"+".pkl"
            if os.path.exists(del_model_name):
                os.remove(del_model_name)
                print("deleting:",del_model_name)
            if os.path.exists(del_model_name1):
                os.remove(del_model_name1)
                print("deleting:",del_model_name1)

            logrow.append(q)
            logrow.append("linear")
            flag=1


        if best == r2_value2 and flag==0:
            del_model_name="kernel/"+machine+"/linear_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"
            del_model_name1="kernel/"+machine+"/linear_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"+".pkl"
            if os.path.exists(del_model_name):
                os.remove(del_model_name)
                print("deleting:",del_model_name)
            if os.path.exists(del_model_name1):
                os.remove(del_model_name1)
                print("deleting:",del_model_name1)

            del_model_name="kernel/"+machine+"/linear3_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"
            del_model_name1="kernel/"+machine+"/linear3_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[q])+"_train_data_"+apps+".csv"+".pkl"
            if os.path.exists(del_model_name):
                os.remove(del_model_name)
                print("deleting:",del_model_name)
            if os.path.exists(del_model_name1):
                os.remove(del_model_name1)
                print("deleting:",del_model_name1)
            flag=1

            logrow.append(q)
            logrow.append("linear2")


        if best == r2_value3 and flag==0:
            del_model_name="kernel/"+machine+"/linear_1D_thres_"+str(thres)+"_class_0_train_data_"+apps+".csv"
            del_model_name1="kernel/"+machine+"/linear_1D_thres_"+str(thres)+"_class_0_train_data_"+apps+".csv"+".pkl"
            if os.path.exists(del_model_name):
                os.remove(del_model_name)
                print("deleting:",del_model_name)
            if os.path.exists(del_model_name1):
                os.remove(del_model_name1)
                print("deleting:",del_model_name1)

            del_model_name="kernel/"+machine+"/linear2_1D_thres_"+str(thres)+"_class_0_train_data_"+apps+".csv"
            del_model_name1="kernel/"+machine+"/linear2_1D_thres_"+str(thres)+"_class_0_train_data_"+apps+".csv"+".pkl"
            if os.path.exists(del_model_name):
                os.remove(del_model_name)
                print("deleting:",del_model_name)
            if os.path.exists(del_model_name1):
                os.remove(del_model_name1)
                print("deleting:",del_model_name1)
            flag=1
            logrow.append(q)
            logrow.append("linear3")





        #rgb = np.random.rand(3,)
        #mylabel="Class: thread_num="+str(threads_num_in_q[q])+", metrix="+str(thres)+", app="+apps+", machine="+machine
        #plt.subplot(1,3,1)
        #plt.scatter(nnzs_per_row,nnzs, s=mythread , marker='.', c=[rgb],label=mylabel)
        #plt.subplot(1,3,2)
        #plt.scatter(nnzs_per_row,nrows,s=mythread , marker='.', c=[rgb],label=mylabel)
        #plt.subplot(1,3,3)
        #plt.scatter(nrows,nnzs, s=mythread , marker='.', c=[rgb],label=mylabel)
    #plt.ylabel('nnzs')
    #plt.xlabel('nrows')
    #plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    #plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    #title="Classes with metrix="+str(thres)+", app="+apps+", machine="+machine
    #plt.title(title)
    #plt.legend(loc='upper left', frameon=False)
    ##plt.show()
    #path="plots/"+title+".pdf"
    #plt.savefig(path, bbox_inches='tight')
    log_name="kernel/"+machine+"/"+apps+"_1D_log_model.csv"
    if os.path.exists(log_name):
        os.remove(log_name)
    with open(log_name,'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(logrow)
    #plt.show()


if knn_class==1:
    #thres="faster_base64"
    filename=folder_prefix+"/1D_thres_"+str(thres)+"_all_class_"+apps+".csv"

    #feature_sets=['nnzs','dimension'] #0.61
    #feature_sets=['nnzs_per_row','dimension'] #0.58
    #feature_sets=['nnzs_per_row','nrows'] #0.64
    #feature_sets=['nnzs_per_row','nnzs'] #0.62
    feature_sets=['nnzs','nrows'] #0.65
    #feature_sets=['nnzs_per_row'] #0.54
    #feature_sets=['nrows','ncols','nnzs'] #0.66
    #feature_sets=['nnzs_per_row','nrows','ncols'] #0.64
    #feature_sets=['nnzs','nnzs_per_row','nrows','ncols'] #0.64


    format_sets=["threads"]

    num_classes = len(format_sets)
    num_features = len(feature_sets)

    raw_train_data = pd.read_csv(filename, usecols=feature_sets).to_numpy()
    #print(min(raw_train_data[:,0]),max(raw_train_data[:,0]))
    #print(min(raw_train_data[:,1]),max(raw_train_data[:,1]))
    #print(min(raw_train_data[:,2]),max(raw_train_data[:,2]))


    train_labels = pd.read_csv(filename, usecols=format_sets).to_numpy()

    #raw_train_data, raw_test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=0)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(raw_train_data)


    knn = KNeighborsClassifier()
    knn.fit(train_data, train_labels.ravel())

    print(feature_sets,len(feature_sets))
    print(format_sets)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(train_data, train_labels)))

    if plot_classes==1:
        if len(feature_sets)==3:
            h = .02 # step size in the mesh
            x_min, x_max = train_data[:,0].min() - .5, train_data[:,0].max() + .5
            y_min, y_max = train_data[:,1].min() - .5, train_data[:,1].max() + .5
            x = np.arange(x_min, x_max, h) # [0.1, 5]
            y = np.arange(y_min, y_max, h)           # [6, 9]
            z_min, z_max = train_data[:,2].min() - .5, train_data[:,2].max() + .5
            z = np.arange(z_min, z_max, h)
            xx, yy,zz = np.meshgrid(x,y,z)
            res = knn.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            res = res.reshape(xx.shape)

            Z = np.outer(z.T,z)
            X, Y = np.meshgrid(x, y)

            color_dimension = X # change to desired fourth dimension
            minn, maxx = color_dimension.min(), color_dimension.max()
            norm = matplotlib.colors.Normalize(minn, maxx)
            m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
            m.set_array([])
            fcolors = m.to_rgba(color_dimension)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
            ax.set_xlabel(feature_sets[0])
            ax.set_ylabel(feature_sets[1])
            ax.set_zlabel(feature_sets[2])
            #plt.show()
            title="KNN classifier"+",app="+apps+", machine="+machine+"feature="+len(feature_sets)
            plt.title(title)
            path="plots/"+title+".pdf"
            plt.savefig(path, bbox_inches='tight')
        if len(feature_sets)==2:
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            h = .02 # step size in the mesh
            x_min, x_max = train_data[:,0].min() - .5, train_data[:,0].max() + .5
            y_min, y_max = train_data[:,1].min() - .5, train_data[:,1].max() + .5

            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            cmap = plt.cm.get_cmap("winter")
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(8, 6))
            plt.contourf(xx, yy, Z, cmap=cm.jet)

            ## Plot also the training points
            #sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y],
            #                palette=cmap_bold, alpha=1.0, edgecolor="black")
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            #plt.title("3-Class classification (k = %i, weights = '%s')"
            #          % (n_neighbors, weights))
            plt.xlabel(feature_sets[0])
            plt.ylabel(feature_sets[1])
            title="KNN classifier"+",app="+apps+", machine="+machine+"feature2"
            plt.title(title)
            path="plots/"+title+".pdf"
            plt.savefig(path, bbox_inches='tight')
    #### predict ######
    matrix_name=[]
    #head=["name", "maskTime","nomaskTime","Aidx","Anrows","Ancols","Annz","Bidx","Bnrows","Bncols","Bnnz","Cidx","nrows","ncols","nnzs","nnzs_per_row","Cnrows_log10","Cncols_log10","Cnnzs_log10"]
    predict_filename= "test/srdan_1_train_0.csv"
    head=["name", "blocks", "nrows","ncols","nnzs","t64","t32","t16","t8","t1","nnzs_per_row","Cnrows_log10","Cncols_log10","Cnnzs_log10"]
    print(predict_filename)
    flag=0
    with open(predict_filename,'r') as csvfile:
        predict_filerow=csv.reader(csvfile,delimiter=",")
        for row in predict_filerow:
            if len(row) == 10:
                if row[0] not in matrix_name:
                    matrix_name.append(row[0])
    #            if flag == 0:
                    log_name="test/"+machine+"/new_srdan_1013_"+row[0]+".csv"
                    if os.path.exists(log_name):
                        has_head=1
                    else:
                        print("Collecting info...",row[0])
                        has_head=0
                    if has_head==0:
                        with open(log_name,'a') as f:
                            writer = csv.writer(f, delimiter=',')
                            writer.writerow(head)
                    if row[4] !='':
                        with open(log_name,'a') as f:
                            writer = csv.writer(f, delimiter=',')
                            if float(row[4]) != 0.0:
                                #print("in if,",float(row[14]), float(row[12]),float(row[14])/float(row[12]))
                                row.append(float(row[4])/float(row[2]))
                                row.append(math.log10(float(row[2])))
                                row.append(math.log10(float(row[3])))
                                row.append(math.log10(float(row[4])))
                            else:
                                #print("in else,",float(row[14]), float(row[12]))
                                row.append('0.00')
                                row.append(math.log10(float(row[2])))
                                row.append(math.log10(float(row[3])))
                                row.append('0.00')
                            writer.writerow(row)
    elem=[1, 4, 16, 64, 256, 1024, 4096, 16384] #number of output blocks == #input blocks:1,2,4,8,16,32,64,128
    config_name=['1x1','1x2','1x4','1x8','1x16','1x32','1x64','1x128']
    blocks1D=[1, 4, 16, 64, 256, 1024, 4096, 16384]
    blocks2D=[] #[8,27,64]
    #matrix_name=["roadNet-CA.mtx","italy_osm.mtx","fullb.mtx","cit-Patents.mtx","road_usa.mtx","road_central.mtx","rgg_n_2_22_s0.mtx","nd12k.mtx","europe_osm.mtx","boneS10.mtx","in-2004.mtx","rgg_n_2_23_s0.mtx","kmer_U1a.mtx","Cube_Coup_dt6.mtx","Flan_1565.mtx","coPapersDBLP.mtx","vas_stokes_4M.mtx","dielFilterV3real.mtx","rgg_n_2_24_s0.mtx","kmer_P1a.mtx","kmer_V1r.mtx","Queen_4147.mtx","nlpkkt200.mtx","stokes.mtx","mip1.mtx","uk-2002.mtx","soc-LiveJournal1.mtx","wikipedia-20070206.mtx","hollywood-2009.mtx","mycielskian17.mtx","mycielskian18.mtx"]
    #matrix_name=["venturiLevel3.mtx","belgium_osm.mtx","delaunay_n24.mtx","kmer_V2a.mtx","Si41Ge41H72.mtx","ML_Geer.mtx","Bump_2911.mtx","Cube_Coup_dt0.mtx","nlpkkt160.mtx","kmer_A2a.mtx","HV15R.mtx","indochina-2004.mtx","com-Orkut.mtx"]
    for name in matrix_name:
        filename= "test/"+machine+"/new_srdan_1013_"+name+".csv"
        #print("processing...",name,filename)
        raw_test_data = pd.read_csv(filename, usecols=feature_sets).to_numpy()
        #print(raw_test_data)
        #test_labels = pd.read_csv(filename, usecols=format_sets).to_numpy()
        test_data = scaler.transform(raw_test_data)
        predict_threads = knn.predict(test_data)
        #print(len(predict_threads))
        nnzs = pd.read_csv(filename, usecols=['nnzs']).to_numpy()
        res = np.where(nnzs == 0) [0]
        #print(res)
        predict_threads[res] = 0
        #print(predict_threads)

        baseTimecol=["t64","t32","t16","t8","t1"]
        raw_baseTime= pd.read_csv(filename, usecols=baseTimecol).to_numpy()
        gbase=raw_baseTime[0][0]
        print("len:",len(raw_baseTime[:,0]))

        c=0
        best_measure=[]
        measure_time=[]

        for line in raw_baseTime:
            idx=minimum(line.tolist(),len(line.tolist()))
            if idx==0:
                best_measure.append("64")
                measure_time.append(line[idx])
            if idx==1:
                best_measure.append("32")
                measure_time.append(line[idx])
            if idx==2:
                best_measure.append("16")
                measure_time.append(line[idx])
            if idx==3:
                best_measure.append("8")
                measure_time.append(line[idx])
            if idx==4:
                best_measure.append("1")
                measure_time.append(line[idx])
            #print(c,line, len(line),minimum(line.tolist(),len(line.tolist())),line[idx])
            c=c+1
        #print(best_measure[0],len(best_measure))
        #print(measure_time[0],len(measure_time))
        final_measure_time=[]
        start=0

        confignum=len(elem)
        #print("default confignum:",confignum)
        old=0
        for i in range(0,len(elem)):
            #print("i=",i,elem[i],np.sum(elem[0:i+1]),len(raw_baseTime[:,0])+1,old)
            if len(raw_baseTime[:,0])+1 > np.sum(elem[0:i+1]):
                old=np.sum(elem[0:i+1])
                #print("continue, old=",old)
                continue
            elif len(raw_baseTime[:,0])+1 == np.sum(elem[0:i+1]):
                confignum=i+1
            elif (len(raw_baseTime[:,0]) +1 < np.sum(elem[0:i+1])) and (len(raw_baseTime[:,0])+1 > old):
                #print("old=",old,"now=",np.sum(elem[0:i+1]))
                confignum=i
            old=np.sum(elem[0:i+1])
        print("confignum:",confignum)
        for config in range(0,confignum):
            #print(name,config,"/",confignum, "output blocks=",elem[config])
            class1_time=[]
            class2_time=[]
            class3_time=[]
            class4_time=[]
            class5_time=[]
            class5_time=[]
            class1_gidx=[]
            class2_gidx=[]
            class3_gidx=[]
            class4_gidx=[]
            class5_gidx=[]
            time=0
            #print(name,"start=",start,"end=",start+elem[config])

            for bid in range(start,start+elem[config]):
                #print("3-",name,bid,measure_time[bid],best_measure[bid])
                if best_measure[bid]=="1":
                    class1_time.append(float(measure_time[bid]))
                    class1_gidx.append(bid)
                    #print("class1 bid",class1_gidx)
                    #print("class1 time",class1_time)
                if best_measure[bid]=="8":
                    class2_time.append(float(measure_time[bid]))
                    class2_gidx.append(bid)
                    #print("class2 bid",class2_gidx)
                    #print("class2 time",class2_time)
                if best_measure[bid]=="16":
                    class3_time.append(float(measure_time[bid]))
                    class3_gidx.append(bid)
                    #print("class3 bid",class3_gidx)
                    #print("class3 time",class3_time)
                if best_measure[bid]=="32":
                    class4_time.append(float(measure_time[bid]))
                    class4_gidx.append(bid)
                    #print("class4 bid",class4_gidx)
                    #print("class4 time",class4_time)
                if best_measure[bid]=="64":
                    class5_time.append(float(measure_time[bid]))
                    class5_gidx.append(bid)
                    #print("class5 bid",class5_gidx)
                    #print("class5 time",class5_time)
            if  class1_time:
                class1_blockidx=[]
                class1_blockidx,tmp_time=sublist_creator(class1_time,64,class1_gidx)
                time=time+tmp_time

            if  class2_time:
                class2_blockidx=[]
                class2_blockidx,tmp_time=sublist_creator(class2_time,8,class2_gidx)
                time=time+tmp_time

            if  class3_time:
                class3_blockidx=[]
                class3_blockidx,tmp_time=sublist_creator(class3_time,4,class3_gidx)
                time=time+tmp_time

            if  class4_time:
                class4_blockidx=[]
                class4_blockidx,tmp_time=sublist_creator(class4_time,2,class4_gidx)
                time=time+tmp_time

            if  class5_time:
                class5_blockidx=[]
                class5_blockidx,tmp_time=sublist_creator(class5_time,1,class5_gidx)
                time=time+tmp_time
            start=start+elem[config]
            final_measure_time.append(float(time))
        best_measurement_res=minimum(final_measure_time,len(final_measure_time))
        print(name,"Best config:",config_name[best_measurement_res],"time:", final_measure_time[best_measurement_res])

        if time_predict==1:
            starttime = timeit.default_timer()
            model=[]
            classes=[]
            model,classes=model_select(machine,apps,qnum)
            #print("out",model)
            #print("out",classes)
            loadmodel={}
            for m in range(1,qnum):
                #print(m)
                pkl_filename = "kernel/"+machine+"/"+model[m-1]+"_1D_thres_"+str(thres)+"_class_"+str(threads_num_in_q[m])+"_train_data_"+apps+".csv.pkl"
                #print(pkl_filename)
                if os.path.exists(pkl_filename):
                    #print("-- Model exists. Loading from file: ",pkl_filename)
                    with open(pkl_filename, 'rb') as file:
                        loadmodel[m] = pickle.load(file)
                else:
                    print("!!!! Model doesn't exists !!!!", pkl_filename)
                    sys.exit(0)
            predict_feature_set1=['Cnnzs_log10']
            predict_feature_set2=['Cnnzs_log10','Cnrows_log10']
            predict_feature_set3=['Cnnzs_log10','Cnrows_log10','Cncols_log10']
            modeled_time={}
            if 'linear' in model:
                indexes = [i for i, j in enumerate(model) if j == 'linear']
                #print(f"linear found at index {indexes}")
                nnz1 = pd.read_csv(filename,usecols=predict_feature_set1)
                for my_indexes in indexes:
                    #print("linear...",my_indexes,modeled_time[my_indexes])
                    modeled_time[my_indexes]= loadmodel[my_indexes+1].predict(nnz1)

            if 'linear2' in model:
                indexes = [i for i, j in enumerate(model) if j == 'linear2']
                #print(f"linear2 found at index {indexes}")
                nnz2 = pd.read_csv(filename,usecols=predict_feature_set2)
                for my_indexes in indexes:
                    modeled_time[my_indexes]= loadmodel[my_indexes+1].predict(nnz2)
            if 'linear3' in model:
                indexes = [i for i, j in enumerate(model) if j == 'linear3']
                #print(f"linear3 found at index {indexes}")
                nnz3 = pd.read_csv(filename,usecols=predict_feature_set3)
                for my_indexes in indexes:
                    modeled_time[my_indexes]= loadmodel[my_indexes+1].predict(nnz3)

            i=0
            final_model_time=[]
            for x in predict_threads:
                #print('threads=',x)
                if x == 0:
                    final_model_time.append(float(0.0))
                    myclass = 'na'
                else:
                    myclass = np.where(threads_num_in_q == x) [0]
                    #print("In time predict, x=",x,",myclass=",myclass[0],',i=',i)
                    final_model_time.append(float(math.pow(10,modeled_time[myclass[0]-1][i])))
                i=i+1

        print("Time prediction:", timeit.default_timer() - starttime)
        if scheduler==1:
            timeit.default_timer
            num_config=len(blocks1D)+len(blocks2D)
            buf=sum(blocks1D)+sum(blocks2D)
            print(num_config,buf)
            apps_time=[]
            apps_time.append(final_model_time[0])
            start=1
            class1_blockidx=[]
            class2_blockidx=[]
            class3_blockidx=[]
            class4_blockidx=[]
            class5_blockidx=[]
            measure_time=[]
            cur_best_time=999
            for i in range(1,confignum):
                #print("Config:",i,config_name[i])
                #print("---start:",start,", end:",start+elem[i])
                class1_time=[]
                class2_time=[]
                class3_time=[]
                class4_time=[]
                class5_time=[]
                class1_gidx=[]
                class2_gidx=[]
                class3_gidx=[]
                class4_gidx=[]
                class5_gidx=[]
                time=0
                g_idx=0
                for each_block_time in range(start,start+elem[i]):
                    if predict_threads[each_block_time]==1:
                        class1_time.append(float(final_model_time[each_block_time]))
                        class1_gidx.append(g_idx)
                    if predict_threads[each_block_time]==8:
                        class2_time.append(float(final_model_time[each_block_time]))
                        class2_gidx.append(g_idx)
                    if predict_threads[each_block_time]==16:
                        class3_time.append(float(final_model_time[each_block_time]))
                        class3_gidx.append(g_idx)
                    if predict_threads[each_block_time]==32:
                        class4_time.append(float(final_model_time[each_block_time]))
                        class4_gidx.append(g_idx)
                    if predict_threads[each_block_time]==64:
                        class5_time.append(float(final_model_time[each_block_time]))
                        class5_gidx.append(g_idx)
                    g_idx=g_idx+1


                if  class5_time:
                    class5_blockidx=[]
                    class5_blockidx,tmp_time=sublist_creator(class5_time,1,class5_gidx)
                    time=time+tmp_time
                if cur_best_time < time:
                    time=999
                    continue

                if  class4_time:
                    class4_blockidx=[]
                    class4_blockidx,tmp_time=sublist_creator(class4_time,2,class4_gidx)
                    time=time+tmp_time
                if cur_best_time < time:
                    time=999
                    continue

                if  class3_time:
                    class3_blockidx=[]
                    class3_blockidx,tmp_time=sublist_creator(class3_time,4,class3_gidx)
                    time=time+tmp_time
                if cur_best_time < time:
                    time=999
                    continue

                if  class2_time:
                    class2_blockidx=[]
                    class2_blockidx,tmp_time=sublist_creator(class2_time,8,class2_gidx)
                    time=time+tmp_time
                if cur_best_time < time:
                    time=999
                    continue

                if  class1_time:
                    class1_blockidx=[]
                    class1_blockidx,tmp_time=sublist_creator(class1_time,64,class1_gidx)
                    time=time+tmp_time
                if cur_best_time < time:
                    time=999
                    continue

                start=start+elem[i]
                apps_time.append(float(time))
                cur_best_time=time

            best_config=minimum(apps_time,len(apps_time))
            print(name,"Best config:",config_name[best_config],"time:", apps_time[best_config])
            print("Time scheduler: ", timeit.default_timer() - starttime)

            with open('res/res_srdan_tri_1013_summary_1','a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow([name,"Model",config_name[best_config],"Mtime", final_measure_time[best_measurement_res],"gbase,", gbase])

            res_file="res/"+machine+"/new_"+apps+"_"+name+"_1013_1.csv"
            res_file1="res/"+machine+"/new_"+apps+"_block_"+name+"_1013_1.csv"
            if os.path.exists(res_file):
                os.remove(res_file)
            if os.path.exists(res_file1):
                os.remove(res_file1)

            if best_config != 0:
                start=1
                for i in range(1,confignum):
                    if i == best_config:
                        class1_time=[]
                        class2_time=[]
                        class3_time=[]
                        class4_time=[]
                        class5_time=[]
                        class1_gidx=[]
                        class2_gidx=[]
                        class3_gidx=[]
                        class4_gidx=[]
                        class5_gidx=[]
                        time=0
                        g_idx=0
                        for each_block_time in range(start,start+elem[i]):
                            if predict_threads[each_block_time]==1:
                                class1_time.append(float(final_model_time[each_block_time]))
                                class1_gidx.append(g_idx)
                            if predict_threads[each_block_time]==8:
                                class2_time.append(float(final_model_time[each_block_time]))
                                class2_gidx.append(g_idx)
                            if predict_threads[each_block_time]==16:
                                class3_time.append(float(final_model_time[each_block_time]))
                                class3_gidx.append(g_idx)
                            if predict_threads[each_block_time]==32:
                                class4_time.append(float(final_model_time[each_block_time]))
                                class4_gidx.append(g_idx)
                            if predict_threads[each_block_time]==64:
                                class5_time.append(float(final_model_time[each_block_time]))
                                class5_gidx.append(g_idx)
                            g_idx=g_idx+1
                        if not class1_time:
                            print("class1 is empty")
                        else:
                            class1_blockidx=[]
                            class1_blockidx,tmp_time=sublist_creator(class1_time,64,class1_gidx)
                            time=time+tmp_time

                        if not class2_time:
                            print("class2 is empty")
                        else:
                            class2_blockidx=[]
                            class2_blockidx,tmp_time=sublist_creator(class2_time,8,class2_gidx)
                            time=time+tmp_time

                        if not class3_time:
                            print("class3 is empty")
                        else:
                            class3_blockidx=[]
                            class3_blockidx,tmp_time=sublist_creator(class3_time,4,class3_gidx)
                            time=time+tmp_time

                        if not class4_time:
                            print("class4 is empty")
                        else:
                            class4_blockidx=[]
                            class4_blockidx,tmp_time=sublist_creator(class4_time,2,class4_gidx)
                            time=time+tmp_time

                        if not class5_time:
                            print("class5 is empty")
                        else:
                            class5_blockidx=[]
                            class5_blockidx,tmp_time=sublist_creator(class5_time,1,class5_gidx)
                            time=time+tmp_time
                    start=start+elem[i]

            tmprow=[]
            with open(res_file1,'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                tmprow.append(config_name[best_config])
                writer.writerow(tmprow)

            tmprow=[]
            with open(res_file,'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                #tmprow.append(config_name[best_config])
                #writer.writerow(tmprow)
                if best_config == 0:
                    tmprow=[]
                    tmprow.append(0)
                    tmprow.append(predict_threads[best_config])
                    writer.writerow(tmprow)

                if best_config != 0:
                    if len(class5_time) !=0:
                        #print(class5_gidx)
                        for x in class5_gidx:
                            tmprow=[]
                            tmprow.append(x)
                            tmprow.append("64")
                            writer.writerow(tmprow)
                        writer.writerow([99999,99999])

                    if len(class4_time) !=0:
                        #print(class5_gidx)
                        for x in class4_gidx:
                            tmprow=[]
                            tmprow.append(x)
                            tmprow.append("32")
                            writer.writerow(tmprow)
                        writer.writerow([99999,99999])

                    if len(class3_time) !=0:
                        #print(class3_gidx)
                        for x in class3_gidx:
                            tmprow=[]
                            tmprow.append(x)
                            tmprow.append("16")
                            writer.writerow(tmprow)
                        writer.writerow([99999,99999])
                        #r_class3_blockidx=[]
                        #r_class3_blockidx = [x for x in class3_blockidx if x != []]
                        #r_class3_blockidx=np.array(r_class3_blockidx)
                        #writer.writerows(r_class3_blockidx)

                    if len(class2_time) !=0:
                        #print(class3_gidx)
                        for x in class2_gidx:
                            tmprow=[]
                            tmprow.append(x)
                            tmprow.append("8")
                            writer.writerow(tmprow)
                        writer.writerow([99999,99999])

                    if len(class1_time) !=0:
                        #print(class3_gidx)
                        for x in class1_gidx:
                            tmprow=[]
                            tmprow.append(x)
                            tmprow.append("1")
                            writer.writerow(tmprow)
                        writer.writerow([99999,99999])
            subprocess.check_call(['./fixfile.sh', res_file])
            subprocess.check_call(['./fixfile.sh', res_file1])








