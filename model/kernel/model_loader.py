import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle


model_prefix="kernel/"
#def model_select(machine,apps,qnum,name,num_config,elem,time,nnzs,nrows,ncols,threads_num_in_q):
def model_select(machine,apps,qnum):
    log_model=model_prefix+"/"+machine+"/"+apps+"_1D_log_model.csv"
    #print("In model loader, log_model name=",log_model)
    model=[]
    classes=[]
    with open(log_model,'r') as csvfile:
        plots=csv.reader(csvfile,delimiter=",")
        for row in plots:
            #print(row)
            for q in range (1,qnum):
                classes.append(row[(q-1)*2])
                model.append(row[(q-1)*2+1])
    #print(model)
    #print(classes)

    return model,classes


    # apply the whole pipeline to data
    #modeled_time = pd.Series(pipe.predict(pr[pred_cols]))
    #print pred

    ##rmsd = np.sqrt(mean_squared_error(Y_test, y_pred))
    ##r2_value = r2_score(Y_test, y_pred)