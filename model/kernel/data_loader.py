import csv
import numpy as np
import os

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

def maximum(a, n):

    # inbuilt function to find the position of minimum
    minpos = a.index(max(a))
    #print("min(a):",min(a), minpos)
    # inbuilt function to find the position of maximum
    #maxpos = a.index(max(a))

    # printing the position
    #print "The maximum is at position", maxpos + 1,max(a)
    #print("The minimum is at position", minpos, min(a))

    return maxpos

data_prefix="data/"
def process_train_data(machine,app,qnum):
#qnum: number of scheduled queues
    name= data_prefix + machine + "_train_data_" + app +".csv"
    directory=data_prefix+machine+"_classification"
    if not os.path.exists(directory):
        os.makedirs(directory)
    included_cols=[]
    for q in range(1,qnum+1):
        included_cols.append(q)
    with open(name) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        next(csvReader)
        for row in csvReader:
            myrow=list(row)
            data = np.array(list(row[i] for i in included_cols)).astype(float)
            row_new = list(np.where(data<0, 9999, data))
            myclass=minimum(row_new, len(row_new))
            class_name=directory+"/class_" +str(myclass)+ "_train_data_" + app+".csv"
            #print(class_name)
            #if myclass==0:
            #    class_name="data/class_1_train_data_tricnt.csv"
            #elif myclass==1:
            #    class_name="data/class_16_train_data_tricnt.csv"
            #elif myclass==2:
            #    class_name="data/class_32_train_data_tricnt.csv"
            #elif myclass==3:
            #    class_name="data/class_48_train_data_tricnt.csv"
            #elif myclass==4:
            #    class_name="data/class_64_train_data_tricnt.csv"
            #elif myclass==5:
            #    class_name="data/class_96_train_data_tricnt.csv"
            #elif myclass==6:
            #    class_name="data/class_128_train_data_tricnt.csv"
            with open(class_name,'a') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(row)

#name="data/new_train_tricnt.csv"
#with open(name) as csvDataFile:
#    csvReader = csv.reader(csvDataFile)
#    next(csvReader)
#    included_cols=[1,3,5,7]
#    for row in csvReader:
#        myrow=list(row)
#        data = np.array(list(row[i] for i in included_cols)).astype(float)
#        row_new = list(np.where(data<0, 9999, data))
#        myclass=minimum(row_new, len(row_new))
#        if myclass==0:
#            class_name="data/new_class_32_train_data_tricnt.csv"
#        elif myclass==1:
#            class_name="data/new_class_64_train_data_tricnt.csv"
#        elif myclass==2:
#            class_name="data/new_class_96_train_data_tricnt.csv"
#        elif myclass==3:
#            class_name="data/new_class_128_train_data_tricnt.csv"
#        with open(class_name,'a') as f:
#            writer = csv.writer(f, delimiter=',')
#            writer.writerow(row)
