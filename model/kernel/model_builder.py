import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from matplotlib.ticker import LinearLocator

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model_prefix="kernel/"
def reorg_data(fname,machine,time,nrows,ncols,nnzs):
    directory=model_prefix+machine
    if not os.path.exists(directory):
        os.makedirs(directory)
    myfname=directory+"/linear_"+fname
    pkl_filename = myfname+".pkl"
    print("-- In reorg_data:",myfname,pkl_filename)
    if os.path.exists(pkl_filename):
        print("-- Model exists. Loading from file: ",pkl_filename)
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        df = pd.read_csv(myfname)
        df.columns = ['time','nnz']
        #df.columns = ['time','nnz','nrows']
        #df.columns = ['time','nnz','sparcity']
        X = df.iloc[:,df.columns != 'time']
        Y = df.iloc[:, 0]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

    else:
        if os.path.exists(myfname):
            os.remove(myfname)
        myrow=[]
        with open(myfname, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time", "nnz"])
            for i in range(len(time)):
                myrow.append(time[i])
                myrow.append(nnzs[i])
                writer.writerow(myrow)
                myrow=[]


        df = pd.read_csv(myfname)
        df.columns = ['time','nnz']
        #print(df.describe())
        X = df.iloc[:,df.columns != 'time']
        Y = df.iloc[:, 0]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
        #print("new:",coeff_df)
        with open(pkl_filename, 'wb') as pklfile:
            pickle.dump(model, pklfile)

    y_pred = model.predict(X_test)
    #print("!!!!!!! SHAPE:",X_train.shape,X_test.shape)
    rmsd = np.sqrt(mean_squared_error(Y_test, y_pred))
    r2_value = r2_score(Y_test, y_pred)

    print("----- Intercept:", model.intercept_)
    print("----- Root Mean Square Error:", rmsd)
    print("----- R^2 Value:", r2_value)
    #nnz = np.array(nnzs).reshape((-1, 1))
    ##try Polynomial
    #transformer = PolynomialFeatures(degree=1, include_bias=False)
    #transformer.fit(nnz)
    #nnz1 = transformer.transform(nnz)
    #nnz1 = PolynomialFeatures(degree=1, include_bias=False).fit_transform(nnz)
    #model = LinearRegression().fit(nnz1, time)
    #r_sq1 = model.score(nnz1, time)
    #y_pred1 = model.predict(nnz1)

    #transformer = PolynomialFeatures(degree=1, include_bias=True)
    #transformer.fit(nnz)
    #nnz2= transformer.transform(nnz)
    #nnz2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz2, time)
    #r_sq2 = model.score(nnz2, time)
    #y_pred2 = model.predict(nnz2)
    #
    #transformer = PolynomialFeatures(degree=2, include_bias=False)
    #transformer.fit(nnz)
    #nnz3= transformer.transform(nnz)
    #nnz3 = PolynomialFeatures(degree=4, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz3, time)
    #r_sq3 = model.score(nnz3, time)
    #y_pred3 = model.predict(nnz3)
    #print("r_sq3: ",r_sq3)
    #
    #transformer = PolynomialFeatures(degree=2, include_bias=True)
    #transformer.fit(nnz)
    #nnz4= transformer.transform(nnz)
    #nnz4 = PolynomialFeatures(degree=2, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz4, time)
    #r_sq4 = model.score(nnz4, time)
    #y_pred4 = model.predict(nnz4)
    return rmsd, r2_value

def reorg_data_linear2(fname,machine,time,nrows,ncols,nnzs):
    directory=model_prefix+machine
    if not os.path.exists(directory):
        os.makedirs(directory)
    myfname=directory+"/linear2_"+fname
    pkl_filename = myfname+".pkl"
    print("-- In reorg_data_linear2:",myfname,pkl_filename)
    if os.path.exists(pkl_filename):
        print("-- Model exists. Loading from file: ",pkl_filename)
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        df = pd.read_csv(myfname)
        #df.columns = ['time','nnz']
        df.columns = ['time','nnz','nrows']
        #df.columns = ['time','nnz','sparcity']
        X = df.iloc[:,df.columns != 'time']
        Y = df.iloc[:, 0]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
        #matrix_name=["333SP.mtx"]
        #for name in matrix_name:
        #    predict_filename= "test/"+machine+"/"+name+".csv"
        #    feature_set=['Cnnz_log10','Cnrows_log10']
        #    nnz = pd.read_csv(predict_filename,usecols=feature_set)
        #    #df = pd.read_csv(predict_filename)
        #    #df.columns = ["name", "maskTime","nomaskTime","Aidx","Anrows","Ancols","Annz","Bidx","Bnrows","Bncols","Bnnz","Cidx","Cnrows","Cncols","Cnnz","Cnnz_log10"]
        #    #nnz = df.iloc[:,df.columns == 'Cnnz_log10']
        #    #nnz = pd.read_csv(predict_filename,usecols=["Cnnz"],index_col=True)
        #    print("nnz:", nnz.shape)

        #    ## apply the whole pipeline to data
        #    modeled_time = model.predict(nnz)
        #    print("modeled_time:", modeled_time)
    else:
        if os.path.exists(myfname):
            os.remove(myfname)
        if os.path.exists(pkl_filename):
            os.remove(pkl_filename)
        myrow=[]
        with open(myfname, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time", "nnz","nrows"])
            #writer.writerow(["time", "nnz","nrows","ncols"])
            for i in range(len(time)):
                myrow.append(time[i])
                myrow.append(nnzs[i])
                myrow.append(nrows[i])
                #myrow.append(ncols[i])
                #myrow.append(nnzs[i]/nrows[i])
                writer.writerow(myrow)
                myrow=[]


        df = pd.read_csv(myfname)
        df.columns = ['time','nnz','nrows']
        #df.columns = ['time','nnz','nrows','ncols']
        print(df.describe())
        X = df.iloc[:,df.columns != 'time']
        Y = df.iloc[:, 0]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
        print(coeff_df)
        with open(pkl_filename, 'wb') as pklfile:
            pickle.dump(model, pklfile)

    y_pred = model.predict(X_test)
    rmsd = np.sqrt(mean_squared_error(Y_test, y_pred))
    r2_value = r2_score(Y_test, y_pred)
    #print(X_test,y_pred)
    print("----- Intercept:", model.intercept_)
    print("----- Root Mean Square Error:", rmsd)
    print("----- R^2 Value:", r2_value)
    #nnz = np.array(nnzs).reshape((-1, 1))
    ##try Polynomial
    #transformer = PolynomialFeatures(degree=1, include_bias=False)
    #transformer.fit(nnz)
    #nnz1 = transformer.transform(nnz)
    #nnz1 = PolynomialFeatures(degree=1, include_bias=False).fit_transform(nnz)
    #model = LinearRegression().fit(nnz1, time)
    #r_sq1 = model.score(nnz1, time)
    #y_pred1 = model.predict(nnz1)

    #transformer = PolynomialFeatures(degree=1, include_bias=True)
    #transformer.fit(nnz)
    #nnz2= transformer.transform(nnz)
    #nnz2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz2, time)
    #r_sq2 = model.score(nnz2, time)
    #y_pred2 = model.predict(nnz2)
    #
    #transformer = PolynomialFeatures(degree=2, include_bias=False)
    #transformer.fit(nnz)
    #nnz3= transformer.transform(nnz)
    #nnz3 = PolynomialFeatures(degree=4, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz3, time)
    #r_sq3 = model.score(nnz3, time)
    #y_pred3 = model.predict(nnz3)
    #print("r_sq3: ",r_sq3)
    #
    #transformer = PolynomialFeatures(degree=2, include_bias=True)
    #transformer.fit(nnz)
    #nnz4= transformer.transform(nnz)
    #nnz4 = PolynomialFeatures(degree=2, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz4, time)
    #r_sq4 = model.score(nnz4, time)
    #y_pred4 = model.predict(nnz4)
    return rmsd, r2_value

def reorg_data_linear3(fname,machine,time,nrows,ncols,nnzs):
    directory=model_prefix+machine
    if not os.path.exists(directory):
        os.makedirs(directory)
    myfname=directory+"/linear3_"+fname
    pkl_filename = myfname+".pkl"
    print("-- In reorg_data_linear3:",myfname,pkl_filename)
    if os.path.exists(pkl_filename):
        print("-- Model exists. Loading from file: ",pkl_filename)
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)
        df = pd.read_csv(myfname)
        #df.columns = ['time','nnz']
        #df.columns = ['time','nnz','nrows']
        df.columns = ['time','nnz','nrows','ncols']
        X = df.iloc[:,df.columns != 'time']
        Y = df.iloc[:, 0]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
    else:
        if os.path.exists(myfname):
            os.remove(myfname)
        myrow=[]
        with open(myfname, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time", "nnz","nrows","ncols"])
            for i in range(len(time)):
                myrow.append(time[i])
                myrow.append(nnzs[i])
                myrow.append(nrows[i])
                myrow.append(ncols[i])
                writer.writerow(myrow)
                myrow=[]


        df = pd.read_csv(myfname)
        df.columns = ['time','nnz','nrows','ncols']
        print(df.describe())
        X = df.iloc[:,df.columns != 'time']
        Y = df.iloc[:, 0]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)
        model = linear_model.LinearRegression()
        model.fit(X_train, Y_train)
        coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
        print(coeff_df)
        with open(pkl_filename, 'wb') as pklfile:
            pickle.dump(model, pklfile)

    y_pred = model.predict(X_test)
    rmsd = np.sqrt(mean_squared_error(Y_test, y_pred))
    r2_value = r2_score(Y_test, y_pred)
    print("----- Intercept:", model.intercept_)
    print("----- Root Mean Square Error:", rmsd)
    print("----- R^2 Value:", r2_value)
    #nnz = np.array(nnzs).reshape((-1, 1))
    ##try Polynomial
    #transformer = PolynomialFeatures(degree=1, include_bias=False)
    #transformer.fit(nnz)
    #nnz1 = transformer.transform(nnz)
    #nnz1 = PolynomialFeatures(degree=1, include_bias=False).fit_transform(nnz)
    #model = LinearRegression().fit(nnz1, time)
    #r_sq1 = model.score(nnz1, time)
    #y_pred1 = model.predict(nnz1)

    #transformer = PolynomialFeatures(degree=1, include_bias=True)
    #transformer.fit(nnz)
    #nnz2= transformer.transform(nnz)
    #nnz2 = PolynomialFeatures(degree=1, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz2, time)
    #r_sq2 = model.score(nnz2, time)
    #y_pred2 = model.predict(nnz2)
    #
    #transformer = PolynomialFeatures(degree=2, include_bias=False)
    #transformer.fit(nnz)
    #nnz3= transformer.transform(nnz)
    #nnz3 = PolynomialFeatures(degree=4, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz3, time)
    #r_sq3 = model.score(nnz3, time)
    #y_pred3 = model.predict(nnz3)
    #print("r_sq3: ",r_sq3)
    #
    #transformer = PolynomialFeatures(degree=2, include_bias=True)
    #transformer.fit(nnz)
    #nnz4= transformer.transform(nnz)
    #nnz4 = PolynomialFeatures(degree=2, include_bias=True).fit_transform(nnz)
    #model = LinearRegression().fit(nnz4, time)
    #r_sq4 = model.score(nnz4, time)
    #y_pred4 = model.predict(nnz4)
    return rmsd, r2_value