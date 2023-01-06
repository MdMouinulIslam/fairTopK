import random as rnd
import  pandas as pd



def readInput(fileName):
    df = pd.read_csv(fileName)
    X_train = []
    Y_train = []
    dataDict_train = {}
    movieId_train = []
    X_test = []
    Y_test = []
    dataDict_test = {}
    movieId_test = []
    for index, row in df.iterrows():
        if index % 2 == 0:
            movieId_train.append(row['movie_id'])
            X_train.append(row['count'])
            Y_train.append(row['avg_ratings'])
            dataDict_train[int(row['movie_id'])] = (row['count'], row['avg_ratings'])
        else:
            movieId_test.append(row['movie_id'])
            X_test.append(row['count'])
            Y_test.append(row['avg_ratings'])
            dataDict_test[int(row['movie_id'])] = (row['count'], row['avg_ratings'])
    inputDict = {"train":(X_train,Y_train,dataDict_train,movieId_train),"test":(X_test,Y_test,dataDict_test,movieId_test)}

    return inputDict

def genData(n):
    X = []
    Y = []
    movieId = []
    dataDict = {}
    for i in range(0,n):
        key = rnd.randint(0,n)
        val = (key*1.0 + 100 + rnd.randint(0,0.2*n)) / (n*2)
        X.append(key)
        Y.append(val)
        movieId.append(i)
        dataDict[i] = (key,val)
    return dataDict,movieId,X,Y