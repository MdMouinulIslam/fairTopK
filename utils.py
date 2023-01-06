from genInputData import genData
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin




def createDataPairs(X_test,Y_test,movieId_test):
    counter = len(X_test)
    dataPair = []
    while (counter):
        counter = counter - 1
        p = (Y_test[counter], X_test[counter], movieId_test[counter])
        dataPair.append(p)
    dataPair.sort(reverse=True)
    return dataPair

def findBestScore(dataPair,k):
    scoreBest = 0
    for i in range(0, k):
        scoreBest = scoreBest + dataPair[i][0]
    return scoreBest


def getCutOff(scoreBest,theta):
    cutOff = scoreBest - scoreBest*theta
    return cutOff


def findEligibleCandidates(dataPair,theta,n,k):
    scoreBest = findBestScore(dataPair, k)
    scoreBest_prime = scoreBest - dataPair[k - 1][0]


    cutOff = getCutOff(scoreBest,theta)
    movieEligible = []
    for i in range(0, n):
        if scoreBest_prime + dataPair[i][0] < cutOff:
            break
        #print(dataPair[i][2])
        movieEligible.append(dataPair[i][2])
    return movieEligible

def getScore(dataDict,topk):
    s = 0
    for mid in topk:
        key,val = dataDict[mid]
        s = s + val
    return s

def getEligibleTopK(dataDict,dataPair,movieEligible, theta,k):
    scoreBest = findBestScore(dataPair, k)
    cutOff = getCutOff(scoreBest, theta)
    allTopk = combinations(movieEligible, k)
    eligibleTopk = []
    for topk in allTopk:
        score = getScore(dataDict,topk)
        if score >= cutOff:
            eligibleTopk.append(topk)
        else:
            break
    return eligibleTopk


def plot(x,y,y_pred,title):
    plt.subplot(1, 2, 1)
    plt.plot(x, y, ".")
    plt.subplot(1, 2, 2)
    plt.plot(x, y_pred)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title(title)
    plt.show()



