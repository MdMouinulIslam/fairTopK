from genInputData import genData,readInput
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import plot
from utils import createDataPairs
from utils import findEligibleCandidates
from  utils import getEligibleTopK

from  exp_1 import  getItemProbExp1
from exp_2 import  getItemProbExp2

import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin


############################## input param ################################
n = 500
theta =  0.1
k = 5
noUsers = 1000

################################ gen Dummy Data ############################

# dataDict_train,movieId_train,X_train,Y_train = genData(100)
# dataDict_test,movieId_test,X_test,Y_test = genData(100)


# plt.bar(X_train, Y_train, color ='maroon',
#         width = 0.4)
# plt.show()

inputFile = 'data/movieLens_100.csv'
X_train,Y_train,dataDict_train,movieId_train = readInput(inputFile)["train"]
X_test,Y_test,dataDict_test,movieId_test = readInput(inputFile)["test"]
################################## train on train data ####################################

model = LinearRegression()
x_train = np.array(X_train).reshape((-1, 1))
y_train = np.array(Y_train)
model.fit(x_train, y_train)
r_sq = model.score(x_train, y_train)
print(f"coefficient of determination train: {r_sq}")
y_pred_train = model.predict(x_train)



################################# test data gen ###########################################



x_test = np.array(X_test).reshape((-1, 1))
y_test = np.array(Y_test)
r_sq = model.score(x_test, y_test)
print(f"coefficient of determination test: {r_sq}")

############################### experiments ###########################################

dataPair = createDataPairs(X_test,Y_test,movieId_test)
movieEligible = findEligibleCandidates(dataPair,theta,n,k)
eligibleTopk = getEligibleTopK(dataDict_test,dataPair,movieEligible, theta,k)


############################# exp 1 ##############################################

x_exp1,y_exp1,item_probs1 = getItemProbExp1(dataDict_test,movieId_test,eligibleTopk,noUsers)

r_sq = model.score(x_exp1,y_exp1)
print(f"coefficient of determination exp 1: {r_sq}")

############################# exp 2 ##############################################

x_exp2,y_exp2,item_probs2 = getItemProbExp2(dataDict_test,movieId_test,eligibleTopk,noUsers)

r_sq = model.score(x_exp2,y_exp2 )
y_exp2_pred = model.predict(x_exp2)
print(f"coefficient of determination exp 2: {r_sq}")

################################## train datae plot ########################################





plt.subplot(1, 2, 1)
plt.plot(x_train, y_train, ".")
plt.plot(x_train, y_pred_train)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title("original")

plt.subplot(1, 2, 2)

plt.plot(x_exp2, y_exp2, ".")
plt.plot(x_exp2, y_exp2_pred)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title("uniform")

plt.show()



# plot(X_train,Y_train,y_pred_train,"train data")
# plot(x_exp2,y_exp2,y_exp2_pred,"random")