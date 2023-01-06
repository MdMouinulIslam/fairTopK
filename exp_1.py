from genInputData import genData
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from itertools import combinations
from fairness import heuristic_leximin,leximin







def getItemProbExp1(dataDict,movieId,eligibleTopk,noUsers):
    nonzero_prob, _ = heuristic_leximin(eligibleTopk)
    item_probs = {}
    for mid in movieId:
        item_probs[mid] = 0
    for key,val in nonzero_prob.items():
        for mid in key:
            item_probs[mid] = item_probs[mid] + val
    total = sum(item_probs.values())
    for key,val in item_probs.items():
        item_probs[key] = val/total
    for mid in movieId:
        item_probs[mid] = item_probs[mid]*noUsers
    X = []
    Y = []
    for mid, (x, y) in dataDict.items():
        if mid in item_probs:
            x_new = x + item_probs[mid]
        else:
            x_new = x
        X.append(x_new)
        Y.append(y)
    x = np.array(X).reshape((-1, 1))
    y = np.array(Y)
    return x,y,item_probs

