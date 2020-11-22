import numpy as np
from numpy.random import choice
from leastSquares import leastSquares
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare


def adaboostUSPS(X, Y, K, nSamples, percent):
    # Adaboost with least squares linear classifier as weak classifier on USPS data
    # for a high dimensional dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (scalar)
    # nSamples  : number of data points obtained by weighted sampling (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (1 x k) 
    # para      : parameters of simple classifier (K x (D+1))            
    #             For a D-dim dataset each simple classifier has D+1 parameters
    # error     : training error (1 x k)


    #Used the least-squares based AdaBoost on real data, i.e. the USPS data (provided in usps.mat). The dataset consists of a matrix X and a label vector Y. Each 	row of the matrix X is an image of size 20 × 14. The first 5000 rows of X contain the images of the digit 2, and the rest contains the images of the digit 	9. Perform a random split of the 10000 data points into two equally sized subsets, one for training and one for validation. Run this at least three times and 		plot the cross validation error estimates vs. the number k of iterations.


    # Sample random a percentage of data as test data set
    N = len(X)
    numb = round( N * percent)
    pos = choice(N, numb, False)
    allpos = range(N)
    restpos = np.setdiff1d(allpos, pos)

    testX = X[pos]
    testY = Y[pos]
    newX = X[restpos]
    newY = Y[restpos]
    X = newX
    Y = newY

    # Initialization
    n = N - numb

    w = (np.ones(n)/n).reshape(n, 1)
    alphaK = np.ones(K)
    error = np.ones(K)
    para = np.ndarray((K, X.shape[1]+1))

    #initialize loop
    for k in range(K):
        
        # weight sampling of data
        #print(w.shape)
        index = choice(n,nSamples,True,w.ravel())
        X_sampled = X[index]
        Y_sampled = Y[index]

       # Train the weak classifier Ck
        weights, bias = leastSquares(X_sampled, Y_sampled)

        para[k,:] = np.append(weights, [bias])

        # classify
        cY = np.sign(np.append(np.ones(n).reshape(n,1), X, axis = 1).dot(para[k].T)).T

        # calculate error for given classifier
        temp = np.where([Y[i] != cY[i] for i in range(n)], 1, 0).reshape(n, 1)
        ek = np.sum(w * temp)

        # If the error is zero, the data set is correct classified - break the loop
        if ek < 1.0e-01:
            alphaK[k]=1
            break

        # Compute the voting weight for the weak classifier alphak
        alphaK[k] = 0.5 * np.log((1 - ek) / ek)

        # recalculate the weights
        w = w * np.exp(-alphaK[k] * (Y*(cY.reshape(len(cY),1)))) 
        # check mult
        w = w/sum(w)

        # calculate error for boosted classifier
        classlabels, _ = eval_adaBoost_leastSquare(testX, alphaK[:k+1], para[:k+1])
        classlabels = classlabels.reshape(len(classlabels), 1)

        error[k] = sum([classlabels[i] != testY[i] for i in range(len(classlabels))])/len(testY)

    return [alphaK, para, error]
