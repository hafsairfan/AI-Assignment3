# -*- coding: utf-8 -*-
"""
CS 351 - Artificial Intelligence
Assignment 3, Question 2

Student 1(Name and ID): Hafsa Irfan - hi05946
Student 2(Name and ID): Aliza Saleem Lakhani - al05435

"""

import numpy as np
import matplotlib.pyplot as plt

"""This function takes actual and predicted ratings and compute total mean square error(mse) in observed ratings.
"""


def computeError(R, predR):
    """Your code to calculate MSE goes here"""
    error = 0
    m, n = R.shape
    for i in range(m):
        for j in range(n):
            if (R[i][j]):
                error += pow(R[i][j] - predR[i][j], 2)
    return error/np.sum(R > 0)


"""
This function takes P (m*k) and Q(k*n) matrices along with user bias (U) and item bias (I) and returns predicted rating.
where m = No of Users, n = No of items
"""


def getPredictedRatings(P, Q, U, I):
    """Your code to predict ratinngs goes here"""
    m, k = P.shape
    k, n = Q.shape
    predR = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            predR[i][j] = np.dot(P[i, :], Q[:, j]) + U[i] + I[j]

    return predR


"""This fucntion runs gradient descent to minimze error in ratings by adjusting P, Q, U and I matrices based on gradients.
   The functions returns a list of (iter,mse) tuple that lists mse in each iteration
"""


def runGradientDescent(R, P, Q, U, I, iterations, alpha):
    """Your gradient descent code goes here"""
    stats = []
    m, k = P.shape
    k, n = Q.shape

    for iterr in range(iterations):
        for i in range(m):
            for j in range(n):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j]) - U[i] - I[j]
                    for k_ in range(k):
                        P[i][k_] = P[i][k_] + alpha * (2 * eij * Q[k_][j])
                        Q[k_][j] = Q[k_][j] + alpha * (2 * eij * P[i][k_])
                    U[i] = U[i] + alpha * (2 * eij)
                    I[j] = I[j] + alpha * (2 * eij)

        predR = getPredictedRatings(P, Q, U, I)
        mse = computeError(R, predR)
        stats.append((iterr, mse))
    """"finally returns (iter,mse) values in a list"""
    return stats


"""
This method applies matrix factorization to predict unobserved values in a rating matrix (R) using gradient descent.
K is number of latent variables and alpha is the learning rate to be used in gradient decent
"""


def matrixFactorization(R, k, iterations, alpha):
    """Your code to initialize P, Q, U and I matrices goes here. P and Q will be randomly initialized whereas U and I will be initialized as zeros.
    Be careful about the dimension of these matrices
    """
    m, n = R.shape
    P = np.random.rand(m, k) + 1
    Q = np.random.rand(k, n) + 1
    U = np.zeros(m)
    I = np.zeros(n)

    # Run gradient descent to minimize error
    stats = runGradientDescent(R, P, Q, U, I, iterations, alpha)

    print('P matrix:')
    print(P)
    print('Q matrix:')
    print(Q)
    print("User bias:")
    print(U)
    print("Item bias:")
    print(I)
    print("P x Q:")
    print(getPredictedRatings(P, Q, U, I))
    print(np.round_(getPredictedRatings(P, Q, U, I)))
    plotGraph(stats)


def plotGraph(stats):
    i = [i for i, e in stats]
    e = [e for i, e in stats]
    plt.plot(i, e)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Square Error")
    plt.show()


""""
User Item rating matrix given ratings of 5 users for 6 items.
Note: If you want, you can change the underlying data structure and can work with starndard python lists instead of np arrays
We may test with different matrices with varying dimensions and number of latent factors. Make sure your code works fine in those cases.
"""
R = np.array([
    [5, 3, 0, 1, 4, 5],
    [1, 0, 2, 0, 0, 0],
    [3, 1, 0, 5, 1, 3],
    [2, 0, 0, 0, 2, 0],
    [0, 1, 5, 2, 0, 0],
])

k = 3
alpha = 0.01
iterations = 500

matrixFactorization(R, k, iterations, alpha)
