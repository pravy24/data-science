# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 11:14:05 2017

@author: Z003EY2A
"""
import numpy as npy
import matplotlib.pyplot as plot


""" Cost Function """
def computeCost(X, y, theta, lamda):
    
    cost = 0
    m = len(y)
    
    #errors = (X.dot(theta) - y ) **2
    #cost = (1/(2*m)) * sum(errors)
    
    # Cost function can also be written as:-
    # (1/2m) * T(X * theta) * (X * theta)
    cost = (1/(2*m)) * npy.transpose(X.dot(theta) - y).dot(X.dot(theta) - y) + lamda/(2*m) * sum(theta * theta)
    return (cost[0])



""" Gradient Descent algorithm """
def gradientDescent(X, y, theta, alpha, lamda, iterations):

    m = len(y)
    n = len(X[0]) #Get the number of features including ...
    cost_hist = npy.zeros((iterations, 1))
    theta_t = theta

    i = 0
    while (i < iterations):

        cost_hist[i] = computeCost(X, y, theta, lamda)

        j = 0
        while j < n:
            theta_t[j] = theta_t[j] - (alpha/m) * npy.transpose(X.dot(theta) - y).dot(X[0:m,j]) + (lamda / m) * j * theta[j]
            j = j+1

        theta = theta_t
        i = i + 1

    return ((theta, cost_hist))



def predict(X, theta):
    
    return (X.dot(theta))

""" The MAIN """
if __name__ == "__main__":
    
    data = npy.loadtxt("D:\IC019134\Learnings\Data Science\Coursera\Machine Learning - Andrew Ng\Excersise\Ex1\ex1data1.csv", delimiter=',')
    
    m = len(data)
    X = data[0:m, 0:1]
    y = data[0:m, 1:2]
    
    plot.figure(1)
    plot.subplot(211)
    
    plot.scatter(X, y, color='blue')
    
    """numpy.ones creates an array of 1.0"""
    """numpy.concatenate merges two arrays"""
    """axis=1 for concatenate merges the arrays vertically"""
    X = npy.concatenate((npy.ones((97,1)), X), axis=1)
    
    theta = npy.zeros((2, 1))
    lamda = 1000
    
    cost = computeCost (X, y, theta, lamda)

    alpha = .01
    iterations = 1500
    
    (theta, cost_hist) = gradientDescent(X, y, theta, alpha, lamda, iterations)
    
    plot.plot(X[0:m, 1], X.dot(theta), color='red')

    plot.subplot(212)
    plot.plot(range(0, iterations), cost_hist, color='green')
    
    p = predict(npy.array([1, 3.5]), theta)
    
    print(p)