from numpy import *

def linEx(y_real, y_predicted):
    alpha = 0.1
    return mean(exp((y_real - y_predicted)*alpha) - alpha * (y_real - y_predicted) - 1)

def linExVec(y_real, y_predicted):
    alpha = 0.1
    return exp((y_real - y_predicted)*alpha) - alpha * (y_real - y_predicted) - 1