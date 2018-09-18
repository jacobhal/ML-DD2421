import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Number of training samples
N = 100
start = numpy.zeros(N)
C = None
B = [(0, C) for b in range(N)]

def zerofun(vec):
    scalar = 0
    return scalar

XC = constraint = {'type':'eq', 'fun':zerofun(B)}

print(XC)

def main():
    ret = minimize(objective, start,
    bounds = B, constraints = XC)
    alpha = ret['x']

def objective(vec):
    scalar = 0
    return scalar
