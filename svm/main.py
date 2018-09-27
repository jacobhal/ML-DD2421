import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import test_data as data

# Number of training samples
N = 100
# Initial guess of the alpha vector
start = numpy.zeros(N)
C = None
# Lower and upper bounds for each value in alpha vector
B = [(0, C) for b in range(N)]

def zerofun(vec):
    scalar = 0
    return scalar

XC = constraint = {'type':'eq', 'fun':zerofun(B)}

# List comprehension: will construct a new list a of the same length as the sequence seq.
a = [expr for x in seq]

def main():
    ret = minimize(objective(a), start, bounds = B, constraints = XC)
    alpha = ret['x']

# Find the alpha vector a that minimizes the function objective within the bounds and the constraints.
def minimize(a, s, bounds, constraints):
    return a

# Take the alpha vector and return a scalar value by implementing the expression that should be minimized.
def objective(a):
    scalar = 0
    return scalar

# Linear kernel function
def kernel_linear(v1, v2):
	return numpy.dot(v1,v2)
