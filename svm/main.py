import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import test_data, kernel_functions

# Number of training samples
N = 10
# Initial guess of the alpha vector
start = numpy.zeros(N)
C = None
# Lower and upper bounds for each value in alpha vector
B = [(0, C) for b in range(N)]

# Pre-compute the matrix P by multiplying the every combination of target values, t, and kernel K.  
preComputedMatrix = numpy.empty([N,N])
for i in range(N):
	for j in range(N):
		preComputedMatrix[i][j] = test_data.targets[i] * test_data.targets[j] * kernel_functions.kernel_linear(test_data.inputs[i], test_data.inputs[j])

def zerofun(vec):
    scalar = 0
    return scalar

XC = constraint = {'type':'eq', 'fun':zerofun(B)}

# List comprehension: will construct a new list a of the same length as the sequence seq.
# a = [expr for x in seq]

def main():
    ret = minimize(objective(a), start, bounds = B, constraints = XC)
    alpha = ret['x']

# Find the alpha vector a that minimizes the function objective within the bounds and the constraints.
def minimize(a, s, bounds, constraints):
    return a

# Take the alpha vector and return a scalar value by implementing the expression that should be minimized.
def objective(alpha_vector):
    alpha_dot = numpy.dot(alpha_vector, alpha_vector)
    alpha_sum = numpy.sum(alpha_vector)

    completeMatrix = numpy.dot(alpha_dot, preComputedMatrix)
    completeMatrixSum = numpy.sum(completeMatrix)
    return 0.5 * completeMatrixSum - alpha_sum

