import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import test_data, kernel_functions as kf

# Number of training samples
N = 40
# Generate our test data
data = test_data.TestData(N, False)
data.generate_data()
# Initial guess of the alpha vector
start = numpy.zeros(N, dtype='float64')
upper_bound = None
# Lower and upper bounds for each value in alpha vector
B = [(0, upper_bound) for b in range(N)]

# Pre-compute the matrix P by multiplying the every combination of target values, t, and kernel K.  
preComputedMatrix = numpy.empty([N,N])
for i in range(N):
	for j in range(N):
		preComputedMatrix[i][j] = data.targets[i] * data.targets[j] * kf.kernel_linear(data.inputs[i], data.inputs[j])

def zerofun(vec):
    scalar = numpy.dot(vec, data.targets)
    return scalar

XC = constraint = {'type':'eq', 'fun':zerofun}

# List comprehension: will construct a new list a of the same length as the sequence seq.
# a = [expr for x in seq]

# Take the alpha vector and return a scalar value by implementing the expression that should be minimized.
def objective(alpha_vector):
    alpha_dot = numpy.dot(alpha_vector, alpha_vector)
    alpha_sum = numpy.sum(alpha_vector)

    completeMatrix = numpy.dot(alpha_dot, preComputedMatrix)
    completeMatrixSum = numpy.sum(completeMatrix)
    return 0.5 * completeMatrixSum - alpha_sum

def indicator(support_vector, alpha_vector, b):
	val = numpy.sum( \
		numpy.dot(alpha_vector, data.targets) * \
		kf.functions['linear'](support_vector, data.inputs)) - b
	return val 

def main():
    ret = minimize(objective, start, bounds = B, constraints = XC)
    success = ret['success']
    alpha = ret['x']
    if success:
    	print("Found a solution\n", alpha)
    	# Find all alpha values above a certain threshhold and get the
    	# corresponding inputs and target values
    	s = alpha[alpha > 10 ** -5]
    	indices = numpy.nonzero(alpha > 10 ** -5)
    	sWithCorrValues = [(alpha[x], data.inputs[x], data.targets[x]) for x in indices[0]]
    	# Unzip to get our target values in a list
    	_, _, t = zip(*sWithCorrValues)
    	# Calculate b value
    	b = numpy.sum(numpy.dot(alpha, data.targets) * kf.kernel_linear(s, data.inputs)) - numpy.sum(t) 
    	# Indicator function
    	print("Indicator:",indicator(s, alpha, b))
    else:
    	print("No solution found")

main()