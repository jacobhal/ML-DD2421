import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import test_data, kernel_functions as kf

# What kernel function to use
kernel_function = kf.functions['linear']
# Number of training samples
N = 20
# Generate our test data
data = test_data.TestData(N, False)
data.generate_data()
# Initial guess of the alpha vector
start = numpy.zeros(N, dtype='float64')
upper_bound = None
# Lower and upper bounds for each value in alpha vector
B = [(0, upper_bound) for b in range(N)]
# Global variable for alpha, targets and support vectors
nonZeroAlpha = []
target_values = []
support_vectors = []


# Pre-compute the matrix P by multiplying the every combination of target values, t, and kernel K.
preComputedMatrix = numpy.empty([N,N])
for i in range(N):
    for j in range(N):
        preComputedMatrix[i][j] = data.targets[i] * data.targets[j] * kernel_function(data.inputs[i], data.inputs[j])

def zerofun(vec):
    scalar = numpy.dot(vec, data.targets)
    sumScalar = numpy.sum(scalar)
    return sumScalar

XC = constraint = {'type':'eq', 'fun':zerofun}

# Take the alpha vector and return a scalar value by implementing the expression that should be minimized.
def objective(alpha_vector):
    alpha_dot = numpy.dot(alpha_vector, alpha_vector)
    alpha_sum = numpy.sum(alpha_vector)

    completeMatrix = numpy.dot(alpha_dot, preComputedMatrix)
    completeMatrixSum = numpy.sum(completeMatrix)
    return 0.5 * completeMatrixSum - alpha_sum

def indicator(support_vector, target_value):
    # Calculate b value
    b = numpy.sum(numpy.dot(nonZeroAlpha, target_values) * [kernel_function(support_vector, x) for x in support_vectors]) - target_value
    val = numpy.sum(numpy.dot(nonZeroAlpha, target_values) * [kernel_function(support_vector, x) for x in support_vectors]) - b
    return val

def plot():
    plt.plot([p[0] for p in data.classA], [p[1] for p in data.classA], 'b. ')
    plt.plot([p[0] for p in data.classB], [p[1] for p in data.classB], 'r. ')
    plt.axis('equal') # Force same scale on both axes
    plt.savefig('svmplot.pdf') # Save a copy in a file
    plt.show() # Show the plot on the screen

def main():
    global support_vectors
    ret = minimize(objective, start, bounds = B, constraints = XC)
    success = ret['success']
    alpha = ret['x']
    if success:
        print("Found a solution\n", alpha)
        # Find all alpha values above a certain threshhold and get the
        # corresponding inputs and target values
        nonZeroAlpha = alpha[alpha > 10 ** -5]
        indices = numpy.nonzero(alpha > 10 ** -5)
        sWithCorrValues = [(alpha[x], data.inputs[x], data.targets[x]) for x in indices[0]]
        # Unzip to get our target values in a list
        _, support_vectors, target_values = zip(*sWithCorrValues)
        # Indicator function
        ind = indicator(support_vectors[0], target_values[0])
        print("Indicator:",ind)

    else:
        print("No solution found")

main()
#plot()
