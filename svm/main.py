import numpy, random, math, sys
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import test_data, kernel_functions as kf

kfun = 'linear'
opt = 2
slack = None
plotId = 0
print(sys.argv)
for i, arg in enumerate(sys.argv):
    if sys.argv[i] == '-kf':
        kfun = sys.argv[i+1]
        opt = int(sys.argv[i+2])
    if sys.argv[i] == '-slack':
        slack = float(sys.argv[i+1])
    if sys.argv[i] == '-pid':
        plotId = int(sys.argv[i+1])
    
# What kernel function to use
kernel_function = lambda x,y : kf.functions[kfun](x, y, opt)
# Number of training samples
N = 200
# Generate our test data
data = test_data.TestData(N, True)
data.generate_data()
# Initial guess of the alpha vector
start = numpy.zeros(N, dtype='float64')
upper_bound = slack
# Lower and upper bounds for each value in alpha vector
B = [(0, upper_bound) for b in range(N)]
# Global variable for alpha, targets and support vectors
nonZeroAlpha = []
target_values = []
support_vectors = []
bValue = 0

# Pre-compute the matrix P by multiplying the every combination of target values, t, and kernel K.
preComputedMatrix = numpy.empty([N,N])
for i in range(N):
    for j in range(N):
        preComputedMatrix[i][j] = data.targets[i] * data.targets[j] * kernel_function(data.inputs[i], data.inputs[j])

def printInputData():
    print("Input data")
    for p in data.classA:
        print(f"({p[0]}, {p[1]})")
    for p in data.classB:
        print(f"({p[0]}, {p[1]})")
    print("")

def zerofun(vec):
    scalar = numpy.dot(vec, data.targets)
    return scalar

XC = constraint = {'type':'eq', 'fun':zerofun}

# Take the alpha vector and return a scalar value by implementing the expression that should be minimized.
def objective(alpha_vector):
    alpha_sum = numpy.sum(alpha_vector)

    matmul = numpy.dot(alpha_vector, preComputedMatrix)
    vecmul = numpy.dot(alpha_vector, matmul)
    return 0.5 * vecmul - alpha_sum

def indicator(point):
    kernelMat = [kernel_function(point, x) for x in support_vectors]
    #val = numpy.sum(numpy.dot(kernelMat, numpy.dot(target_values, nonZeroAlpha))) - bValue
    val = 0
    for i,p in enumerate(kernelMat):
        val += nonZeroAlpha[i] * target_values[i] * kernelMat[i]
    #print(f"{point} has indicator value {val}")
    return val - bValue

def plot():
    plt.clf()
    # Plot input data
    plt.plot([p[0] for p in data.classA], [p[1] for p in data.classA], 'b. ')
    plt.plot([p[0] for p in data.classB], [p[1] for p in data.classB], 'r. ')
    plt.axis('equal') # Force same scale on both axes
    
    # Plot decision boundary
    xgrid = numpy.linspace(-5, 5)
    ygrid = numpy.linspace(-4, 4)

    grid = numpy.array([[indicator([x,y]) for x in xgrid] for y in ygrid])
    #print(grid)
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

    # Plot support vector points in green
    for i, point in enumerate(support_vectors):
        ind = indicator(point)
        plt.plot(point[0], point[1], 'k.')

    path = 'C:/Users/Adrian/Pictures/ML-DD2421/svm/slack C 1/svmplot{:d}.png'.format(plotId)
    plt.savefig(path) # Save a copy in a file
    #plt.show() # Show the plot on the screen

def main():
    #printInputData()
    global nonZeroAlpha, support_vectors, target_values, bValue
    ret = minimize(objective, start, bounds = B, constraints = XC)
    success = ret['success']
    alpha = ret['x']
    if success:
        #print("Alpha vector\n", alpha)
        # Find all alpha values above a certain threshhold and get the
        # corresponding inputs and target values
        nonZeroAlpha = alpha[alpha > 10 ** -5]
        indices = numpy.nonzero(alpha > 10 ** -5)
        sWithCorrValues = [(alpha[x], data.inputs[x], data.targets[x]) for x in indices[0]]
        # Unzip to get our target values in a list
        _, support_vectors, target_values = zip(*sWithCorrValues)

        # Calculate b value
        tmp = [kernel_function(support_vectors[0], x) for x in support_vectors]
        #bValue = numpy.sum(numpy.dot(numpy.dot(nonZeroAlpha, target_values), tmp)) - target_values[0]
        for i,p in enumerate(tmp):
            bValue += nonZeroAlpha[i] * target_values[i] * tmp[i]
        bValue -= target_values[0]    
        #print(bValue)

        #print("\nSupport vectors")
        # Indicator function
        """
        for i, point in enumerate(support_vectors):
            ind = indicator(point)
            print(f"({point[0]}, {point[1]}) classified as {ind}")
            print(f"alpha = {nonZeroAlpha[i]}")
            print(f"target = {target_values[i]}")
        """
        plot()
        #plotDecisionBoundary()
    else:
        print("No solution found")

main()
