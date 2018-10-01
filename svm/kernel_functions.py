import numpy
from scipy.spatial import distance

# Linear kernel function
def kernel_linear(v1, v2):
	return numpy.dot(v1,v2)

# Polynomial kernel function
def kernel_polynomial(v1, v2, degree=2):
	return (numpy.dot(v1,v2) + 1) ** degree

# Radial Basis Function
def kernel_RBF(v1, v2, smoothness=1):
	euc = distance.euclidean(v1,v2)
	return math.exp(-((euc ** 2)/(2*smoothness**2)))

functions = {"linear": kernel_linear, \
			 "polynomial": kernel_polynomial \
			 "RBF": kernel_RBF}