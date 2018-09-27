import numpy

# Linear kernel function
def kernel_linear(v1, v2):
	return numpy.dot(v1,v2)