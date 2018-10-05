import numpy as np

def printMatrix(mat):
    print("Printing matrix...\n", mat)

def gen2dMatrix(val, rows, cols):
    if(val):
        return np.ones((rows, cols)) # 2D Matrix of ones
    else:
        return np.zeros((rows, cols)) # 2D Matrix of zeros

def gen3dMatrix(val, rows, cols, depth):
    if(val):
        return np.ones((depth, rows, cols))  # 3D Matrix of ones
    else:
        return np.zeros((depth, rows, cols))  # 3D Matrix of zeros

def gen2d3x3Matrix(r1, r2, r3):
    return np.array([r1, r2, r3])

# Get diagonal as row vector
def getDiagonal(mat):
    return np.diag(mat)

# Numpy reshaping
def vecToMat(vec):
    return np.reshape(vec.reshape(-1, 1))

def vecToMatReversed(vec):
    return np.reshape(vec.reshape(1, -1))

def matToVec(mat):
    return np.reshape(mat.reshape(-1))

# If operation='add' add rowVector to matrix, if operation='subtract' subtract rowVector from matrix
def broadcast(mat, rowVector, operation):
    if(operation == 'add'):
        return mat + rowVector
    elif(operation == 'subtract'):
        return mat - rowVector

# TODO!!!!!
def nli(self, X, y):
    res = None
    classes = np.unique(y) # Get the unique examples
    # Iterate over both index and value
    for jdx,c in enumerate(classes):
        idx = y==c # Returns a true or false with the length of y
        # Or more compactly extract the indices for which y==class is true,
        # analogous to MATLAB’s find
        idx = np.where(y==c)[0]
        xlc = X[idx,:] # Get the x for the class labels. Vectors are rows.
        res += xlc
    return res
