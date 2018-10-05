import numpy as np

class Matrix:
    def __init__(self, rows, cols, depth):
        # Dimensions of Matrix
        self.rows = rows
        self.cols = cols
        self.depth = depth

    def getMatrix(self):
        return self.matrix

    def setMatrix(self, m):
        self.matrix = m

    def printMatrix(self):
        print("Printing matrix...\n", self.matrix)

    def gen2dMatrix(self, val):
        if(val):
            self.matrix = np.ones((self.rows, self.cols)) # 2D Matrix of ones
        else:
            self.matrix = np.zeros((self.rows, self.cols)) # 2D Matrix of zeros

    def gen3dMatrix(self, val):
        if(val):
            self.matrix = np.ones((self.depth, self.rows, self.cols))  # 3D Matrix of ones
        else:
            self.matrix = np.zeros((self.depth, self.rows, self.cols))  # 3D Matrix of zeros

    def gen2d3x3Matrix(self, r1, r2, r3):
        self.matrix = np.array([r1, r2, r3])

    # Get diagonal as row vector
    def getDiagonal(self):
        return np.diag(self.matrix)

    # Turn a row vector into a diagonal matrix
    def setDiagonalMatrix(self, rowVector):
        self.matrix = np.diag(rowVector)

    # Numpy reshaping
    def vecToMat(self):
        self.matrix = np.reshape(self.matrix.reshape(-1, 1))

    def vecToMatReversed(self):
        self.matrix = np.reshape(self.matrix.reshape(1, -1))

    def matToVec(self):
        self.matrix = np.reshape(self.matrix.reshape(-1))

    # If sign>=1 add rowVector to matrix, if sign<=0 subtract rowVector from matrix
    def broadcast(self, rowVector, operation):
        if(operation == 'add'):
            self.matrix = self.matrix + rowVector
        elif(operation == 'subtract'):
            self.matrix = self.matrix - rowVector

    # TODO!!!!!
    def nli(self, y):
        res = None
        classes = np.unique(y) # Get the unique examples
        # Iterate over both index and value
        for jdx,c in enumerate(classes):
            idx = y==c # Returns a true or false with the length of y
            # Or more compactly extract the indices for which y==class is true,
            # analogous to MATLAB’s find
            idx = np.where(y==c)[0]
            xlc = self.matrix[idx,:] # Get the x for the class labels. Vectors are rows.
            res += xlc
        return res
