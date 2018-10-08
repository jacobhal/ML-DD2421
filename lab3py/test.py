import matrixfuns as m
#import lab3 as l

# Dimensions of Matrices
rows = 7
cols = 4
depth = 5

matrixzeros = m.gen2dMatrix(0, rows, cols)
m.printMatrix(matrixzeros)

matrixones = m.gen2dMatrix(1, rows, cols)
m.printMatrix(matrixones)

matrix3dzeros = m.gen3dMatrix(0, rows, cols, depth)
m.printMatrix(matrix3dzeros)

matrix3dones = m.gen3dMatrix(1, rows, cols, depth)
m.printMatrix(matrix3dones)

matrix2d3x3 = m.gen2d3x3Matrix((1,2,3), (4,5,6), (7,8,9))
m.printMatrix(matrix2d3x3)

diagonal = m.getDiagonal(matrix2d3x3)
print(diagonal)

matrixRevert = m.getDiagonal(diagonal)
m.printMatrix(matrixRevert)

# This works
print(1.0/matrix2d3x3)

matrixSubtracted = m.broadcast(matrix2d3x3, (1,1,1), 'subtract')
m.printMatrix(matrixSubtracted)

matrixAdded = m.broadcast(matrix2d3x3, (1,1,1), 'add')
m.printMatrix(matrixAdded)

#X, labels = l.genBlobs(centers=5)
#print(m.nli(X, labels))
#print(m.nliClass(X, labels, 0))
