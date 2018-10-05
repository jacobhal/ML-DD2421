import matrix as m

matrix = m.Matrix(7, 4, 5)

matrix.gen2dMatrix(0)
matrix.printMatrix()

matrix.gen2dMatrix(1)
matrix.printMatrix()

matrix.gen3dMatrix(0)
matrix.printMatrix()

matrix.gen3dMatrix(1)
matrix.printMatrix()

matrix.gen2d3x3Matrix((1,2,3), (4,5,6), (7,8,9))
matrix.printMatrix()

diagonal = matrix.getDiagonal()
print(diagonal)

matrix.setDiagonalMatrix(diagonal)
matrix.printMatrix()

matrix.gen2d3x3Matrix((1,2,3), (4,5,6), (7,8,9))
print(1.0/matrix.getMatrix())

matrix.broadcast((1,1,1), 'subtract')
matrix.printMatrix()

matrix.broadcast((1,1,1), 'add')
matrix.printMatrix()
