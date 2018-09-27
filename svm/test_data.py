classA = numpy.concatenate ((numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
numpy.random.randn (10 , 2) ∗ 0.2 + [ −1.5 , 0 . 5 ] ) )
classB = numpy . random . randn (20 , 2) ∗ 0.2 + [ 0 . 0 , −0.5]
inputs = numpy . concatenate ( ( classA , classB ) )
t a r g e t s = numpy . concatenate (
(numpy . ones ( classA . shape [ 0 ] ) ,
−numpy . ones ( classB . shape [ 0 ] ) ) )
N = inputs . shape [ 0 ] # Number of rows ( samples )
permute=l i s t ( range (N) )
random . s h u f f l e ( permute )
inputs = inputs [ permute , : ]
t a r g e t s = t a r g e t s [ permute ]
