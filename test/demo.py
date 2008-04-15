import numpy
import pyublas

assert pyublas.have_sparse_wrappers()

a = pyublas.zeros((5,5), flavor=pyublas.SparseBuildMatrix, dtype=float)

a[4,2] = 19

b = numpy.random.randn(2,2)
a.add_block(2, 2, b)

a_fast = pyublas.asarray(a, flavor=pyublas.SparseExecuteMatrix)

vec = numpy.random.randn(5)

res = a_fast * vec

print a_fast
print res
