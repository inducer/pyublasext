import sys, random, math
import pyublas
import pyublasext
import numpy
import numpy.linalg as la
import unittest
import test_data as tmd
from testtools import *




class TestMatrices(unittest.TestCase):
    def for_all_dtypes(self, f):
        f(numpy.float)
        f(numpy.complex)

    def assert_small(self, matrix, thresh=1e-10):
        self.assert_(numpy.linalg.norm(matrix) < thresh)

    def assert_zero(self, matrix):
        if len(matrix.shape) == 2:
          for i in matrix:
              for j in i:
                  self.assert_(j == 0)
        else:
            for j in matrix:
                self.assert_(j == 0)

    def do_test_add_scattered(self, dtype):
        a = pyublas.zeros((10,10), dtype, flavor=pyublas.SparseBuildMatrix)
        vec = numpy.array([3., 5.], dtype=dtype)
        vec2 = numpy.array([2., 4.], dtype=dtype)
        b = numpy.outer(vec, vec2)
        a.add_scattered([5,7], [1,3], pyublas.why_not(b, matrix=True, dtype=dtype))

    def test_add_scattered(self):
        if not pyublas.has_sparse_wrappers():
            return

        self.for_all_dtypes(self.do_test_add_scattered)

    def do_test_umfpack(self, dtype):
        if not pyublas.has_sparse_wrappers():
            return
        if not pyublasext.has_umfpack():
            return

        size = 100
        #A = make_random_matrix(size, dtype, numpy.SparseExecuteMatrix)
        #b = make_random_vector(size, dtype)
        A = tmd.umf_a[dtype]
        b = tmd.umf_b[dtype]

        umf_op = pyublasext.UMFPACKOperator.make(A)
        x = numpy.zeros((size,), dtype)

        umf_op.apply(b, x)

        self.assert_(la.norm(b - A * x) < 1e-10)

    def test_umfpack(self):
        self.for_all_dtypes(self.do_test_umfpack)

    def do_test_arpack_classic(self, dtype):
        if not pyublasext.has_arpack():
            return

        size = 10
        #A = make_random_matrix(size, dtype)
        A = tmd.aclassmat[dtype]
        Aop = pyublasext.MatrixOperator.make(A)

        results = pyublasext.operator_eigenvectors(Aop, 3)

        for value,vector in results:
            err = numpy.dot(A, vector) - value*vector
            self.assert_(la.norm(err) < 1e-7)

    def test_arpack_classic(self):
        self.for_all_dtypes(self.do_test_arpack_classic)

    def do_test_arpack_generalized(self, dtype):
        if not pyublasext.has_arpack():
            return

        size = 100
        A = tmd.agen_a[dtype]
        Aop = pyublasext.MatrixOperator.make(A)

        M = tmd.agen_m[dtype]
        Mop = pyublasext.MatrixOperator.make(M)

        Minvop = pyublasext.LUInverseOperator.make(M)

        results = pyublasext.operator_eigenvectors(Minvop*Aop, 5, Mop)

        for value,vector in results:
            if A.dtype == float:
                A_vec = A*vector.real.copy() + 1j*(A*vector.imag.copy())
            else:
                A_vec = A*vector
            err = A_vec - value * numpy.dot(M, vector)
            self.assert_(la.norm(err) < 1e-7)

    def test_arpack_generalized(self):
        if not pyublas.has_sparse_wrappers():
            return
        self.for_all_dtypes(self.do_test_arpack_generalized)

    def do_test_arpack_shift_invert(self, dtype):
        if not pyublasext.has_arpack():
            return
        size = 100
        sigma = 1

        #A = make_random_matrix(size, dtype)
        #M = make_random_spd_matrix(size, dtype)
        A = tmd.arpsi_a[dtype]
        M = tmd.arpsi_m[dtype]
        Mop = pyublasext.MatrixOperator.make(M)

        shifted_mat_invop = pyublasext.LUInverseOperator.make(A - sigma * M)

        results = pyublasext.operator_eigenvectors(
            shifted_mat_invop * Mop, 5, Mop, spectral_shift=sigma)

        for value,vector in results:
            self.assert_(la.norm(numpy.dot(A,vector) - value*numpy.dot(M, vector)) < 1e-10)

    def test_arpack_shift_invert(self):
        self.for_all_dtypes(self.do_test_arpack_shift_invert)

    def test_lu(self):
        size = 50
        A = 10*numpy.identity(size, float) + numpy.random.randn(size, size)

        inv_op = pyublasext.LUInverseOperator.make(A)
        for count in range(20):
            x = numpy.random.randn(size)
            self.assert_small(inv_op(numpy.dot(A, x)) - x)

    def do_test_sparse(self, dtype):
        def count_elements(mat):
            count = 0
            for i in mat.indices():
                count += 1
            return count

        size = 100
        A1 = make_random_matrix(size, dtype, pyublas.SparseBuildMatrix)
        A2 = pyublas.asarray(A1, dtype, pyublas.SparseExecuteMatrix)
        self.assert_(count_elements(A1) == count_elements(A2))

    def test_sparse(self):
        if not pyublas.has_sparse_wrappers():
            return
        self.for_all_dtypes(self.do_test_sparse)

    def do_test_bicgstab(self, dtype):
        # real case fails sometimes
        size = 30

        #A = make_random_full_matrix(size, dtype)
        #b = make_random_vector(size, dtype)
        #print repr(A)
        #print repr(b)

        # bicgstab is prone to failing on bad matrices
        A = tmd.bicgmat[dtype]
        b = tmd.bicgvec[dtype]

        A_op = pyublasext.MatrixOperator.make(A)
        bicgstab_op = pyublasext.BiCGSTABOperator.make(A_op, 40000, 1e-10)
        #bicgstab_op.debug_level = 1
        x = numpy.zeros((size,), dtype)

        initial_resid = la.norm(b - numpy.dot(A, x))
        bicgstab_op.apply(b, x)
        end_resid = la.norm(b - numpy.dot(A, x))
        self.assert_(end_resid/initial_resid < 1e-10)

    def test_bicgstab(self):
        self.for_all_dtypes(self.do_test_bicgstab)

    def test_complex_adaptor(self):
        size = 10

        a = numpy.random.randn(size,size) + 1j * numpy.random.randn(size,size)
        a_op = pyublasext.MatrixOperator.make(a)
        a2_op = pyublasext.adapt_real_to_complex_operator(
            pyublasext.MatrixOperator.make(a.real.copy()), 
            pyublasext.MatrixOperator.make(a.imag.copy()))

        for i in range(20):
            b = make_random_vector(size, complex)
            result1 = a_op(b)
            result2 = a2_op(b)
            self.assert_(la.norm(result1 - result2) < 1e-11)

    def test_python_operator(self):
        class MyOperator(pyublasext.Operator(float)):
            def size1(self):
                return A.shape[0]

            def size2(self):
                return A.shape[1]

            def apply(self, operand, result):
                result[:] = numpy.dot(A, operand)

        size = 10

        A = make_random_spd_matrix(size, float)
        
        Aop = MyOperator()
        b = numpy.random.randn(size)
        cg_op = pyublasext.CGOperator.make(Aop, 4000, 1e-10)

        initial_resid = la.norm(b)
        end_resid = la.norm(b - numpy.dot(A,cg_op(b)))
        self.assert_(end_resid/initial_resid < 1e-10)

    def test_daskr(self):
        def f(t, y):
            return numpy.array([y[1], -y[0]])

        t, y, yp = pyublasext.integrate_ode(numpy.array([0,1]), f, 0, 10)
        times = numpy.array(t)
        analytic_solution = numpy.sin(times)
        y = numpy.array(y)[:,0]

        self.assert_(la.norm(y-analytic_solution) < 1e-5)

    def do_test_sparse_operators(self, flavor):
        n = 100
        mat1 = make_random_matrix(n, float, flavor)
        mat2 = make_random_matrix(n, float, flavor)

        v = make_random_vector(n, float)

        err = mat1*(mat2*v)  - (mat1*mat2)*v
        self.assert_small(err, 1e-9)

        err = mat1*v + mat2*v - (mat1+mat2)*v
        self.assert_small(err)

        err = mat1*v - mat2*v - (mat1-mat2)*v
        self.assert_small(err)

    def test_sparse_operators(self):
        if not pyublas.has_sparse_wrappers():
            return
        for flavor in [pyublas.SparseBuildMatrix, pyublas.SparseExecuteMatrix]:
            self.do_test_sparse_operators(flavor)
            
if __name__ == '__main__':
    unittest.main()
