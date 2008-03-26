#
#  Copyright (c) 2004-2008
#  Andreas Kloeckner
#
#  Permission to use, copy, modify, distribute and sell this software
#  and its documentation for any purpose is hereby granted without fee,
#  provided that the above copyright notice appear in all copies and
#  that both that copyright notice and this permission notice appear
#  in supporting documentation.  The authors make no representations
#  about the suitability of this software for any purpose.
#  It is provided "as is" without express or implied warranty.
#




"""
Matrix-free methods, ARPACK and DASKR for Numpy.
"""




import pyublas
import numpy
import pyublasext._internal




has_blas = pyublasext._internal.has_blas
has_lapack = pyublasext._internal.has_lapack
has_arpack = pyublasext._internal.has_arpack
has_umfpack = pyublasext._internal.has_umfpack
has_daskr = pyublasext._internal.has_daskr




# operator parameterized types ------------------------------------------------
Operator = pyublas.ParameterizedType(
  "MatrixOperator", pyublasext._internal.__dict__)
IdentityOperator = pyublas.ParameterizedType(
  "IdentityMatrixOperator", pyublasext._internal.__dict__)
ScalarMultiplicationOperator = pyublas.ParameterizedType(
  "ScalarMultiplicationMatrixOperator", pyublasext._internal.__dict__)

class _MatrixOperatorParameterizedType(object):
    def is_a(self, instance):
        # FIXME
        raise NotImplementedError

    def __call__(self, dtype, flavor):
        # FIXME
        raise NotImplementedError

    def make(self, matrix):
        return pyublasext._internal.make_matrix_operator(matrix)

MatrixOperator = _MatrixOperatorParameterizedType()

class _CGParameterizedType(pyublas.ParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.dtype, w)
        if matrix_op.dtype is not precon_op.dtype:
            raise TypeError, "matrix_op and precon_op must have matching dtypes"
        return self.TypeDict[matrix_op.dtype](matrix_op, precon_op, max_it, tolerance)
    
CGOperator = _CGParameterizedType("CGMatrixOperator", pyublasext._internal.__dict__)

class _BiCGSTABParameterizedType(pyublas.ParameterizedType):
    def make(self, matrix_op, max_it=None, tolerance=1e-12, precon_op=None):
        if max_it is None:
            max_it = matrix_op.shape[0] * 10
        if precon_op is None:
            h,w = matrix_op.shape
            precon_op = IdentityOperator.make(matrix_op.dtype, w)
        if matrix_op.dtype is not precon_op.dtype:
            raise TypeError, "matrix_op and precon_op must have matching dtypes"
        return self.TypeDict[matrix_op.dtype](matrix_op, precon_op, max_it, tolerance)
    
BiCGSTABOperator = _BiCGSTABParameterizedType(
    "BiCGSTABMatrixOperator", pyublasext._internal.__dict__)

if has_umfpack():
    class _UMFPACKParameterizedType(pyublas.ParameterizedType):
        def make(self, matrix):
            matrix.complete_index1_data()
            return self.TypeDict[matrix.dtype](matrix)

    UMFPACKOperator = _UMFPACKParameterizedType("UMFPACKMatrixOperator", 
                                                        pyublasext._internal.__dict__)




def adapt_real_to_complex_operator(real_part, imaginary_part):
    if real_part.dtype != imaginary_part.dtype:
        raise TypeError, "outer and inner must have matching dtypes"
    return pyublasext._internal.ComplexMatrixOperatorAdaptorFloat64(
            real_part, imaginary_part)




# LU inverse operator ---------------------------------------------------------
class _LUInverseOperator:
    def __init__(self, mat, lu_fact):
        self.mat = mat
        self.lu_fact = lu_fact

    def size1(self):
        return self.mat.shape[0]
    
    def size2(self):
        return self.mat.shape[1]

    def apply(self, operand, result):
        import scipy.linalg
        result[:] = scipy.linalg.lu_solve(self.lu_fact, operand)

class _LUInverseOperatorFloat64(_LUInverseOperator, Operator(float)):
    def __init__(self, mat, lu_fact):
        _LUInverseOperator.__init__(self, mat, lu_fact)
        Operator(float).__init__(self)

class _LUInverseOperatorComplex128(_LUInverseOperator, Operator(complex)):
    def __init__(self, mat, lu_fact):
        _LUInverseOperator.__init__(self, mat, lu_fact)
        Operator(complex).__init__(self)

class _LUInverseParameterizedType(pyublas.ParameterizedType):
    def make(self, mat):
        import scipy.linalg
        return self.TypeDict[mat.dtype](mat, scipy.linalg.lu_factor(mat))

LUInverseOperator = _LUInverseParameterizedType("_LUInverseOperator", 
        globals())




# operator operators ----------------------------------------------------------
def _add_operator_behaviors():
    _SumOfOperators = pyublas.ParameterizedType(
      "SumOfMatrixOperators", pyublasext._internal.__dict__)
    _ScalarMultiplicationOperator = pyublas.ParameterizedType(
      "ScalarMultiplicationMatrixOperator", pyublasext._internal.__dict__)
    _CompositeOfOperators = pyublas.ParameterizedType(
      "CompositeMatrixOperator", pyublasext._internal.__dict__)




    def _neg_operator(op):
        return _compose_operators(
            _ScalarMultiplicationOperator(op.dtype)(-1, op.shape[0]),
            op)

    def _add_operators(op1, op2):
        return _SumOfOperators(op1.dtype)(op1, op2)

    def _subtract_operators(op1, op2):
        return _add_operators(op1, _neg_operator(op2))

    def _compose_operators(outer, inner):
        return _CompositeOfOperators(outer.dtype)(outer, inner)

    def _multiply_operators(op1, op2):
        if isinstance(op2, (float, int, complex)):
            return _compose_operators(
                op1,
                _ScalarMultiplicationOperator(op1.dtype)(op2, op1.shape[0]))
        else:
            return _compose_operators(op1, op2)

    def _reverse_multiply_operators(op1, op2):
        # i.e. op2 * op1
        assert num._is_number(op2)
        return _compose_operators(
            _ScalarMultiplicationOperator(op1.dtype)(op2, op1.shape[0]),
            op1)

    def _call_operator(op1, op2):
        try:
            temp = numpy.zeros((op1.shape[0],), op2.dtype)
            op1.apply(op2, temp)
            return temp
        except TypeError:
            # attempt applying a real operator to a complex problem
            temp_r = numpy.zeros((op1.shape[0],), float)
            temp_i = numpy.zeros((op1.shape[0],), float)
            op1.apply(op2.real, temp_r)
            op1.apply(op2.imaginary, temp_i)
            return temp_r + 1j*temp_i

    def get_returner(value):
        # This routine is necessary since we don't want the lambda in
        # the top-level scope, whose variables change.
        return lambda self: value

    for dtype in [float, complex]:
        Operator(dtype).__neg__ = _neg_operator
        Operator(dtype).__add__ = _add_operators
        Operator(dtype).__sub__ = _subtract_operators
        Operator(dtype).__mul__ = _multiply_operators
        Operator(dtype).__rmul__ = _reverse_multiply_operators
        Operator(dtype).__call__ = _call_operator
        Operator(dtype).typecode = get_returner(dtype)
        Operator(dtype).dtype = property(get_returner(dtype))




_add_operator_behaviors()




# arpack interface ------------------------------------------------------------
if has_arpack():
    SMALLEST_MAGNITUDE = pyublasext._internal.SMALLEST_MAGNITUDE
    LARGEST_MAGNITUDE = pyublasext._internal.LARGEST_MAGNITUDE
    SMALLEST_REAL_PART = pyublasext._internal.SMALLEST_REAL_PART
    LARGEST_REAL_PART = pyublasext._internal.LARGEST_REAL_PART
    SMALLEST_IMAGINARY_PART = pyublasext._internal.SMALLEST_IMAGINARY_PART
    LARGEST_IMAGINARY_PART = pyublasext._internal.LARGEST_IMAGINARY_PART

    def operator_eigenvectors(
        operator,
        n_eigenvectors,
        right_hand_operator=None,
        spectral_shift=None,
        which=LARGEST_MAGNITUDE,
        n_arnoldi_vectors=None,
        tolerance=1e-12,
        max_iterations=None):

        if n_arnoldi_vectors is None:
            n_arnoldi_vectors = min(2 * n_eigenvectors + 1, operator.size1())

        mode = pyublasext._internal.REGULAR_NON_GENERALIZED
        if right_hand_operator is not None:
            mode = pyublasext._internal.REGULAR_GENERALIZED
        if spectral_shift is not None:
            mode = pyublasext._internal.SHIFT_AND_INVERT_GENERALIZED

        if max_iterations is None:
            max_iterations = 0

        result = pyublasext._internal.run_arpack(operator, right_hand_operator,
                               mode, spectral_shift or 0,
                               n_eigenvectors,
                               n_arnoldi_vectors,
                               which,
                               tolerance,
                               max_iterations)

        return zip(result.RitzValues, result.RitzVectors)




# daskr interface -------------------------------------------------------------
if has_daskr():
    DAE = pyublasext._internal.DAE
    DAESolver = pyublasext._internal.DAESolver

    def integrate_dae(dae, t, y0, yprime0, t_end, steps=100, 
            intermediate_steps=False):
        solver = DAESolver(dae)

        y0 = numpy.asarray(y0, dtype=float)
        yprime0 = numpy.asarray(yprime0, dtype=float)

        times = [t]
        y_data = [y0]
        yprime_data = [yprime0]

        dt = float(t_end-t)/steps

        if intermediate_steps:
            solver.want_intermediate_steps = True

        t_start = t

        y = y0.copy()
        yprime = yprime0.copy()

        while t < t_end:
            progress_in_current_timestep = (t-t_start)%dt
            if progress_in_current_timestep > 0.99 * dt:
                next_timestep = t+2*dt-progress_in_current_timestep
            else:
                next_timestep = t+dt-progress_in_current_timestep

            state, t = solver.step(t, next_timestep, y, yprime0)

            times.append(t)
            y_data.append(y.copy())
            yprime_data.append(yprime.copy())

        return times, y_data, yprime_data




    def integrate_ode(initial, f, t, t_end, steps=100):
        n = len(f(t, initial))

        class my_dae(DAE):
            def dimension(self):
                return n

            def residual(self, t, y, yprime):
                #print t, y
                return yprime - f(t, y)

        return integrate_dae(my_dae(), t, initial, f(t, initial), t_end, steps=steps)




