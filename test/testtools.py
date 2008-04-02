import pyublas
import pyublasext
import numpy
import numpy.linalg as la
import random




def write_random_vector(vec):
    size, = vec.shape
    for i in range(size):
        value = random.normalvariate(0,10)
        if vec.dtype == complex:
            value += 1j*random.normalvariate(0,10)
        vec[i] = value




def make_random_vector(size, dtype):
    vec = numpy.zeros((size,), dtype)
    write_random_vector(vec)
    return vec




def make_random_matrix(size, dtype, flavor):
    result = pyublas.zeros((size, size), dtype, flavor)
    elements = size ** 2 / 10

    for i in range(elements):
        row = random.randrange(0, size)
        col = random.randrange(0, size)
    
        value = random.normalvariate(0,10)
        if dtype == complex:
            value += 1j*random.normalvariate(0,10)

        result[row,col] += value
    return result




def orthonormalize(vectors, discard_threshold=None):
    """Carry out a modified [1] Gram-Schmidt orthonormalization on
    vectors.

    If, during orthonormalization, the 2-norm of a vector drops 
    below C{discard_threshold}, then this vector is silently 
    discarded. If C{discard_threshold} is C{None}, then no vector
    will ever be dropped, and a zero 2-norm encountered during
    orthonormalization will throw an L{OrthonormalizationError}.

    [1] U{http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process}
    """

    from numpy import dot
    done_vectors = []

    for v in vectors:
        my_v = v.copy()
        for done_v in done_vectors:
            my_v = my_v - dot(my_v, done_v.conjugate()) * done_v
        v_norm = la.norm(my_v)

        if discard_threshold is None:
            if v_norm == 0:
                raise RuntimeError, "Orthogonalization failed"
        else:
            if v_norm < discard_threshold:
                continue

        my_v /= v_norm
        done_vectors.append(my_v)

    return done_vectors




def make_random_orthogonal_matrix(size, dtype):
    vectors = []
    for i in range(size):
        v = numpy.zeros((size,), dtype)
        write_random_vector(v)
        vectors.append(v)

    orth_vectors = orthonormalize(vectors)

    mat = numpy.zeros((size,size), dtype)
    for i in range(size):
        mat[:,i] = orth_vectors[i]

    return mat


  

def make_random_spd_matrix(size, dtype):
    eigenvalues = make_random_vector(size, dtype)
    eigenmat = numpy.zeros((size,size), dtype)
    for i in range(size):
        eigenmat[i,i] = abs(eigenvalues[i])

    orthomat = make_random_orthogonal_matrix(size, dtype)
    return numpy.dot(
            numpy.conjugate(numpy.transpose(orthomat)), 
                numpy.dot(eigenmat, orthomat))





