//
// Copyright (c) 2004-2008 Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//




#include <boost/python.hpp>
#include <pyublas/numpy.hpp>
#include <pyublasext/cg.hpp>
#include <pyublasext/bicgstab.hpp>

#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/type.hpp>





// umfpack --------------------------------------------------------------------
#ifdef USE_UMFPACK
#include <pyublasext/umfpack.hpp>
#endif // USE_UMFPACK

// arpack ---------------------------------------------------------------------
#ifdef USE_ARPACK
#include <pyublasext/arpack.hpp>

namespace arpack = boost::numeric::bindings::arpack;
#endif // USE_ARPACK




namespace python = boost::python;
namespace ublas = boost::numeric::ublas;
using namespace pyublas;
using namespace pyublasext;




// shape accessors ------------------------------------------------------------
template <typename MatrixType>
inline python::object get_shape(const MatrixType &m)
{ 
  return python::make_tuple(m.size1(), m.size2());
}




#define MOP_TP_ARGS(VT) numpy_vector<VT>, numpy_vector<VT>




// wrappers -------------------------------------------------------------------
template <typename ValueType>
struct matrix_operator_wrapper 
: public matrix_operator<MOP_TP_ARGS(ValueType)>, 
  python::wrapper<matrix_operator<MOP_TP_ARGS(ValueType)> >
{
  private:
    typedef matrix_operator<MOP_TP_ARGS(ValueType)> super;
    
  public:
    unsigned size1() const
    {
      return this->get_override("size1")();
    }
    unsigned size2() const
    {
      return this->get_override("size2")();
    }
    void apply(const typename super::operand_type &operand, 
        typename super::complete_result_type result) const
    {
      this->get_override("apply")(boost::cref(operand), result);
    }
};




// ublas_matrix_operator ------------------------------------------------------
template <typename MatrixType>
static ublas_matrix_operator<
  MatrixType,
  numpy_vector<typename MatrixType::value_type>,
  numpy_vector<typename MatrixType::value_type>
  > *
make_matrix_operator(const MatrixType &mat)
{
  typedef
   ublas_matrix_operator<
   MatrixType,
    numpy_vector<typename MatrixType::value_type>,
    numpy_vector<typename MatrixType::value_type> >
      mop_type;
  return new mop_type(mat);
}




struct ublas_matrix_operator_exposer
{
  template <typename MatrixType>
  void expose(const std::string &python_mattype, MatrixType) const
  {
    typedef typename MatrixType::value_type value_type;
    typedef numpy_vector<value_type> vector_type;
    typedef
      ublas_matrix_operator<MatrixType, vector_type, vector_type>
        mop_type;

    typedef
      matrix_operator<vector_type, vector_type>
      super_type;

    python::class_<mop_type, 
    python::bases<super_type> >
      (("MatrixOperator" + python_mattype).c_str(),
       python::init<const MatrixType &>()[python::with_custodian_and_ward<1,2>()]);
    python::def("make_matrix_operator", make_matrix_operator<MatrixType>,
        python::return_value_policy<
        python::manage_new_object,
        python::with_custodian_and_ward_postcall<0, 1> >());
  }
};




// matrix operators -----------------------------------------------------------
template <typename ValueType>
static void expose_matrix_operators(const std::string &python_eltname, ValueType)
{
  {
    typedef matrix_operator<MOP_TP_ARGS(ValueType)> cl;
    python::class_<matrix_operator_wrapper<ValueType>, boost::noncopyable>
      (("MatrixOperator"+python_eltname).c_str())
      .add_property("shape", &get_shape<cl>)
      .def("size1", python::pure_virtual(&cl::size1))
      .def("size2", python::pure_virtual(&cl::size2))
      .def("apply", &cl::apply)
      ;
  }

  {
    typedef algorithm_matrix_operator<MOP_TP_ARGS(ValueType)> cl;
    python::class_<cl, 
      python::bases<matrix_operator<MOP_TP_ARGS(ValueType)> >,
      boost::noncopyable>
      (("AlgorithmMatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("debug_level", &cl::get_debug_level, &cl::set_debug_level)
      .add_property("last_iteration_count", &cl::get_last_iteration_count)
      ;
  }

  {
    typedef iterative_solver_matrix_operator<MOP_TP_ARGS(ValueType)> cl;

    python::class_<cl, 
      python::bases<algorithm_matrix_operator<MOP_TP_ARGS(ValueType)> >,
      boost::noncopyable >
      (("IterativeSolverMatrixOperator"+python_eltname).c_str(), python::no_init)
      .add_property("max_iterations", &cl::get_max_iterations, &cl::set_max_iterations)
      .add_property("tolerance", &cl::get_tolerance, &cl::set_tolerance)
      ;
  }

  {
    python::class_<identity_matrix_operator<MOP_TP_ARGS(ValueType)>, 
    python::bases<matrix_operator<MOP_TP_ARGS(ValueType)> > >
      (("IdentityMatrixOperator"+python_eltname).c_str(), 
       python::init<unsigned>());
  }

  {
    python::class_<composite_matrix_operator<MOP_TP_ARGS(ValueType)>, 
    python::bases<matrix_operator<MOP_TP_ARGS(ValueType)> > >
      (("CompositeMatrixOperator"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<MOP_TP_ARGS(ValueType)> &, 
         const matrix_operator<MOP_TP_ARGS(ValueType)> &>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  {
    python::class_<sum_of_matrix_operators<MOP_TP_ARGS(ValueType)>, 
    python::bases<matrix_operator<MOP_TP_ARGS(ValueType)> > >
      (("SumOfMatrixOperators"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<MOP_TP_ARGS(ValueType)> &, 
         const matrix_operator<MOP_TP_ARGS(ValueType)> &>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  {
    python::class_<scalar_multiplication_matrix_operator<
      numpy_vector<ValueType>, 
      ValueType,
      numpy_vector<ValueType>
      >, 
    python::bases<matrix_operator<MOP_TP_ARGS(ValueType)> > >
      (("ScalarMultiplicationMatrixOperator"+python_eltname).c_str(), 
       python::init<ValueType, unsigned>());
  }

  {
    typedef cg_matrix_operator<MOP_TP_ARGS(ValueType)> cl;
    python::class_<cl, 
    python::bases<iterative_solver_matrix_operator<MOP_TP_ARGS(ValueType)> > >
      (("CGMatrixOperator"+python_eltname).c_str(), 
       python::init<
         const matrix_operator<MOP_TP_ARGS(ValueType)> &, 
         const matrix_operator<MOP_TP_ARGS(ValueType)>&, 
         unsigned, double>()
         [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  {
    typedef bicgstab_matrix_operator<MOP_TP_ARGS(ValueType)> cl;
    python::class_<cl, 
      python::bases<iterative_solver_matrix_operator<MOP_TP_ARGS(ValueType)> > >
      (("BiCGSTABMatrixOperator"+python_eltname).c_str(), 
       python::init<
       const matrix_operator<MOP_TP_ARGS(ValueType)> &, 
       const matrix_operator<MOP_TP_ARGS(ValueType)>&, 
       unsigned, double>()
       [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

#ifdef USE_UMFPACK
  { 
    typedef umfpack_matrix_operator<MOP_TP_ARGS(ValueType)> cl;
    typedef 
      python::class_<cl, 
      python::bases<algorithm_matrix_operator<MOP_TP_ARGS(ValueType)> >, boost::noncopyable>    
        wrapper_type;

    wrapper_type pyclass(("UMFPACKMatrixOperator"+python_eltname).c_str(), 
       python::init<const typename cl::matrix_type &>()
       [python::with_custodian_and_ward<1,2>()]);
  }
#endif // USE_UMFPACK
}




// arpack ---------------------------------------------------------------------
#ifdef USE_ARPACK
template <typename ResultsType>
static typename ResultsType::value_container::iterator beginRitzValues(ResultsType &res)
{
  return res.m_ritz_values.begin();
}

template <typename ResultsType>
static typename ResultsType::value_container::iterator endRitzValues(ResultsType &res)
{
  return res.m_ritz_values.end();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator beginRitzVectors(ResultsType &res)
{
  return res.m_ritz_vectors.begin();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator endRitzVectors(ResultsType &res)
{
  return res.m_ritz_vectors.end();
}

template <typename ValueType, typename RealType>
arpack::results<ublas::vector<std::complex<RealType> > > *wrapArpack(
      const matrix_operator<MOP_TP_ARGS(ValueType)> &op, 
      const matrix_operator<MOP_TP_ARGS(ValueType)> *m,
      arpack::arpack_mode mode,
      std::complex<RealType> spectral_shift,
      int number_of_eigenvalues,
      int number_of_arnoldi_vectors,
      arpack::which_eigenvalues which_e,
      RealType tolerance,
      int max_iterations
      )
{
  typedef arpack::results<ublas::vector<std::complex<RealType> > > results_type;
  std::auto_ptr<results_type> results(new results_type());
  ublas::vector<ValueType> start_vector = ublas::unit_vector<ValueType>(op.size1(), 0);
  try
  {
    arpack::performReverseCommunication(
      op, m, mode, spectral_shift, 
      number_of_eigenvalues, number_of_arnoldi_vectors,
      *results, start_vector,
      which_e, tolerance, max_iterations);
  }
  catch (std::exception &ex)
  {
    std::cerr << ex.what() << std::endl;
    throw;
  }
  return results.release();
}




template <typename ValueType>
static void exposeArpack(const std::string &python_valuetypename, ValueType)
{
  typedef typename arpack::results<ublas::vector<ValueType> > results_type;
  typedef typename ublas::type_traits<ValueType>::real_type real_type;

  python::class_<results_type>
    (("ArpackResults"+python_valuetypename).c_str())
    .add_property("RitzValues", 
        python::range(beginRitzValues<results_type>, endRitzValues<results_type>))
    .add_property("RitzVectors", 
        python::range(beginRitzVectors<results_type>, endRitzVectors<results_type>))
    ;

  python::def("runArpack", wrapArpack<ValueType, real_type>,
              python::return_value_policy<python::manage_new_object>());
}
#endif // USE_ARPACK




// library support queries ----------------------------------------------------
bool has_blas() { 
#ifdef USE_BLAS
  return true; 
#else
  return false; 
#endif
}

bool has_lapack() { 
#ifdef USE_LAPACK
  return true; 
#else
  return false; 
#endif
}

bool has_arpack() { 
#ifdef USE_ARPACK
  return true; 
#else
  return false; 
#endif
}

bool has_umfpack() { 
#ifdef USE_UMFPACK
  return true; 
#else
  return false; 
#endif
}

bool has_daskr() { 
#ifdef USE_DASKR
  return true; 
#else
  return false; 
#endif
}




// expose helpers -------------------------------------------------------------
namespace
{
  template <typename Exposer, typename ValueType>
    void expose_for_all_simple_types(const std::string &python_eltname, const Exposer &exposer, ValueType)
    {
      exposer.expose("Matrix" + python_eltname, numpy_matrix<ValueType>());
      exposer.expose("SparseExecuteMatrix" + python_eltname, 
          ublas::compressed_matrix<ValueType, 
          ublas::column_major, 0, ublas::unbounded_array<int> >());
      exposer.expose("SparseBuildMatrix" + python_eltname, 
          ublas::coordinate_matrix<ValueType, ublas::column_major>());
    }




  template <typename Exposer, typename T>
    void expose_for_all_matrices(const Exposer &exposer, T)
    {
      expose_for_all_simple_types("Float64", exposer, T());
    }




  template <typename Exposer, typename T>
    static void expose_for_all_matrices(const Exposer &exposer, std::complex<T>)
    {
      expose_for_all_simple_types("Complex64", exposer, std::complex<T>());
    }




  template <typename Exposer>
    static void expose_for_all_matrices(const Exposer &exposer)
    {
      expose_for_all_matrices(exposer, double());
      expose_for_all_matrices(exposer, std::complex<double>());
    }




  template <typename Exposer,typename T>
    static void exposeForMatricesConvertibleTo(const Exposer &exposer, T)
    {
      expose_for_all_matrices(exposer, T());
    }




  template <typename Exposer,typename T>
    static void exposeForMatricesConvertibleTo(const Exposer &exposer, std::complex<T>)
    {
      expose_for_all_matrices(exposer);
    }

}




// main -----------------------------------------------------------------------
void pyublasext_expose_daskr();




BOOST_PYTHON_MODULE(_operation)
{
  pyublasext_expose_daskr();

  expose_matrix_operators("Float64", double());
  expose_matrix_operators("Complex64", std::complex<double>());

  // expose complex adaptor only for real-valued matrices
  {
    typedef double ValueType;
    typedef matrix_operator<MOP_TP_ARGS(ValueType)> real_op;
    typedef complex_matrix_operator_adaptor<real_op,
      MOP_TP_ARGS(std::complex<ValueType>)> cl;

    python::class_<cl, 
      python::bases<matrix_operator<MOP_TP_ARGS(std::complex<ValueType>)> > >
      ("ComplexMatrixOperatorAdaptorFloat64", 
       python::init< real_op &, real_op &>()
       [python::with_custodian_and_ward<1, 2, python::with_custodian_and_ward<1, 3> >()]);
  }

  expose_for_all_matrices(ublas_matrix_operator_exposer());

#ifdef USE_ARPACK
  python::enum_<arpack::which_eigenvalues>("arpack_which_eigenvalues")
    .value("SMALLEST_MAGNITUDE", arpack::SMALLEST_MAGNITUDE)
    .value("LARGEST_MAGNITUDE", arpack::LARGEST_MAGNITUDE)
    .value("SMALLEST_REAL_PART", arpack::SMALLEST_REAL_PART)
    .value("LARGEST_REAL_PART", arpack::LARGEST_REAL_PART)
    .value("SMALLEST_IMAGINARY_PART", arpack::SMALLEST_IMAGINARY_PART)
    .value("LARGEST_IMAGINARY_PART", arpack::LARGEST_IMAGINARY_PART)
    .export_values();

  python::enum_<arpack::arpack_mode>("arpack_mode")
    .value("REGULAR_NON_GENERALIZED", arpack::REGULAR_NON_GENERALIZED)
    .value("REGULAR_GENERALIZED", arpack::REGULAR_GENERALIZED)
    .value("SHIFT_AND_INVERT_GENERALIZED", arpack::SHIFT_AND_INVERT_GENERALIZED)
    .export_values();

  exposeArpack("Float64", double());
  exposeArpack("Complex64", std::complex<double>());
#endif // USE_ARPACK

  python::def("has_blas", has_blas, 
          "Return a bool indicating whether BLAS is available.");
  python::def("has_lapack", has_lapack,
          "Return a bool indicating whether LAPACK is available.");
  python::def("has_arpack", has_arpack,
          "Return a bool indicating whether ARPACK is available.");
  python::def("has_umfpack", has_umfpack,
          "Return a bool indicating whether UMFPACK is available.");
  python::def("has_daskr", has_daskr,
          "Return a bool indicating whether DASKR is available.");
}
