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




#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/type.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/python.hpp>
#include <pyublas/numpy.hpp>
#include <pyublasext/cg.hpp>
#include <pyublasext/bicgstab.hpp>




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
      this->get_override("apply")(operand, result);
    }
};




// ublas_matrix_operator ------------------------------------------------------
struct ublas_matrix_operator_exposer
{
  template <typename MatrixType>
  struct type
  {
    typedef MatrixType matrix_type;
    typedef typename matrix_type::value_type value_type;
    typedef numpy_vector<value_type> vector_type;
    typedef
      ublas_matrix_operator<matrix_type, vector_type, vector_type>
      mop_type;
    typedef
      matrix_operator<vector_type, vector_type>
      super_type;

    static mop_type *make(const matrix_type &mat)
    { return new mop_type(mat); }

    static void expose(const std::string &python_mattype)
    {
      python::class_<mop_type, 
        python::bases<super_type> >
          (("MatrixOperator" + python_mattype).c_str(),
           python::init<const matrix_type &>()[python::with_custodian_and_ward<1,2>()]);
      python::def("make_matrix_operator", type::make,
          python::return_value_policy<
          python::manage_new_object,
          python::with_custodian_and_ward_postcall<0, 1> >());
    }
  };

  template <typename ValueType>
  struct type<numpy_matrix<ValueType> >
  {
    typedef numpy_matrix<ValueType> matrix_type;
    typedef typename matrix_type::value_type value_type;
    typedef numpy_vector<value_type> vector_type;
    typedef
      ublas_matrix_operator<matrix_type, vector_type, vector_type, matrix_type>
      mop_type;
    typedef
      matrix_operator<vector_type, vector_type>
      super_type;

    static mop_type *make(const matrix_type &mat)
    { return new mop_type(mat); }

    static void expose(const std::string &python_mattype)
    {
      python::class_<mop_type, 
        python::bases<super_type> >
          (("MatrixOperator" + python_mattype).c_str(),
           python::init<const matrix_type &>()[python::with_custodian_and_ward<1,2>()]);
      python::def("make_matrix_operator", make,
          python::return_value_policy<python::manage_new_object>());
    }
  };
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
static typename ResultsType::value_container::iterator begin_ritz_values(ResultsType &res)
{
  return res.m_ritz_values.begin();
}

template <typename ResultsType>
static typename ResultsType::value_container::iterator end_ritz_values(ResultsType &res)
{
  return res.m_ritz_values.end();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator begin_ritz_vectors(ResultsType &res)
{
  return res.m_ritz_vectors.begin();
}

template <typename ResultsType>
static typename ResultsType::vector_container::iterator end_ritz_vectors(ResultsType &res)
{
  return res.m_ritz_vectors.end();
}

template <typename ValueType, typename RealType>
arpack::results<numpy_vector<std::complex<RealType> > > *wrap_arpack(
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
  typedef matrix_operator<MOP_TP_ARGS(ValueType)> mop_type;
  typedef numpy_vector<ValueType> it_vec_type;
  typedef numpy_vector<std::complex<RealType> > res_vec_type;
  typedef arpack::results<res_vec_type> results_type;

  std::auto_ptr<results_type> results(new results_type());
  it_vec_type start_vector = ublas::unit_vector<ValueType>(op.size1(), 0);

  try
  {
    arpack::perform_reverse_communication<mop_type, res_vec_type, it_vec_type>(
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
  typedef typename arpack::results<numpy_vector<ValueType> > results_type;
  typedef typename ublas::type_traits<ValueType>::real_type real_type;

  python::class_<results_type>
    (("ArpackResults"+python_valuetypename).c_str())
    .add_property("RitzValues", 
        python::range(begin_ritz_values<results_type>, end_ritz_values<results_type>))
    .add_property("RitzVectors", 
        python::range(begin_ritz_vectors<results_type>, end_ritz_vectors<results_type>))
    ;

  python::def("run_arpack", wrap_arpack<ValueType, real_type>,
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
    void expose_with_matrices_with_value_type(const std::string &python_eltname)
    {
      Exposer::template type<numpy_matrix<ValueType> >::expose("Matrix" + python_eltname);
      Exposer::template type<
        ublas::compressed_matrix<ValueType, 
        ublas::column_major, 0, ublas::unbounded_array<int> 
        > >::expose("SparseExecuteMatrix" + python_eltname);
      Exposer::template type<
        ublas::coordinate_matrix<ValueType, ublas::column_major>
        >::expose("SparseBuildMatrix" + python_eltname);
    }




  template <typename Exposer>
    static void expose_with_all_matrices()
    {
      expose_with_matrices_with_value_type<Exposer, double>("Float64");
      expose_with_matrices_with_value_type<Exposer, std::complex<double> >("Complex128");
    }
}




// main -----------------------------------------------------------------------
void pyublasext_expose_daskr();




BOOST_PYTHON_MODULE(_internal)
{
  pyublasext_expose_daskr();

  expose_matrix_operators("Float64", double());
  expose_matrix_operators("Complex128", std::complex<double>());

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

  expose_with_all_matrices<ublas_matrix_operator_exposer>();

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
  exposeArpack("Complex128", std::complex<double>());
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
