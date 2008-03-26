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

#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/type.hpp>

// daskr ----------------------------------------------------------------------
#ifdef USE_DASKR
#include <pyublasext/daskr.hpp>

namespace python = boost::python;
namespace ublas = boost::numeric::ublas;
using namespace pyublas;
namespace daskr = boost::numeric::bindings::daskr;

namespace 
{
  template <typename Vector>
  struct dae_wrapper : public daskr::dae<Vector>, 
  python::wrapper<daskr::dae<Vector> >
  {
      unsigned dimension() const
      {
        return this->get_override("dimension")();
      }

      virtual Vector residual(
          double t, 
          const Vector &y,
          const Vector &yprime,
          bool &invalid) const
      {
        python::object result = this->get_override("residual")(t, y, yprime);

        if (result.ptr() == Py_None)
        {
          invalid = true;
          return Vector(dimension());
        }
        else
          return python::extract<Vector>(result);
      }
  };




  template <typename DAESolver>
  python::object daskr_step_wrapper(DAESolver &s,
      double t, double tout, 
      typename DAESolver::vector_ref_type y, 
      typename DAESolver::vector_ref_type yprime)
  {
    daskr::state state = s.step(t, tout, y, yprime);
    return python::make_tuple(state, t);
  }
}
#endif




void pyublasext_expose_daskr()
{
#ifdef USE_DASKR
  python::enum_<daskr::consistency>("daskr_consistency")
    .value("CONSISTENT", daskr::CONSISTENT)
    .value("COMPUTE_ALGEBRAIC", daskr::COMPUTE_ALGEBRAIC)
    .value("COMPUTE_Y", daskr::COMPUTE_Y)
    .export_values();

  python::enum_<daskr::state>("daskr_state")
    .value("STEP_TAKEN_NOT_TOUT", daskr::STEP_TAKEN_NOT_TOUT)
    .value("STEP_EXACTLY_TO_TSTOP", daskr::STEP_EXACTLY_TO_TSTOP)
    .value("STEP_PAST_TOUT", daskr::STEP_PAST_TOUT)
    .value("IC_SUCCESSFUL", daskr::IC_SUCCESSFUL)
    .value("FOUND_ROOT", daskr::FOUND_ROOT)
    .export_values();

  typedef numpy_vector<double> vec;
  {
    typedef daskr::dae<vec> cl;

    python::class_<dae_wrapper<vec>,
      boost::noncopyable>("DAE")
      .def("dimension", python::pure_virtual(&cl::dimension))
      .def("residual", &cl::residual)
      ;
  }

  {
    typedef daskr::dae_solver<vec, vec> cl;

#define RWPROP(name) \
      add_property(#name,  \
          &cl::name, \
          &cl::set_##name)
#define RWPROPDOC(name,doc) \
      add_property(#name,  \
          &cl::name, \
          &cl::set_##name, doc)

    python::class_<cl>("DAESolver", 
        python::init<daskr::dae<vec> &>()
        [python::with_custodian_and_ward<1,2>()])
      .RWPROP(relative_tolerance)
      .RWPROP(absolute_tolerance)
      .RWPROP(want_intermediate_steps)

      .RWPROP(want_tstop)
      .RWPROP(tstop)

      .RWPROP(want_max_step)
      .RWPROP(max_step)

      .RWPROP(want_ini_step)
      .RWPROP(ini_step)

      .RWPROP(init_consistency)
      .RWPROPDOC(want_return_with_ic,
          "Whether the integrator will return after consistent initial conditions"
          "have been computed.")

      .def("step", daskr_step_wrapper<cl>,
              "(self,t,tout,y,yprime) -> (state, t) Integrate the DAE by one step.\n\n"
              "Return a tuple (state, t) containing exit status and ending"
              "time of integration."
              )
      ;
  }
#endif // USE_DASKR
}
