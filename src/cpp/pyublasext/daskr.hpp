//
// Boost C++ DDASKR bindings
//
// Copyright (c) 2004-2006
// Andreas Kloeckner
//
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.
// It is provided "as is" without express or implied warranty.
//




#ifndef HEADER_SEEN_DASKR_HPP
#define HEADER_SEEN_DASKR_HPP




#include <boost/numeric/bindings/traits/vector_traits.hpp>  	
#include <boost/shared_array.hpp>  	




#define DASKR_F77NAME(x) x##_
extern "C"
{
  typedef void (*residual_function)(
      double &t, 
      double *y,
      double *yprime,
      double &cj,
      double *delta, // output
      int &ires,
      double *rpar,
      int *ipar);
  typedef void (*jacobian_function)(
      double *t, 
      double *y,
      double *yprime,
      double &cj,
      double *pd,
      double *rpar,
      int *ipar);

  extern void DASKR_F77NAME(ddaskr)(
      residual_function res,
      int &neq,
      double &t,
      double *y,
      double *yprime,
      const double &tout,
      int *info,
      double *rtol,
      double *atol,
      int &idid,
      double *rwork,
      int *lrw,
      int *iwork,
      int *liw,
      double *rpar,
      int *ipar,
      jacobian_function jac,
      void *psol, // not supported for now
      void *rt, // not supported for now
      int *nrt,
      int *jroot);
}




namespace boost { namespace numeric { namespace bindings {  namespace daskr {
  
  /** Describe a given Differential-Algebraic Equation.
   *
   */
  template <typename Vector>
  class dae
  {
    public:
      virtual ~dae() { }

      /** Return the dimension of the given DAE.
       * The dimension is also the number of equations.
       */
      virtual unsigned dimension() const = 0;

      /** Return the residual of the given DAE.
       */
      virtual Vector residual(
          double t, 
          const Vector &y,
          const Vector &yprime,
          bool &invalid
          ) const = 0;
  };

  enum consistency
  {
    CONSISTENT=0,
    COMPUTE_ALGEBRAIC=1,
    COMPUTE_Y=2
  };

  enum state
  {
    STEP_TAKEN_NOT_TOUT = 1,
    STEP_EXACTLY_TO_TSTOP = 2,
    STEP_PAST_TOUT = 3,
    IC_SUCCESSFUL = 4,
    FOUND_ROOT = 5
  };

  template <typename Vector, typename VectorRef = Vector &>
  class dae_solver
  {
    private:
      int INFO[20];
      dae<Vector> &Equation;
      double RelativeTolerance, AbsoluteTolerance;
      double TStop, MaxStep, IniStep;

      int LRW, LIW, NRT;

      boost::shared_array<double> RWORK;
      boost::shared_array<int> IWORK;

      int IPAR;

      bool Initialized;

      static dae<Vector> *StaticEquation;

    public:
      typedef Vector vector_type;
      typedef VectorRef vector_ref_type;

      dae_solver(dae<Vector> &eq)
        : Equation(eq), RelativeTolerance(1e-10), AbsoluteTolerance(1e-10),
        NRT(0), Initialized(false)
      {
        memset(INFO, 0, sizeof(INFO));
      }

      // tolerances -----------------------------------------------------------
      double relative_tolerance() const
      {
        return RelativeTolerance;
      }
      void set_relative_tolerance(double value)
      {
        RelativeTolerance = value;
      }

      double absolute_tolerance() const
      {
        return AbsoluteTolerance;
      }
      void set_absolute_tolerance(double value)
      {
        AbsoluteTolerance = value;
      }

      // want_intermediate_steps ----------------------------------------------
      bool want_intermediate_steps() const
      {
        return bool(INFO[3-1]);
      }
      void set_want_intermediate_steps(bool value)
      {
        INFO[3-1] = value ? 1 : 0;
      }

      // tstop ----------------------------------------------------------------
      bool want_tstop() const
      {
        return bool(INFO[4-1]);
      }
      void set_want_tstop(bool value)
      {
        INFO[4-1] = value ? 1 : 0;
      }

      double tstop() const
      {
        if (!Initialized)
          return TStop;
        else
          return RWORK[1-1];
      }
      void set_tstop(double value)
      {
        set_want_tstop(true);
        if (!Initialized)
          TStop = value;
        else
          RWORK[1-1] = value;
      }

      // max_step -------------------------------------------------------------
      bool want_max_step() const
      {
        return bool(INFO[7-1]);
      }
      void set_want_max_step(bool value)
      {
        INFO[7-1] = value ? 1 : 0;
      }

      double max_step() const
      {
        if (!Initialized)
          return MaxStep;
        else
          return RWORK[2-1];
      }
      void set_max_step(double value)
      {
        set_want_max_step(true);

        if (!Initialized)
          MaxStep = value;
        else
          RWORK[2-1] = value;
      }

      // ini_step -------------------------------------------------------------
      bool want_ini_step() const
      {
        return bool(INFO[8-1]);
      }
      void set_want_ini_step(bool value)
      {
        INFO[8-1] = value ? 1 : 0;
      }

      bool ini_step() const
      {
        if (!Initialized)
          return IniStep;
        else
          return RWORK[3-1];
      }
      void set_ini_step(double value)
      {
        set_want_ini_step(true);

        if (!Initialized)
          IniStep = value;
        else
          RWORK[3-1] = value;
      }

      // init_consistency -----------------------------------------------------
      consistency init_consistency() const
      {
        return consistency(INFO[11-1]);
      }
      void set_init_consistency(consistency value)
      {
        INFO[11-1] = value;
      }

      // want_return_with_ic --------------------------------------------------
      bool want_return_with_ic() const
      {
        return bool(INFO[14-1]);
      }
      void set_want_return_with_ic(bool value)
      {
        INFO[14-1] = value ? 1 : 0;
      }

      // driver ---------------------------------------------------------------
    private:
      void initialize()
      {
        if (Initialized)
          throw std::runtime_error("dae solver already initialized");

        unsigned neq = Equation.dimension(); 

        LRW = 60 + 9*neq + 3*NRT;
        LRW += neq*neq; /* INFO[6-1] == 0! */
        if (INFO[16-1] == 1)
          LRW += neq;

        LIW = 40 + neq;
        if (INFO[10-1] == 1 || INFO[10-1] == 3)
          LIW += neq;
        if (INFO[11-1] == 1 || INFO[16-1] == 1)
          LIW += neq;

        RWORK = boost::shared_array<double>(new double[LRW]);
        IWORK = boost::shared_array<int>(new int[LIW]);

        IPAR = 0;

        RWORK[1-1] = TStop;
        RWORK[2-1] = MaxStep;
        RWORK[3-1] = IniStep;

        Initialized = true;
      }

    public:
      state step(
          double &t, 
          double tout,
          VectorRef y,
          VectorRef yprime)
      {
        if (!Initialized)
          initialize();

        int neq = Equation.dimension(); 

        if (neq != traits::vector_size(y))
          throw std::runtime_error("y_initial has wrong dimension");

        if (neq != traits::vector_size(yprime))
          throw std::runtime_error("yprime_initial has wrong dimension");

        int idid;

        StaticEquation = &Equation;

        DASKR_F77NAME(ddaskr)(
            &res_callback,
            neq, t,
            traits::vector_storage(y), 
            traits::vector_storage(yprime), 
            tout, INFO, 
            &RelativeTolerance, &AbsoluteTolerance,
            idid,
            RWORK.get(), &LRW,
            IWORK.get(), &LIW,
            NULL /*RPAR*/, &IPAR,
            NULL /*JAC*/, NULL /*PSOL*/, NULL /*RT*/,
            &NRT, NULL /*JROOT*/
            );
        if (idid > 0)
          return state(idid);

        if (idid == -1)
          throw std::runtime_error("daskr: a large amount of work has "
              "been expended (about 500 steps) (-1)");
        else if (idid == -2)
          throw std::runtime_error("daskr: error tolerances too strict (-2)");
        else if (idid == -3)
          throw std::runtime_error("daskr: local error test unsatisfiable (-3)");
        else if (idid == -5)
          throw std::runtime_error("daskr: repeated failures in preconditioner (-5)");
        else if (idid == -6)
          throw std::runtime_error("daskr: repeated error test failures (-6)");
        else if (idid == -7)
          throw std::runtime_error("daskr: nonlinear system solver did not converge (-7)");
        else if (idid == -8)
          throw std::runtime_error("daskr: partial derivative matrix is singular (-8)");
        else if (idid == -9)
          throw std::runtime_error("daskr: nonlinear system solver failed (-9)");
        else if (idid == -10)
          throw std::runtime_error("daskr: nonlinear system solver failed -> ires (-10)");
        else if (idid == -11)
          throw std::runtime_error("daskr: failure in residual (-11)");
        else if (idid == -12)
          throw std::runtime_error("daskr: failed to compute initial y, yprime (-12)");
        else if (idid == -13)
          throw std::runtime_error("daskr: user psol failure (-13)");
        else if (idid == -14)
          throw std::runtime_error("daskr: krylov solver did not converge (-14)");
        else if (idid == -33)
          throw std::runtime_error("daskr: unrecoverable error (invalid input?) (-14)");
        else
          throw std::runtime_error("invalid error return from daskr");
      }

    private:
      static void res_callback(
        double &t, 
        double *y,
        double *yprime,
        double &cj,
        double *delta, // output
        int &ires,
        double *rpar,
        int *ipar)
      {
        dae<Vector> &eq(*StaticEquation);

        unsigned n = eq.dimension();

        Vector vec_y(n), vec_yprime(n);
        for (unsigned j = 0; j < n; j++)
        {
          vec_y[j] = y[j];
          vec_yprime[j] = yprime[j];
        }

        try
        {
          bool invalid = false;
          Vector result = eq.residual(t, vec_y, vec_yprime, invalid);

          if (invalid)
            ires = -1;
          else
            for (unsigned j = 0; j < n; j++)
              delta[j] = result[j];
        }
        catch (...)
        {
          ires = -2;
        }
      }
  };

  template<class Vector, typename VectorRef>
    dae<Vector> *dae_solver<Vector, VectorRef>::StaticEquation;

} } } }




#endif
