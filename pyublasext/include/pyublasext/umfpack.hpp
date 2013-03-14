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




#ifndef HEADER_SEEN_UMFPACK_HPP
#define HEADER_SEEN_UMFPACK_HPP




#include <stdexcept>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/ublas_sparse.hpp>
#include <boost/numeric/bindings/umfpack/umfpack.hpp>
#include "matrix_operator.hpp"




namespace pyublasext
{




  template <typename OperandType, typename ResultType = OperandType &>
  class umfpack_matrix_operator : public algorithm_matrix_operator<OperandType, ResultType>,
  boost::noncopyable
  {
    private:
      typedef algorithm_matrix_operator<OperandType, ResultType> super;

      typedef typename OperandType::value_type value_type;

    public:
      typedef
        boost::numeric::ublas::compressed_matrix<
        value_type, boost::numeric::ublas::column_major, 0, 
        boost::numeric::ublas::unbounded_array<int> >
          matrix_type;

    private:
      const matrix_type &m_matrix;
      boost::numeric::bindings::umfpack::numeric_type<value_type> m_numeric;

    public:
      umfpack_matrix_operator(const matrix_type &src)
        : m_matrix(src)
      { 
        process_umfpack_error(
            boost::numeric::bindings::umfpack::factor(m_matrix, m_numeric));
      }

      unsigned size1() const
      {
        return m_matrix.size2();
      }
      unsigned size2() const
      {
        return m_matrix.size1();
      }

      void apply(const OperandType &operand, ResultType result) const
      {
        super::apply(operand, result);
        process_umfpack_error(
            boost::numeric::bindings::umfpack::solve(
              m_matrix, result, operand, m_numeric));
        // FIXME: honor debug levels?
      }

    private:
      static void process_umfpack_error(int umf_error) 
      {
        switch (umf_error)
        {
          case UMFPACK_OK: 
            return;
          case UMFPACK_ERROR_out_of_memory: 
            throw std::runtime_error("umfpack: out of memory");
          case UMFPACK_ERROR_invalid_Numeric_object: 
            throw std::runtime_error("umfpack: invalid numeric object");
          case UMFPACK_ERROR_invalid_Symbolic_object:
            throw std::runtime_error("umfpack: invalid symbolic object");
          case UMFPACK_ERROR_argument_missing:
            throw std::runtime_error("umfpack: argument missing");
          case UMFPACK_ERROR_n_nonpositive:
            throw std::runtime_error("umfpack: n non-positive");
          case UMFPACK_ERROR_invalid_matrix:
            throw std::runtime_error("umfpack: invalid matrix");
          case UMFPACK_ERROR_different_pattern:
            throw std::runtime_error("umfpack: different pattern");
          case UMFPACK_ERROR_invalid_system:
            throw std::runtime_error("umfpack: invalid system");
          case UMFPACK_ERROR_invalid_permutation:
            throw std::runtime_error("umfpack: invalid permutation");
          case UMFPACK_ERROR_internal_error:
            throw std::runtime_error("umfpack: internal error");
          case UMFPACK_ERROR_file_IO:
            throw std::runtime_error("umfpack: file i/o error");
          default:
            throw std::runtime_error("umfpack: invalid error code");
        }
      }
  };
}




#endif
