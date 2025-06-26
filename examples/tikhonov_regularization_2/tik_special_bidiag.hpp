#include "tlapack/base/utils.hpp"

/// Solves tikhonov regularized least squares using special bidiag method
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixx_t>
void tik_special_bidiag(matrixA_t& A, matrixb_t& b, real_t lambda, matrixx_t& x)
{}