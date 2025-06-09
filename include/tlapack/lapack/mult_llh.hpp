/// @file mult_llh.hpp
/// @author Brian Dang, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_MULT_LLH
#define TLAPACK_MULT_LLH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/trmm.hpp"

namespace tlapack {

/// @brief Options struct for llh_mult()
struct mult_llh_Opts {
    /// Optimization parameter. Matrices smaller than nx will not
    /// be multiplied using recursion. Must be at least 1.
    size_t nx = 1;
};

/**
 *
 * @brief in-place multiplication of lower triangular matrix L and upper
 * triangular matrix L^H. This is the recursive variant.
 *
 * @param[in,out] L n-by-n matrix
 * On entry, the lower triangular matrix L. On exit, L contains the lower
 * part of the Hermitian product L*L^H. The upper triangular entries of L are
 * not referenced.
 *
 * @param[in] opts Options.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SMATRIX matrix_t>
void mult_llh(matrix_t& L, const mult_llh_Opts& opts = {})
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    const idx_t n = nrows(L);
    tlapack_check(n == ncols(L));
    tlapack_check(opts.nx >= 1);

    // Quick return
    if (n == 0) return;

    if (n <= opts.nx) {
        for (idx_t j = n; j-- > 0;) {
            T real_part_of_ljj;
            real_part_of_ljj =
                real(L(j, j)) * real(L(j, j)) + imag(L(j, j)) * imag(L(j, j));
            for (idx_t k = 0; k < j; ++k) {
                // sum += C(i, k) * std::conj(C(i, k));
                real_part_of_ljj += real(L(j, k)) * real(L(j, k)) +
                                    imag(L(j, k)) * imag(L(j, k));
            }
            L(j, j) = real_part_of_ljj;

            for (idx_t i = j; i-- > 0;) {
                L(j, i) = L(j, i) * conj(L(i, i));
                for (idx_t k = 0; k < i; ++k) {
                    L(j, i) += L(j, k) * conj(L(i, k));
                }
            }
        }

        return;
    }

    // Recursive case: divide into blocks
    const idx_t n0 = n / 2;

    auto L00 = slice(L, range(0, n0), range(0, n0));
    auto L10 = slice(L, range(n0, n), range(0, n0));
    auto L11 = slice(L, range(n0, n), range(n0, n));

    // L11 = L11*L11^H
    mult_llh(L11, opts);

    // L11 += L10 * L10^H
    herk(Uplo::Lower, Op::NoTrans, real_t(1), L10, real_t(1), L11);

    // A10 = A10 * A00^H
    trmm(Side::Right, Uplo::Lower, Op::ConjTrans, Diag::NonUnit, T(1), L00,
         L10);

    // A00 = A00 * A00^H
    mult_llh(L00, opts);

    return;
}

}  // namespace tlapack

#endif  // TLAPACK_MULT_LLH
