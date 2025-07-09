/// @file trtr_ung2r.hpp
/// @author Julien Langou, L. Carlos Gutierrez, University of Colorado Denver,
/// USA
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Be careful that the lower part of Q0 is zeros and the routines does not touch
// the lower part of Q0. So if you want the exact Q0, you need to set zeros
// explicitly. Welcoming feedback on whether this routine should set the zeros
// for the users with a code like
//
// for (idx_t j = 0; j < n - 1; j++)
//     for (idx_t i = j + 1; i < n; i++)
//         A0(i, j) = real_t(0);

#ifndef TLAPACK_TRTR_UNG2R_HH
#define TLAPACK_TRTR_UNG2R_HH

#include "tlapack/base/utils.hpp"

using namespace tlapack;

template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
void trtr_ung2r(matrix_t& A0, matrix_t& A1, vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_t>;
    using real_t = real_type<T>;

    // A0 is output only

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    idx_t n = nrows(A0);
    idx_t m = nrows(A1);
    idx_t k = ncols(A0);

    auto view_block_A0 = slice(A0, range{0, n}, range{n, k});
    laset(GENERAL, real_t(0.0), real_t(0.0), view_block_A0);

    // Initialize A1 to the identity
    auto view_block_A1 = slice(A1, range{0, n}, range{n, k});
    laset(GENERAL, real_t(0.0), real_t(1.0), view_block_A1);

    auto view_A1 = slice(A1, range{0, n}, n - 1);
    auto view_A0 = slice(A0, n - 1, range{n, k});

    std::vector<T> work_tr_g2r_;
    auto work_tr_g2r = new_matrix(work_tr_g2r_, k, 1);

    if (k > n)
        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, view_A1, tau[n - 1], view_A0,
                  view_block_A1, work_tr_g2r);

    for (idx_t j = 0; j < n - 1; j++)
        A0(j, n - 1) = 0.;
    scal(-tau[n - 1], view_A1);
    A0(n - 1, n - 1) = T(1) - tau[n - 1];

    for (idx_t i = n - 1; i-- > 0;) {
        auto view_A1 = slice(A1, range{0, n}, i);
        auto view_A0 = slice(A0, i, range{i + 1, k});
        auto view_block_A1 = slice(A1, range{0, n}, range{i + 1, k});

        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, view_A1, tau[i], view_A0,
                  view_block_A1, work_tr_g2r);

        for (idx_t j = 0; j < i; j++)
            A0(j, i) = 0.;
        scal(-tau[i], view_A1);
        A0(i, i) = T(1) - tau[i];
    }
}
#endif