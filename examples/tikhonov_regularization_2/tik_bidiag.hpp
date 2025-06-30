#include "tik_chol.hpp"
// FIX LATER: Find a way to use include in .cpp ^
#include "tlapack/base/utils.hpp"

// <T>LAPACK
#include <tlapack/lapack/unmlq.hpp>

using namespace tlapack;

/// Solves tikhonov regularized least squares using bidiag method
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tik_bidiag(matrixA_t& A, matrixb_t& b, real_t lambda)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;

    Create<matrixA_t> new_matrix;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    std::vector<T> tauv(n);
    std::vector<T> tauw(n);

    // Step 1
    // Bidiagonal decomposition
    bidiag(A, tauv, tauw);

    // Step 2
    // Q1.H*b -> btmp1
    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, b);

    auto x = slice(b, range{0, n}, range{0, k});

    // Step 3: Tikhonov on bidiagonal matrix

    // Extract diagonal and superdiagonal
    std::vector<real_t> d(n);
    std::vector<real_t> e(n - 1);

    for (idx_t j = 0; j < n; ++j)
        d[j] = real(A(j, j));
    for (idx_t j = 0; j < n - 1; ++j)
        e[j] = real(A(j, j + 1));

    // Declare a biadiagonal matrix B
    std::vector<T> B_;
    auto B = new_matrix(B_, n, n);

    // Initialize the bidiagonals of B with d and e

    for (idx_t j = 0; j < n; ++j)
        B(j, j) = d[j];
    for (idx_t j = 0; j < n - 1; j++)
        B(j, j + 1) = e[j];

    // Step 4
    std::vector<T> y_;
    auto y = new_matrix(y_, n, k);
    // lacpy(GENERAL, x, y2);
    lacpy(GENERAL, x, y);
    tik_chol(B, y, lambda, x);

    // auto x2 = slice(x, range{1, n}, range{0, k});
    // unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{1, n}, range{1, n}),
    //       slice(tauw, range{0, n - 1}), x2);

    std::vector<T> P1_;
    auto P1 = new_matrix(P1_, n, n);
    lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);

    ungbr_p(n, P1, tauw);

    std::vector<T> x4_;
    auto x4 = new_matrix(x4_, n, k);

    lacpy(GENERAL, x, x4);

    gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, x4, real_t(0), x);
}
// TODO: understand if bidag() does A = QBP^T or A = QBP
// TODO: understand Tikhonov algorithm with SVD (math behind it)
// TODO: understand Tikhonov algorithm with bidiag (math behind it)
// TODO: in Tikhonov SVD: why can't we give Q1^Tb to SVD? So as to get Q2^T
// Q^1T b on the fly?