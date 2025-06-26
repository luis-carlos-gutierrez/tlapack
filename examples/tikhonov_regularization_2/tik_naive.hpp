#include "tlapack/base/utils.hpp"

using namespace tlapack;

/// Solves Least Squares with Tikhonov Reg using Naive method
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixx_t>
void tik_naive(matrixA_t& A, matrixb_t& b, real_t lambda, matrixx_t& x)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;
    using vector_t = LegacyVector<T>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    // Functors for creating new matrices and vectors
    Create<matrixA_t> new_matrix;
    Create<vector_t> new_vector;

    using range = pair<idx_t, idx_t>;

    // Declare augmented matrices
    std::vector<T> Aaug_;
    auto Aaug = new_matrix(Aaug_, m + n, n);
    std::vector<T> baug_;
    auto baug = new_matrix(baug_, m + n, k);

    // Declare vectors and matrices
    std::vector<T> tau_;
    auto tau = new_vector(tau_, std::min(m, n));
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);

    // Initialize augmented matrices
    init_aug_mtx_vect(A, b, lambda, Aaug, baug);

    // Begin routine

    // AHH -> Aaug
    geqrf(Aaug, tau);

    // Apply Q^H to b using the compact QR representation
    std::vector<T> x2_;
    auto x2 = new_matrix(x2_, m + n, k);

    lacpy(GENERAL, baug, x2);
    unmqr(LEFT_SIDE, CONJ_TRANS, Aaug, tau, x2);

    // Extract upper triangular R from Aaug
    lacpy(UPPER_TRIANGLE, slice(Aaug, range{0, n}, range{0, n}), R);

    // Keep only the first n rows of Q^H * b as x
    lacpy(GENERAL, slice(x2, range{0, n}, range{0, k}), x);

    // Solve R * x = Q^H * b (now in x)
    trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), R, x);
}