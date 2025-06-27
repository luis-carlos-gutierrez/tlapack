#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

using namespace tlapack;

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

// /// Solves tikhonov regularized least squares using special bidiag method
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t>
void tik_special_bidiag(matrixA_t& A, matrixb_t& b, real_t lambda)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;
    using vector_t = LegacyVector<T>;

    using range = pair<idx_t, idx_t>;

    Create<matrixA_t> new_matrix;
    Create<vector_t> new_vector;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    std::vector<T> tauv(n);
    std::vector<T> tauw(n);

    // Define lambda vector for special bidiag routine
    std::vector<T> lamv_;
    auto lamv = new_matrix(lamv_, n, 1);

    // Initialize lambda vector
    for (idx_t i = 0; i < n; i++)
        lamv(i, 0) = lambda;

    // Bidiagonal decomposition
    bidiag(A, tauv, tauw);

    unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, b);

    auto x = slice(b, range{0, n}, range{0, k});

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

    std::vector<T> Baug_;
    auto Baug = new_matrix(Baug_, 2 * n, n);

    std::vector<T> baug_;
    auto baug = new_matrix(baug_, 2 * n, k);

    // Augment Gamma onto B
    auto Baug_top = slice(Baug, range{0, n}, range{0, n});
    auto Baug_bottom = slice(Baug, range{n, n + n}, range{0, n});
    lacpy(GENERAL, B, Baug_top);
    laset(GENERAL, real_t(0), lambda, Baug_bottom);

    // Augment zeros onto b
    auto baug_top = slice(baug, range{0, n}, range{0, k});
    auto baug_bottom = slice(baug, range{n, n + n}, range{0, k});
    lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), baug_top);
    laset(GENERAL, real_t(0), real_t(0), baug_bottom);

    //// Carlos works here
    real_t cs;
    T sn;

    // First Iteration
    // Print Baug to follow GR elimination algorithm
    std::cout << "\n\nBaug Before = \n";
    printMatrix(Baug);
    std::cout << std::endl;
    std::cout << "\n\nbaug Before = \n";
    printMatrix(baug);
    std::cout << std::endl;

    // Step 1

    rotg(Baug(0, 0), Baug(n, 0), cs, sn);
    Baug(n, 0) = 0;

    // // Step 2
    // auto xv = slice(Baug, 0, range{1, 2});
    // auto yv = slice(Baug, n, range{1, 2});
    // rot(xv, yv, cs, sn);

    Baug(n, 1) = -sn * Baug(0, 1);
    Baug(0, 1) = cs * Baug(0, 1);

    // Step 3
    auto bv = slice(baug, 0, range{0, k});
    auto cv = slice(baug, n, range{0, k});
    rot(bv, cv, cs, sn);

    // it is better to use rot( ) for this, since we act on rows
    // T tmp;
    // for (idx_t j = 0; j < k; ++j) {
    //     tmp = cs * baug(0, j) + sn * baug(n, j);
    //     baug(n, j) = -sn * baug(0, j) + cs * baug(n, j);
    //     baug(0, j) = tmp;
    // }

    std::cout << "\n\nBaug After = \n";
    printMatrix(Baug);
    std::cout << std::endl;
    std::cout << "\n\nbaug After = \n";
    printMatrix(baug);
    std::cout << std::endl;

    // // Second Iteration

    // rotg(Baug(n + 1, 0 + 1), Baug(n, 0 + 1), cs, sn);
    // Baug(n, 0 + 1) = 0;

    // bv = slice(baug, n + 1, range{0, k});
    // cv = slice(baug, n, range{0, k});
    // rot(bv, cv, cs, sn);
    // std::cout << "\n\nBaug3 = \n";
    // printMatrix(Baug);
    // std::cout << std::endl;
    // std::cout << "\n\nbaug3 = \n";
    // printMatrix(baug);
    // std::cout << std::endl;

    //// Carlos works here ↑.↑.↑.↑.↑.↑.↑.↑.↑.

    std::vector<T> tau_;
    auto tau = new_vector(tau_, n);

    geqrf(Baug, tau);

    unmqr(LEFT_SIDE, CONJ_TRANS, Baug, tau, baug);

    auto xN = slice(baug, range{0, n}, range{0, k});

    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);

    lacpy(UPPER_TRIANGLE, slice(Baug, range{0, n}, range{0, n}), R);

    trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), R, xN);

    lacpy(GENERAL, xN, x);

    // auto x2 = slice(x, range{1, n}, range{0, k});
    // unmlq(LEFT_SIDE, CONJ_TRANS, slice(A, range{1, n}, range{1, n}),
    //   slice(tauw, range{0, n - 1}), x2);

    // Apply P1ᵀ

    std::vector<T> P1_;
    auto P1 = new_matrix(P1_, n, n);
    lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);
    ungbr_p(n, P1, tauw);
    std::vector<T> x4_;
    auto x4 = new_matrix(x4_, n, k);
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, xN, real_t(0), x);
}