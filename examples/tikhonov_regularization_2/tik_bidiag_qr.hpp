#include <tlapack/lapack/larfg.hpp>

#include "tlapack/base/utils.hpp"

using namespace tlapack;

/// Solves tikhonov regularized least squares using QR factorization and
/// elimination algorithm
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixx_t>
void tik_bidiag_qr(matrixA_t& A, matrixb_t& b, real_t lambda, matrixx_t& x)
{
    using T = type_t<matrixA_t>;
    using idx_t = size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using vector_t = LegacyVector<T>;

    Create<matrixA_t> new_matrix;
    Create<vector_t> new_vector;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    std::vector<T> tau_;
    auto tau = new_vector(tau_, n);

    geqrf(A, tau);

    unmqr(LEFT_SIDE, CONJ_TRANS, A, tau, x);

    auto Rview = slice(A, range{0, n}, range{0, n});

    // std::vector<T> R_;
    // auto R = new_matrix(R_, n, n);

    // lacpy(UPPER_TRIANGLE, Rview, R);

    // printMatrix(R);

    std::vector<T> Raug_;
    auto Raug = new_matrix(Raug_, n + n, n);

    auto Upper_Raug = slice(Raug, range{0, n}, range{0, n});
    auto Lower_Raug = slice(Raug, range{n, n + n}, range{0, n});

    lacpy(UPPER_TRIANGLE, Rview, Upper_Raug);
    laset(GENERAL, real_t(0), lambda, Lower_Raug);

    std::cout << "\nRaug =\n";
    // printMatrix(Raug);

    std::vector<T> tau0_;
    auto tau0 = new_vector(tau0_, n);

    // 2) Apply Householder reflectors to zero out the λ·I block
    for (idx_t j = 0; j < n; ++j) {
        // 2a) Form the vector x = B[j:2n-1, j]
        auto x0 = slice(Raug, range{j, 2 * n}, j);

        // 2b) Compute reflector H = I - τ v vᵀ so that H·x = [β, 0, …, 0]ᵀ
        type_t<vector_t> tau0;
        larfg(Direction::Forward, StoreV::Columnwise, x0, tau0);

        // 2c) Apply H to all columns j…n-1 of rows j…2n-1:
        auto block = slice(Raug, range{j, 2 * n}, range{j, n});
        // Note: correct argument order is Side, Direction, StoreV, v, tau,
        // matrix
        larf(Side::Left, Direction::Forward, StoreV::Columnwise, x0, tau0,
             block);
    }

    std::cout << "\nRaug Updated =\n";
    // printMatrix(Raug);
}