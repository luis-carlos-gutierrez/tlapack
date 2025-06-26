#include "tlapack/base/utils.hpp"

using namespace tlapack;

/// Initialize  A augmented with Gamma and Vector b with zeros
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixAaug_t,
          TLAPACK_MATRIX matrixbaug_t>
void init_aug_mtx_vect(matrixA_t& A,
                       matrixb_t& b,
                       real_t lambda,
                       matrixAaug_t& Aaug,
                       matrixbaug_t& baug)
{
    using T = tlapack::type_t<matrixA_t>;
    using idx_t = tlapack::size_type<matrixA_t>;
    using matrix_t = tlapack::LegacyMatrix<T>;

    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = ncols(b);

    std::vector<T> Gamma_;
    auto Gamma = new_matrix(Gamma_, n, n);

    // Initialize Gamma = lambda*I
    laset(GENERAL, real_t(0), lambda, Gamma);

    // Augment Gamma onto A
    auto Aaug_top = slice(Aaug, range{0, m}, range{0, n});
    auto Aaug_bottom = slice(Aaug, range{m, m + n}, range{0, n});
    lacpy(GENERAL, A, Aaug_top);
    lacpy(GENERAL, Gamma, Aaug_bottom);

    // Augment zeros onto b
    std::vector<T> zeros_;
    auto zeros = new_matrix(zeros_, n, k);
    laset(GENERAL, real_t(0), real_t(0), zeros);

    auto baug_top = slice(baug, range{0, m}, range{0, k});
    auto baug_bottom = slice(baug, range{m, m + n}, range{0, k});

    lacpy(GENERAL, b, baug_top);
    lacpy(GENERAL, zeros, baug_bottom);
}
