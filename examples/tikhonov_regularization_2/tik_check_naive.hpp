#include "tlapack/base/utils.hpp"

using namespace tlapack;

/// Conducts a naive check for naive_tik
template <TLAPACK_MATRIX matrixA_copy_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_REAL real_t,
          TLAPACK_MATRIX matrixx_t>
void tik_check_naive(matrixA_copy_t A_copy,
                     matrixb_t b,
                     real_t lambda,
                     matrixx_t x)
{
    using T = type_t<matrixA_copy_t>;
    using idx_t = size_type<matrixA_copy_t>;

    Create<matrixA_copy_t> new_matrix;

    const idx_t m = nrows(A_copy);
    const idx_t n = ncols(A_copy);
    const idx_t k = ncols(b);

    // Declare augmented matrices
    std::vector<T> Aaug_;
    auto Aaug = new_matrix(Aaug_, m + n, n);
    std::vector<T> baug_;
    auto baug = new_matrix(baug_, m + n, k);

    std::vector<T> y_;
    auto y = new_matrix(y_, n, k);

    // Initialize augmented matrices
    init_aug_mtx_vect(A_copy, b, lambda, Aaug, baug);

    // Compute baug - Aaug *x
    gemm(NO_TRANS, NO_TRANS, real_t(-1), Aaug, x, real_t(1), baug);

    // Compute Aaug.H*(baug - Aaug*x)
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), Aaug, baug, y);

    double dotProdNorm = lange(FROB_NORM, y);

    double normAcopy = lange(FROB_NORM, Aaug);

    std::cout << std::endl
              << "(||Aaug.H*(baug - Aaug*x)||_F) / (||A||_F) = " << std::endl
              << (dotProdNorm / normAcopy) << std::endl;
}