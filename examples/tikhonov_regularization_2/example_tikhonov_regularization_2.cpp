#include <tlapack/plugins/legacyArray.hpp>

#include "../../test/include/MatrixMarket.hpp"
#include "tlapack/base/utils.hpp"

// <T>LAPACK
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

//------------------------------------------------------------------
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
    std::cout << "\n\n";
}
//------------------------------------------------------------------
/// Initialize  A augmented with Gamma and Vector b with zeros
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_SCALAR scalar_t,
          TLAPACK_MATRIX matrixAaug_t,
          TLAPACK_MATRIX matrixbaug_t>
void initAugMtxVect(matrixA_t& A,
                    matrixb_t& b,
                    scalar_t lambda,
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
    laset(GENERAL, scalar_t(0), lambda, Gamma);

    // Augment Gamma onto A
    auto Aaug_top = slice(Aaug, range{0, m}, range{0, n});
    auto Aaug_bottom = slice(Aaug, range{m, m + n}, range{0, n});
    lacpy(GENERAL, A, Aaug_top);
    lacpy(GENERAL, Gamma, Aaug_bottom);

    // Augment zeros onto b
    std::vector<T> zeros_;
    auto zeros = new_matrix(zeros_, n, k);
    laset(GENERAL, scalar_t(0), scalar_t(0), zeros);

    auto baug_top = slice(baug, range{0, m}, range{0, k});
    auto baug_bottom = slice(baug, range{m, m + n}, range{0, k});

    lacpy(GENERAL, b, baug_top);
    lacpy(GENERAL, zeros, baug_bottom);
}
//------------------------------------------------------------------
/// Conducts the check for tikhonov QR and Cholesky
template <TLAPACK_MATRIX matrixA_copy_t,
          TLAPACK_MATRIX matrixb_t,
          TLAPACK_MATRIX matrixx_t>
void check_tik_QR_chol(matrixA_copy_t A_copy, matrixb_t b, matrixx_t x)
{
    using T = tlapack::type_t<matrixA_copy_t>;
    using scalar_t = tlapack::scalar_type<T>;

    // Compute b - A *x
    gemm(NO_TRANS, NO_TRANS, scalar_t(-1), A_copy, x, scalar_t(1), b);

    // Compute A.H*(b - A*x) -> x
    gemm(CONJ_TRANS, NO_TRANS, scalar_t(1), A_copy, b, x);

    // Compute ||A.H*(b - A*x)||_F
    double dotProdNorm = lange(FROB_NORM, x);

    // Compute ||A||_F
    double normAcopy = lange(FROB_NORM, A_copy);

    std::cout << "\n(||A.H*(b - A*x)||_F) / (||A||_F) =\n"
              << (dotProdNorm / normAcopy) << std::endl;
}
//------------------------------------------------------------------
/// Solves Least Squares with Tikhonov Reg using QR method
template <TLAPACK_MATRIX matrixAaug_t,
          TLAPACK_MATRIX matrixbaug_t,
          TLAPACK_SCALAR scalar_t,
          TLAPACK_MATRIX matrixx_t>
void tikqr(matrixAaug_t& Aaug,
           matrixbaug_t& baug,
           scalar_t lambda,
           matrixx_t& x)
{
    using T = type_t<matrixAaug_t>;
    using idx_t = tlapack::size_type<matrixAaug_t>;
    using vector_t = tlapack::LegacyVector<T>;

    const idx_t m = nrows(Aaug);
    const idx_t n = ncols(Aaug);
    const idx_t k = ncols(baug);

    // Functors for creating new matrices and vectors
    tlapack::Create<matrixAaug_t> new_matrix;
    tlapack::Create<vector_t> new_vector;

    using range = pair<idx_t, idx_t>;

    std::vector<T> tau_;
    auto tau = new_vector(tau_, std::min(m, n));
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);

    // Begin routine

    // AHH -> Aaug
    geqrf(Aaug, tau);

    // Apply Q^H to b using the compact QR representation
    std::vector<T> x2_;
    auto x2 = new_matrix(x2_, m, k);

    lacpy(GENERAL, baug, x2);
    unmqr(LEFT_SIDE, CONJ_TRANS, Aaug, tau, x2);

    // Extract upper triangular R from Aaug
    lacpy(UPPER_TRIANGLE, slice(Aaug, range{0, n}, range{0, n}), R);

    // Keep only the first n rows of Q^H * b as x
    lacpy(GENERAL, slice(x2, range{0, n}, range{0, k}), x);

    // Solve R * x = Q^H * b (now in x)
    trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, scalar_t(1), R, x);
}
//------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using scalar_t = tlapack::scalar_type<T>;

    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Declare Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> Aaug_;
    auto Aaug = new_matrix(Aaug_, m + n, n);
    std::vector<T> Aaug_copy_;
    auto Aaug_copy = new_matrix(Aaug_copy_, m + n, n);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, k);
    std::vector<T> b_copy_;
    auto b_copy = new_matrix(b_copy_, m, k);
    std::vector<T> baug_;
    auto baug = new_matrix(baug_, m + n, k);
    std::vector<T> baug_copy_;
    auto baug_copy = new_matrix(baug_copy_, m + n, k);
    std::vector<T> x_;
    auto x = new_matrix(x_, n, k);

    // Begin routine

    // Initializing matrices randomly
    MatrixMarket mm;
    mm.random(A);
    mm.random(b);

    // Initialize scalars
    scalar_t lambda(2);

    // Initialize augmented matrix
    initAugMtxVect(A, b, lambda, Aaug, baug);

    // Create matrix copies
    lacpy(GENERAL, A, A_copy);
    lacpy(GENERAL, b, b_copy);
    lacpy(GENERAL, Aaug, Aaug_copy);
    lacpy(GENERAL, baug, baug_copy);

    // Choose tikhonov method to solve least squares
    std::string method = "Tikhonov QR";
    // std::string method = "Tikhonov Cholesky";
    // std::string method = "Tikhonov SVD";

    std::cout << "\n\nSolving Least Squares using method: " << method
              << std::endl;

    // Pick appropriate subroutine
    if (method == "Tikhonov QR") {
        tikqr(Aaug, baug, lambda, x);
        check_tik_QR_chol(Aaug_copy, baug, x);
    }
    else if (method == "Tikhonov Cholesky") {
    }
    else if (method == "Tikhonov SVD") {
    }
    else {
        method = "No method chosen";
    }
}
//------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;
    int m, n, k;

    // Default arguments
    m = 3;
    n = 2;
    k = 5;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("----------------------------------------------------------\n");
    printf("run< float  >( %d, %d, %d )", m, n, k);
    run<float>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< double >( %d, %d, %d )", m, n, k);
    run<double>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< long double >( %d, %d, %d )", m, n, k);
    run<long double>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< complex<float> >( %d, %d, %d )", m, n, k);
    run<std::complex<float>>(m, n, k);
    printf("----------------------------------------------------------\n");

    printf("run< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double>>(m, n, k);
    printf("----------------------------------------------------------\n");
    return 0;
}
