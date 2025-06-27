#include <tlapack/plugins/legacyArray.hpp>

#include "../../test/include/MatrixMarket.hpp"
#include "tlapack/base/utils.hpp"

// Utility functions created for the example
#include "init_aug_mtx_vect.hpp"

// Check functions created for the example
#include "tik_check.hpp"
#include "tik_check_naive.hpp"

// lsq solver functions created for the example
#include "tik_bidiag.hpp"
// #include "tik_chol.hpp"
#include "tik_naive.hpp"
#include "tik_special_bidiag.hpp"
#include "tik_svd.hpp"
// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/lapack/unmqr.hpp>

using namespace tlapack;

//------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using real_t = real_type<T>;
    using matrix_t = LegacyMatrix<T>;
    using idx_t = size_type<matrix_t>;

    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Declare Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, k);
    std::vector<T> x_;
    auto x = new_matrix(x_, n, k);

    // Begin routine

    // Initializing matrices randomly
    MatrixMarket mm;
    mm.random(A);
    mm.random(b);

    // Create matrix copy
    lacpy(GENERAL, A, A_copy);

    // Initialize scalars
    real_t lambda(4);

    // Choose tikhonov method to solve least squares
    std::string method;
    int option = 5;

    switch (option) {
        case 1:
            method = "Tikhonov Naive";
            break;
        case 2:
            method = "Tikhonov Cholesky";
            break;
        case 3:
            method = "Tikhonov SVD";
            break;
        case 4:
            method = "Tikhonov Bidiag";
            break;
        case 5:
            method = "Tikhonov Special Bidiag";
            break;
        default:
            method = "No method chosen";
    }

    // Outputs method used to solve Least Squares
    std::cout << "\n\nSolving Least Squares using method: " << method << "\n";

    // Executes desired subroutine
    if (method == "Tikhonov Naive") {
        // Naive method for reference
        tik_naive(A, b, lambda, x);
        // tik_check_naive(A_copy, b, lambda, x);
        tik_check(A_copy, b, lambda, x);
    }
    else if (method == "Tikhonov Cholesky") {
        tik_chol(A, b, lambda, x);
        tik_check(A_copy, b, lambda, x);
    }
    else if (method == "Tikhonov SVD") {
        tik_svd(A, b, lambda, x);
        tik_check(A_copy, b, lambda, x);
    }
    else if (method == "Tikhonov Bidiag") {
        std::vector<T> bcopy_;
        auto bcopy = new_matrix(bcopy_, m, k);

        lacpy(GENERAL, b, bcopy);

        tik_bidiag(A, b, lambda);

        lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);
        lacpy(GENERAL, bcopy, b);

        tik_check(A_copy, b, lambda, x);
    }
    else if (method == "Tikhonov Special Bidiag") {
        std::vector<T> bcopy_;
        auto bcopy = new_matrix(bcopy_, m, k);

        lacpy(GENERAL, b, bcopy);

        tik_special_bidiag(A, b, lambda);

        lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);
        lacpy(GENERAL, bcopy, b);

        tik_check(A_copy, b, lambda, x);
    }
    //  maybe TODO
    else if (method == "Tikhonov Special QR") {
        // tik_special_bidiag(A, b, lambda, x);
        // tik_check(A_copy, b, lambda, x);
    }
}
//------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, k;

    // Default arguments
    m = 6;
    n = 3;
    k = 2;

    // Init random seed
    srand(3);

    // Set output format
    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    // Execute run for different variable types
    // printf("----------------------------------------------------------\n");
    // printf("run< float  >( %d, %d, %d )", m, n, k);
    // run<float>(m, n, k);
    // printf("----------------------------------------------------------\n");

    // printf("run< double >( %d, %d, %d )", m, n, k);
    // run<double>(m, n, k);
    // printf("----------------------------------------------------------\n");

    // printf("run< long double >( %d, %d, %d )", m, n, k);
    // run<long double>(m, n, k);
    // printf("----------------------------------------------------------\n");

    // printf("run< complex<float> >( %d, %d, %d )", m, n, k);
    // run<std::complex<float>>(m, n, k);
    // printf("----------------------------------------------------------\n");

    printf("run< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double>>(m, n, k);
    printf("----------------------------------------------------------\n");
    return 0;
}
