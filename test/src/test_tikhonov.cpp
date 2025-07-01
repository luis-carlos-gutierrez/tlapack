#include "testutils.hpp"
//
#include <tlapack/lapack/tik_bidiag_elden.hpp>
#include <tlapack/plugins/stdvector.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("Testing all cases of Tikhonov",
                   "[tikhonov check]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;
    using range = pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    const idx_t m = GENERATE(10, 19, 30);
    const idx_t n = GENERATE(3, 7, 9);
    const idx_t k = GENERATE(7, 12, 19);
    const real_t lambda = GENERATE(7, 15, 24);

    DYNAMIC_SECTION("n = " << n << " m = " << m << " k = " << k
                           << " lambda = " << lambda)
    {
        // eps is the machine precision, and tol is the tolerance we accept for
        // tests to pass
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> A_copy_;
        auto A_copy = new_matrix(A_copy_, m, n);
        std::vector<T> b_;
        auto b = new_matrix(b_, m, k);
        std::vector<T> bcopy_;
        auto bcopy = new_matrix(bcopy_, m, k);
        std::vector<T> x_;
        auto x = new_matrix(x_, n, k);
        std::vector<T> y_;
        auto y = new_matrix(y_, n, k);

        // Initializing matrices randomly
        MatrixMarket mm;
        mm.random(A);
        mm.random(b);

        // Create copies for check
        lacpy(GENERAL, A, A_copy);
        lacpy(GENERAL, b, bcopy);

        // Run routine
        tik_bidiag_elden(A, b, lambda);

        // Check routine
        lacpy(GENERAL, slice(b, range{0, n}, range{0, k}), x);

        // Compute b - A *x -> b
        gemm(NO_TRANS, NO_TRANS, real_t(-1), A_copy, x, real_t(1), bcopy);

        // Compute A.H*(b - A x) -> y
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), A_copy, bcopy, y);

        // Compute A.H*(b - A x) - (lambda^2)*x -> y
        for (idx_t j = 0; j < k; j++)
            for (idx_t i = 0; i < n; i++)
                y(i, j) -= (lambda) * (lambda)*x(i, j);

        double dotProdNorm = lange(FROB_NORM, y);

        double normAcopy = lange(FROB_NORM, A_copy);

        double normbcopy = lange(FROB_NORM, bcopy);

        real_t error = dotProdNorm / (normAcopy * normbcopy);

        std::cout << "\nError =" << error;
        std::cout << "\ntol =" << tol;
        CHECK(error <= tol);
    }
}