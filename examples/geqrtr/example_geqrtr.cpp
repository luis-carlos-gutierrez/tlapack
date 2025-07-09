#include <tlapack/plugins/legacyArray.hpp>
//
#include <../test/include/MatrixMarket.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/larf.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/trge_qr2.hpp>
#include <tlapack/lapack/trge_ung2r.hpp>
#include <tlapack/lapack/trge_unm2r.hpp>
#include <tlapack/lapack/ung2r.hpp>

using namespace tlapack;

template <typename T>
void run(size_t m, size_t n, size_t k)
{
    // Create utilities for code
    using matrix_t = LegacyMatrix<T>;
    using real_t = real_type<T>;

    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    MatrixMarket mm;

    ///////////////// START ROUTINE FOR TRGE_QR2 ////////////////////

    // Create matrices to be used in trge_qr2
    std::vector<T> A0_;
    auto A0 = new_matrix(A0_, n, n);
    std::vector<T> A1_;
    auto A1 = new_matrix(A1_, m, n);

    mm.random(A0);
    mm.random(A1);

    // Declare Arrays
    std::vector<T> tau(n);

    std::vector<T> Q0_check_;
    auto Q0_check = new_matrix(Q0_check_, n, n);

    // Create a copy A0 to preserve A0 to make sure bottom part untouched later
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);

    lacpy(GENERAL, A0, R);

    // Create a copy A1, to preserve A1 to be used as norm in check #3
    std::vector<T> V1_;
    auto V1 = new_matrix(V1_, m, n);

    lacpy(GENERAL, A1, V1);

    // Call function
    trge_qr2(R, V1, tau);

    // // check that the lower part of R has not been touched
    // bool touched = false;
    // for (idx_t j = 0; j < n - 1; j++)
    //     for (idx_t i = j + 1; i < n; i++)
    //         if (R(i, j) != A0(i, j)) touched = true;
    // if (touched)
    //     std::cout << std::endl
    //               << "error, lower part of A0 has been modified by geqrtr!"
    //               << std::endl;

    ///////////////// START ROUTINE FOR TRGE_UNG2R ////////////////////

    // Create matrix Q0 and Q1 that will be generated in ung2r
    // Note: Q0 output only in ung2r
    std::vector<T> Q0_;
    auto Q0 = new_matrix(Q0_, n, k);
    std::vector<T> Q1_;
    auto Q1 = new_matrix(Q1_, m, k);

    mm.random(Q1);

    // Copy Q in HH form into Q1 to be used in ung2r
    auto view_Q1 = slice(Q1, range{0, m}, range{0, n});
    lacpy(GENERAL, V1, view_Q1);

    // Call function
    trge_ung2r(Q0, Q1, tau);

    ///////////////// START ROUTINE FOR TRGE_UNM2R ////////////////////

    // Create matrices C0 and C1 to have Q in HH form applied to them
    idx_t l = 7;
    std::vector<T> C0_;
    auto C0 = new_matrix(C0_, n, l);
    std::vector<T> C1_;
    auto C1 = new_matrix(C1_, m, l);

    mm.random(C0);
    mm.random(C1);

    // Create copies of C0 and C1 to be used later in check #1
    std::vector<T> C0_copy_;
    auto C0_copy = new_matrix(C0_copy_, n, l);
    std::vector<T> C1_copy_;
    auto C1_copy = new_matrix(C1_copy_, m, l);

    lacpy(GENERAL, C0, C0_copy);
    lacpy(GENERAL, C1, C1_copy);

    // Put copies of C0 and C1 into a fat matrix that will be used later in
    // check #1
    std::vector<T> CC_copy_;
    auto CC_copy = new_matrix(CC_copy_, n + m, l);

    auto view_CC_copy = slice(CC_copy, range{0, n}, range{0, l});
    lacpy(GENERAL, C0_copy, view_CC_copy);

    view_CC_copy = slice(CC_copy, range{n, n + m}, range{0, l});
    lacpy(GENERAL, C1_copy, view_CC_copy);

    // Call Function
    trge_unm2r(V1, tau, C0, C1);

    /////////////////// START CHECKS //////////////////////////

    // // check that the lower part of Q0 has not been touched
    // touched = false;
    // for (idx_t j = 0; j < n - 1; j++)
    //     for (idx_t i = j + 1; i < n; i++)
    //         if (Q0(i, j) != Q0_check(i, j)) touched = true;
    // if (touched)
    //     std::cout << std::endl
    //               << "error, lower part of A0 has been modified by
    //               trge_ung2r!"
    //               << std::endl;

    // Note: So far, putting zeros does not affect the test
    //
    // Put zeros below Upper Triangle R in A
    // for (idx_t j = 0; j < n - 1; j++)
    //     for (idx_t i = j + 1; i < n; i++)
    //         Q0(i, j) = real_t(0);

    ///////////// START CHECK #1 ///////////

    // Create a fat matrix CC to be used in check
    std::vector<T> CC_;
    auto CC = new_matrix(CC_, n + m, l);

    auto view_CC = slice(CC, range{0, n}, range{0, l});
    lacpy(GENERAL, C0, view_CC);

    view_CC = slice(CC, range{n, n + m}, range{0, l});
    lacpy(GENERAL, C1, view_CC);

    // Create a fat matrix QQ to be used in check
    std::vector<T> QQ_;
    auto QQ = new_matrix(QQ_, n + m, k);

    auto view_QQ = slice(QQ, range{0, n}, range{0, k});
    lacpy(GENERAL, Q0, view_QQ);

    view_QQ = slice(QQ, range{n, n + m}, range{0, k});
    lacpy(GENERAL, Q1, view_QQ);

    auto normC = lange(FROB_NORM, CC);

    // Compute QQ.H*CC_copy - CC
    gemm(Op::ConjTrans, Op::NoTrans, real_t(1.), QQ, CC_copy, real_t(-1.), CC);

    std::cout << "\n\n(||QQ.H*CC_copy - CC||_F) / ||CC|| = "
              << lange(FROB_NORM, CC) / normC;

    // 2.5) Compute ||Qᴴ Q - I||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, k, k);
        for (size_t j = 0; j < k; ++j)
            for (size_t i = 0; i < k; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        laset(UPPER_TRIANGLE, 0.0, 1.0, work);

        // work receives QᴴQ - I
        herk(Uplo::Upper, Op::ConjTrans, real_t(1.0), QQ, real_t(-1.0), work);

        // Compute ||QᴴQ - I||_F
        real_t norm_orth = lanhe(FROB_NORM, UPPER_TRIANGLE, work);

        std::cout << "\n\n||Qᴴ Q - I||_F = " << norm_orth;
    }

    ///////////// START CHECK #3 ///////////

    // 3) Compute ||QR - A||_F / ||A||_F

    std::vector<T> AA_;
    auto AA = new_matrix(AA_, n + m, n);

    auto view_AA = slice(AA, range{0, n}, range{0, n});
    lacpy(GENERAL, A0, view_AA);

    view_AA = slice(AA, range{n, n + m}, range{0, n});
    lacpy(GENERAL, A1, view_AA);

    // Create a Q_thin from a view of QQ
    std::vector<T> Q_thin_;
    auto Q_thin = new_matrix(Q_thin_, m + n, n);

    view_QQ = slice(QQ, range{0, n + m}, range{0, n});
    lacpy(GENERAL, view_QQ, Q_thin);

    // Compute Q_thin*R
    trmm(RIGHT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1.0), R,
         Q_thin);

    // Note: This is necessary for test
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = j + 1; i < n; i++)
            AA(i, j) = real_t(0);

    // Compute Q_thin*R - A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m + n; ++i)
            Q_thin(i, j) -= AA(i, j);

    // // Frobenius norm of A
    auto normA = lange(FROB_NORM, AA);

    real_t norm_repres = lange(FROB_NORM, Q_thin) / normA;

    std::cout << "\n\n||QR - A||_F/||A||_F = " << norm_repres << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, k;

    // Default arguments
    m = (argc < 2) ? 3 : atoi(argv[1]);
    n = (argc < 3) ? 2 : atoi(argv[2]);
    k = (argc < 4) ? 2 : atoi(argv[3]);

    // k is between n and m+n

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double> >(m, n, k);
    std::cout << "-----------------------" << std::endl;

    return 0;
}
