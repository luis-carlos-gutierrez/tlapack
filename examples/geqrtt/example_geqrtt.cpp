#include <tlapack/plugins/legacyArray.hpp>
//
#include <../test/include/MatrixMarket.hpp>

#include "trtr_qr2.hpp"
#include "trtr_ung2r.hpp"
#include "trtr_unm2r.hpp"

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

    MatrixMarket mm;

    Create<matrix_t> new_matrix;

    ////////////// START ROUTINE FOR TRTR_QR2 ////////////////////

    // Declare Upper trianguler matrices for trtr_qr2
    std::vector<T> A0_;
    auto A0 = new_matrix(A0_, n, n);
    std::vector<T> A1_;
    auto A1 = new_matrix(A1_, n, n);

    mm.random(A0);
    mm.random(A1);

    // Create copies of A0 and A1 to make sure lower parts are untouched
    std::vector<T> A0_copy_;
    auto A0_copy = new_matrix(A0_copy_, n, n);
    std::vector<T> A1_copy_;
    auto A1_copy = new_matrix(A1_copy_, n, n);

    lacpy(GENERAL, A0, A0_copy);
    lacpy(GENERAL, A1, A1_copy);

    // Declare tau for trtr_qr2
    std::vector<T> tau(n);

    // call function
    trtr_qr2(A0, A1, tau);

    // Check that lower parts of A0 and A1 are untouched
    bool touchedA0 = false;
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            if (A0_copy(i, j) != A0(i, j)) touchedA0 = true;
    if (touchedA0)
        std::cout
            << "\nerror, lower part of A0 has been modified by trtr_qr2!\n";
    bool touchedA1 = false;
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            if (A1_copy(i, j) != A1(i, j)) touchedA1 = true;
    if (touchedA1)
        std::cout
            << "\nerror, lower part of A1 has been modified by trtr_qr2!\n";

    /////////////////// START ROUTINE FOR TRTR_UNG2R ////////////////////////

    // Declare Q0 and Q1 matrices to be created by ung2r
    std::vector<T> Q0_;
    auto Q0 = new_matrix(Q0_, n, k);
    std::vector<T> Q1_;
    auto Q1 = new_matrix(Q1_, n, k);

    // // Create copies of Q0 and Q1 to make sure the lower parts are untouched
    std::vector<T> Q0_copy_;
    auto Q0_copy = new_matrix(Q0_copy_, n, k);
    std::vector<T> Q1_copy_;
    auto Q1_copy = new_matrix(Q1_copy_, n, k);

    lacpy(GENERAL, Q0, Q0_copy);
    lacpy(GENERAL, Q1, Q1_copy);

    // Put Q in HH form into Q1 to be used in ung2r
    auto view_Q1 = slice(Q1, range{0, n}, range{0, n});
    lacpy(GENERAL, A1, view_Q1);

    // Call function
    //
    // Note: Q0 output only
    trtr_ung2r(Q0, Q1, tau);

    // Check that lower parts of Q0 and Q1 are untouched
    bool touchedQ0 = false;
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            if (Q0_copy(i, j) != Q0(i, j)) touchedQ0 = true;
    if (touchedQ0)
        std::cout
            << "\nerror, lower part of Q0 has been modified by trtr_ung2r !\n ";

    bool touchedQ1 = false;
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            if (Q1_copy(i, j) != Q1(i, j)) touchedQ1 = true;
    if (touchedQ1)
        std::cout
            << "\nerror, lower part of Q1 has been modified by trtr_ung2r !\n ";

    /////////////////// START ROUTINE FOR TRTR_UNM2R //////////////////

    // Declare and init C0 C1 to apply unm2r
    idx_t l = 6;
    std::vector<T> C0_;
    auto C0 = new_matrix(C0_, n, l);
    std::vector<T> C1_;
    auto C1 = new_matrix(C1_, n, l);

    mm.random(C0);
    mm.random(C1);

    // Create copies of C0 and C1 to be used in check #1
    std::vector<T> C0_copy_;
    auto C0_copy = new_matrix(C0_copy_, n, l);
    std::vector<T> C1_copy_;
    auto C1_copy = new_matrix(C1_copy_, n, l);

    lacpy(GENERAL, C0, C0_copy);
    lacpy(GENERAL, C1, C1_copy);

    // Call function
    // trtr_unm2r(A1, tau, C0, C1);

    /////////////////// START CHECKS ///////////////

    // Create fat Q for check
    std::vector<T> QQ_;
    auto QQ = new_matrix(QQ_, n + n, k);

    auto view_QQ = slice(QQ, range{0, n}, range{0, k});
    lacpy(GENERAL, Q0, view_QQ);

    view_QQ = slice(QQ, range{n, n + n}, range{0, k});
    lacpy(GENERAL, Q1, view_QQ);

    // 1) Compute check created by Julien
    // {
    //     // Create fat C for check
    //     std::vector<T> CC_;
    //     auto CC = new_matrix(CC_, n + n, l);

    //     auto view_CC = slice(CC, range{0, n}, range{0, l});
    //     lacpy(GENERAL, C0, view_CC);

    //     view_CC = slice(CC, range{n, n + n}, range{0, l});
    //     lacpy(GENERAL, C1, view_CC);

    //     // Create copy of CC for check
    //     std::vector<T> CC_copy_;
    //     auto CC_copy = new_matrix(CC_copy_, n + n, l);

    //     lacpy(GENERAL, CC, CC_copy);

    //     auto normC = lange(FROB_NORM, CC);

    //     // gemm(CONJ_TRANS, NO_TRANS, real_t(1.), QQ, CC_copy, real_t(-1.),
    //     CC);

    //     std::cout << lange(FROB_NORM, CC) / normC;
    // }

    // 2) Compute ||Qᴴ Q - I||_F

    // {
    //     // Create fat Q for check
    //     std::vector<T> QQ_;
    //     auto QQ = new_matrix(QQ_, n + n, k);

    //     auto view_QQ = slice(QQ, range{0, n}, range{0, k});
    //     lacpy(GENERAL, Q0, view_QQ);

    //     view_QQ = slice(QQ, range{n, n + n}, range{0, k});
    //     lacpy(GENERAL, Q1, view_QQ);

    //     std::vector<T> I_;
    //     auto I = new_matrix(I_, k, k);
    //     for (size_t j = 0; j < k; ++j)
    //         for (size_t i = 0; i < k; ++i)
    //             I(i, j) = static_cast<float>(0xABADBABE);

    //     // I receives the identity n*n
    //     laset(UPPER_TRIANGLE, real_t(0.0), real_t(1.0), I);

    //     // I receives QᴴQ - I
    //     herk(Uplo::Upper, Op::ConjTrans, real_t(1.0), QQ, real_t(-1.0), I);

    //     // Compute ||QᴴQ - I||_F
    //     real_t norm_orth = lanhe(FROB_NORM, UPPER_TRIANGLE, I);

    //     std::cout << "\n||Qᴴ Q - I||_F = " << norm_orth;
    // }

        ///////////// START CHECK #3 ///////////

    // 3) Compute ||QR - A||_F / ||A||_F

    // Create a fat matrix AA to be used in check
    std::vector<T> AA_;
    auto AA = new_matrix(AA_, n + n, n);

    // Put A0_copy and A1_copy into AA
    auto view_AA = slice(AA, range{0, n}, range{0, n});
    lacpy(GENERAL, A0_copy, view_AA);

    view_AA = slice(AA, range{n, n + n}, range{0, n});
    lacpy(GENERAL, A1_copy, view_AA);

    // Create a Q_thin from a view of QQ
    std::vector<T> Q_thin_;
    auto Q_thin = new_matrix(Q_thin_, n + n, n);

    view_QQ = slice(QQ, range{0, n + n}, range{0, n});
    lacpy(GENERAL, view_QQ, Q_thin);

    // Compute Q_thin*R
    trmm(RIGHT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1.0), A0,
         Q_thin);

    // Note: This is necessary for test
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = j + 1; i < n; i++)
            AA(i, j) = real_t(0);
    for (idx_t j = 0; j < n; j++)
        for (idx_t i = j + 1; i < n; i++)
            AA(n + i, j) = real_t(0);

    // Compute Q_thin*R - A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n + n; ++i)
            Q_thin(i, j) -= AA(i, j);

    // // Frobenius norm of A
    auto normA = lange(FROB_NORM, AA);

    real_t norm_repres = lange(FROB_NORM, Q_thin) / normA;

    std::cout << "\n||QR - A||_F/||A||_F = " << norm_repres;
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
