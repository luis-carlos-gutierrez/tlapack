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

    // Functors for creating new matrices
    Create<matrix_t> new_matrix;

    // Declare Arrays
    std::vector<T> tau(n);

    // Declare Matrices

    std::vector<T> A0_;
    auto A0 = new_matrix(A0_, n, n);
    std::vector<T> A1_;
    auto A1 = new_matrix(A1_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);

    std::vector<T> Q0_;
    auto Q0 = new_matrix(Q0_, n, k);
    std::vector<T> Q1_;
    auto Q1 = new_matrix(Q1_, m, k);

    std::vector<T> V0_;
    auto V0 = new_matrix(V0_, n, n);
    std::vector<T> V1_;
    auto V1 = new_matrix(V1_, m, n);

    std::vector<T> Q0_check_;
    auto Q0_check = new_matrix(Q0_check_, n, n);

    // Randomly initialize matrices
    MatrixMarket mm;

    mm.random(A0);
    mm.random(A1);
    mm.random(Q0);
    mm.random(Q1);
    mm.random(R);

    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            Q0_check(i, j) = Q0(i, j);

    // lacpy(GENERAL, AA, QQthin);

    // trge_qr2(QQthin, AA0, tau);

    lacpy(GENERAL, A0, R);
    lacpy(GENERAL, A1, V1);

    /////////// CALL FUNCTION /////////////////
    trge_qr2(R, V1, tau);

    auto Q1thin = slice(Q1, range{0, m}, range{0, n});
    lacpy(GENERAL, V1, Q1thin);

    // Generate Q in place
    //
    /////////// CALL FUNCTION /////////////////
    //
    // Q0 output only
    trge_ung2r(Q0, Q1, tau);

    /////////////////////////// UNM2R
    idx_t l = 7;
    std::vector<T> C0_;
    auto C0 = new_matrix(C0_, n, l);
    std::vector<T> C1_;
    auto C1 = new_matrix(C1_, m, l);

    mm.random(C0);
    mm.random(C1);

    std::vector<T> D0_;
    auto D0 = new_matrix(D0_, n, l);
    std::vector<T> D1_;
    auto D1 = new_matrix(D1_, m, l);
    lacpy(GENERAL, C0, D0);
    lacpy(GENERAL, C1, D1);

    // Julien's trge_unm2r code
    {
        // std::vector<T> work_;
        // auto work = new_matrix(work_, l, 1);

        // for (idx_t i = 0; i < n; ++i) {
        //     auto v = slice(V1, range{0, m}, i);
        //     auto C0_view = slice(C0, i, range{0, l});
        //     auto C1_view = slice(C1, range{0, m}, range{0, l});

        //     larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, conj(tau[i]),
        //     C0_view,
        //               C1_view, work);
        // }
    }

    /////////// CALL FUNCTION /////////////////
    trge_unm2r(V1, tau, C0, C1);

    /////////////////////////// test start here

    // check that the lower part of R has not been touched
    bool touched = false;
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            if (R(i, j) != A0(i, j)) touched = true;
    if (touched)
        std::cout << std::endl
                  << "error, lower part of A0 has been modified by geqrtr!"
                  << std::endl;

    // check that the lower part of Q0 has not been touched
    touched = false;
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            if (Q0(i, j) != Q0_check(i, j)) touched = true;
    if (touched)
        std::cout << std::endl
                  << "error, lower part of A0 has been modified by trge_ung2r!"
                  << std::endl;

    // Put zeros below Upper Triangle R in A
    for (idx_t j = 0; j < n - 1; j++)
        for (idx_t i = j + 1; i < n; i++)
            Q0(i, j) = real_t(0);

    std::vector<T> CC_;
    auto CC = new_matrix(CC_, n + m, l);

    std::vector<T> QQ_;
    auto QQ = new_matrix(QQ_, n + m, k);

    std::vector<T> DD_;
    auto DD = new_matrix(DD_, n + m, l);

    auto QQ0 = slice(QQ, range{0, n}, range{0, k});
    auto QQ1 = slice(QQ, range{n, n + m}, range{0, k});
    lacpy(GENERAL, Q0, QQ0);
    lacpy(GENERAL, Q1, QQ1);

    auto CC0 = slice(CC, range{0, n}, range{0, l});
    auto CC1 = slice(CC, range{n, n + m}, range{0, l});
    lacpy(GENERAL, C0, CC0);
    lacpy(GENERAL, C1, CC1);

    auto DD0 = slice(DD, range{0, n}, range{0, l});
    auto DD1 = slice(DD, range{n, n + m}, range{0, l});
    lacpy(GENERAL, D0, DD0);
    lacpy(GENERAL, D1, DD1);

    auto normC = lange(FROB_NORM, CC);

    gemm(Op::ConjTrans, Op::NoTrans, real_t(1.), QQ, DD, real_t(-1.), CC);

    std::cout << lange(FROB_NORM, CC) / normC;

    /////////////////////////// test start here
    std::vector<T> AA_;
    auto AA = new_matrix(AA_, n + m, n);

    auto AA0 = slice(AA, range{0, n}, range{0, n});
    auto AA1 = slice(AA, range{n, n + m}, range{0, n});
    lacpy(GENERAL, A0, AA0);
    lacpy(GENERAL, A1, AA1);

    auto QQthin = slice(QQ, range{0, m + n}, range{0, n});

    // // Apply trge_unm2r()
    // std::vector<T> I_;
    // auto I = new_matrix(I_, n + m, k);

    // laset(GENERAL, real_t(0.), real_t(1.), I);

    // // Compute (H1*H2*...*Hk)*I = Q -> I
    // trge_unm2r(Qcopy, tau, I);

    // for (idx_t j = 0; j < n - 1; j++)
    //     for (idx_t i = j + 1; i < n; i++)
    //         Qcopy(i, j) = real_t(0);

    // // Compute Q - (H1*H2*...*Hk)*I -> Qcopy
    // for (idx_t j = 0; j < k; j++)
    //     for (idx_t i = 0; i < n + m; i++)
    //         Qcopy(i, j) -= I(i, j);

    real_t norm_orth, norm_repres;

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
        norm_orth = lanhe(FROB_NORM, UPPER_TRIANGLE, work);
    }

    // 3) Compute ||QR - A||_F / ||A||_F

    {
        for (idx_t j = 0; j < n; j++)
            for (idx_t i = j + 1; i < n; i++)
                AA(i, j) = real_t(0);

        // // Frobenius norm of A
        auto normA = lange(FROB_NORM, AA);

        std::vector<T> work_;
        auto work = new_matrix(work_, m + n, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m + n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // Copy Q1 to work
        auto QQthin = slice(QQ, range{0, n + m}, range{0, n});

        lacpy(GENERAL, QQthin, work);

        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, real_t(1.0),
             R, work);
        std::cout << std::endl;

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m + n; ++i)
                work(i, j) -= AA(i, j);

        norm_repres = lange(FROB_NORM, work) / normA;
    }

    // // *) Output

    std::cout << std::endl;
    std::cout << "||QR - A||_F/||A||_F = " << norm_repres
              << "\n||Qᴴ Q - I||_F = " << norm_orth;
    //   << "\n||Q - (H1*H2*...*Hk)*I||_F = " << norm_Qcopy;
    std::cout << std::endl;
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
