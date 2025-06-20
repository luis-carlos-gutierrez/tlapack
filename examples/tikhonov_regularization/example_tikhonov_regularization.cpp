/// @file example_template.cpp
/// @author author(s), University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

#include "../../test/include/MatrixMarket.hpp"
#include "tlapack/base/utils.hpp"

// <T>LAPACK
#include <tlapack/blas/gemv.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/nrm2.hpp>
#include <tlapack/blas/trsv.hpp>
#include <tlapack/lapack/bidiag.hpp>
#include <tlapack/lapack/geqrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/lapack/svd_qr.hpp>
#include <tlapack/lapack/trtri_recursive.hpp>
#include <tlapack/lapack/ungbr.hpp>
#include <tlapack/lapack/ungqr.hpp>
#include <tlapack/lapack/unmqr.hpp>

// C++ headers
// FIX LATER?: Do I need these includes?
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

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

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using vector_t = tlapack::LegacyVector<T>;
    using type_t = tlapack::type_t<T>;

    using range = pair<idx_t, idx_t>;

    // Functors for creating new matrices and vectors
    tlapack::Create<matrix_t> new_matrix;
    tlapack::Create<vector_t> new_vector;

    // Turn it off if m or n are large
    bool verbose = true;

    // Declacre matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> AQR_;
    auto AQR = new_matrix(AQR_, m, n);
    std::vector<T> AHH_;
    auto AHH = new_matrix(AHH_, m, n);
    std::vector<T> H_;
    auto H = new_matrix(H_, n, n);
    std::vector<T> U_;
    auto U = new_matrix(U_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> Gamma_;
    auto Gamma = new_matrix(Gamma_, n, n);
    std::vector<T> Aaug_;
    auto Aaug = new_matrix(Aaug_, m + n, n);

    // Declare vector
    std::vector<T> xHat_;
    auto xHat = new_matrix(xHat_, n, 1);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, 1);
    std::vector<T> bHat_;
    auto bHat = new_matrix(bHat_, m, 1);
    std::vector<T> y_;
    auto y = new_matrix(y_, n, 1);
    std::vector<T> tau_;
    auto tau = new_vector(tau_, std::min(m, n));
    std::vector<T> baug_;
    auto baug = new_vector(baug_, m + n);
    // FIX LATER?: Initialize ALL matrices/vectors to zero?
    // Ask Julien ^
    MatrixMarket mm;

    // std::string option = "Cholesky";
    std::string option = "QR";
    // std::string option = "SVD";
    // std::string option = "Tikhonov";

    // Initializing A randomly
    // FIX LATER: Figure out how to randomly seed matrix market
    mm.random(A);

    if (verbose) {
        std::cout << std::endl << "A (randomly initialized)";
        printMatrix(A);
        std::cout << std::endl;
    }

    // Compute H := A.H*A
    // Strong zero function call
    herk(UPPER_TRIANGLE, CONJ_TRANS, real_t(1), A, H);

    // FIX LATER?: Initialize with 0xDEADBEEF
    if (verbose) {
        std::cout << std::endl << "H (a hermitian matrix := A.H*A =";
        printMatrix(H);
        std::cout << std::endl;
    }

    // Initailize A according to method of solve Lsq problem
    if (option == "Cholesky") {
        // FIX LATER: Insert code to initiliaze A to result in a LS problem that
        // uses Cholesky
    }

    else if (option == "QR") {
        /*
        Copy the 0th column of A into the first column to guarentee that
        the product A.H*A is not positive defnite This will result in the QR
        method of solving LS. This is because a matrix with linearly dependent
        columns will not be positive definite.
        */

        // TALK TO JULIEN ABOUT ACHIEVING RANK DEFICIENY W/0 NUM INSTABILITY

        // for (int i = 0; i < m; ++i)
        //     A_[1 * m + i] = A_[0 * m + i];
    }

    // if (verbose) {
    //     std::cout << std::endl;
    //     std::cout << std::endl
    //               << "A with a copied column (for QR factorization) =";
    //     printMatrix(A);
    //     std::cout << std::endl;
    // }

    // Begin solving the ordinary least squares linear regression

    mm.random(b);

    if (verbose) {
        std::cout << std::endl << "b =" << std::endl;
        for (idx_t i = 0; i < m; i++)
            std::cout << b(i, 0) << std::endl;
        std::cout << std::endl;
    }

    if (option == "Cholesky") {
        // Solving the normal equations using Cholesky Factorization if A.H*A is
        // SPD
        // FIX LATER?: Delete info?

        // Fix later: Initialize H with garbage

        // Fix later?: Get rid of U

        std::cout << std::endl
                  << "Using Cholesky factorization to solve LSq Problem"
                  << std::endl;

        lacpy(UPPER_TRIANGLE, H, U);

        if (verbose) {
            std::cout << std::endl << "U copied from H =";
            printMatrix(U);
            std::cout << std::endl;
        }

        int info = potrf(UPPER_TRIANGLE, U);

        if (verbose) {
            std::cout << std::endl << "U after Cholesky Factorization =";
            printMatrix(U);
            std::cout << std::endl;
        }

        // Solving the normal equations: U.H*U*xHat = A.H*b

        // Let y = U*xHat => U.H*y = A.H*b
        // Now, solve for y

        if (verbose) {
            std::cout << std::endl << "y initialized =" << std::endl;
            for (idx_t i = 0; i < n; i++)
                std::cout << y(i, 0) << std::endl;
            std::cout << std::endl;
        }

        // Let y~ = A.H*b => U.H*y = y~
        // Compute y~ = A.H*b
        // y~ -> y

        // FIX LATER: use a strong zero
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), A, b, real_t(0), y);

        if (verbose) {
            std::cout << std::endl << "y~ := A.H*b =" << std::endl;
            for (idx_t i = 0; i < n; i++)
                std::cout << y(i, 0) << std::endl;
            std::cout << std::endl;
        }

        // Now solve for y in U.H*y = y~
        // y -> y
        // FIX LATER?: I feel unsure about alpha = real_t(1). Not sure if y
        // shoulld be unnafted by the 1
        trsm(LEFT_SIDE, UPPER_TRIANGLE, CONJ_TRANS, NON_UNIT_DIAG, real_t(1), U,
             y);

        if (verbose) {
            std::cout << std::endl
                      << "After solving for y in U.H*y = y~," << std::endl
                      << std::endl
                      << "y =" << std::endl;
            for (idx_t i = 0; i < n; i++)
                std::cout << y(i, 0) << std::endl;
            std::cout << std::endl;
        }

        // Finally solve for xHat in U*xHat = y
        // xHat -> y
        // FIX LATER?: I feel unsure about alpha = real_t(1). Not sure if y
        // should be unnaffedted by the 1
        trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), U,
             y);

        // Write y into xHat for clarity
        for (idx_t i = 0; i < n; i++)
            xHat(i, 0) = y(i, 0);

        if (verbose) {
            std::cout << std::endl
                      << "After solving for xHat in U*xHat = y," << std::endl
                      << std::endl
                      << "xhat = " << std::endl;
            for (idx_t i = 0; i < n; i++)
                std::cout << xHat(i, 0) << std::endl;
            std::cout << std::endl;
        }
    }
    else if (option == "QR") {  // A.H*Ais not SPD => Solve Lsq using QR
        std::cout << std::endl
                  << "Using QR factorization to solve LSq Problem" << std::endl;

        lacpy(GENERAL, A, AHH);

        // Factor A into A = QR := AHH
        geqrf(AHH, tau);

        if (verbose) {
            std::cout << std::endl << "A after geqrf() := AHH =";
            printMatrix(AHH);
            std::cout << std::endl;
        }

        // Copy A_HH -> Q to produce Q
        lacpy(GENERAL, AHH, Q);

        if (verbose) {
            std::cout << std::endl << "AHH copied into Q =";
            printMatrix(Q);
            std::cout << std::endl;
        }

        // Compute Q from A_HH
        ungqr(Q, tau);

        if (verbose) {
            std::cout << std::endl << "Q =";
            printMatrix(Q);
            std::cout << std::endl;
        }

        // Check to see if QR factorization was done correctly
        // Compute Q*R to see if result equals A

        // Extract R from AHH
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i <= j && i < m; ++i)
                R(i, j) = AHH(i, j);  // Copy upper triangle only

        if (verbose) {
            std::cout << std::endl << "R =";
            printMatrix(R);
            std::cout << std::endl;
        }

        // Compute Q*R
        // FIX LATER?: Use gemmtr to save flops once gemmtr merged
        // FIX LATER: Strong zero?
        gemm(NO_TRANS, NO_TRANS, real_t(1), Q, R, real_t(0), AQR);

        if (verbose) {
            std::cout << std::endl << "AQR := Q*R =";
            printMatrix(AQR);
            std::cout << std::endl;
        }

        // Solve the normal equations Q*R*xhat = b
        // => R*xHat = Q.H*b

        // FIX LATER?: Can I store the result in b?
        // FIX LATER?: For all alpha in all gemm func and similar, real_t(1) ->
        // T(1) Ask Julien ^
        // FIX LATER?: Is this the strong zero call?

        // Q.H*b -> xHat
        // gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q, b, xHat);
        // FIX LATER please use UNMQR

        std::vector<T> xHat2_;
        auto xHat2 = new_matrix(xHat2_, m, 1);
        lacpy(GENERAL, b, xHat2);
        unmqr(LEFT_SIDE, CONJ_TRANS, AHH, tau, xHat2);

        lacpy(GENERAL, slice(xHat2, range{0, n}, range{0, 1}), xHat);

        // Solve for xHat in R*xHat = Q.H*b
        trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), R,
             xHat);

    }  // End QR factorization routine
    else if (option == "SVD") {  // A.H*Ais not SPD => Solve Lsq using QR
        std::vector<T> tauv(n);
        std::vector<T> tauw(n);

        std::vector<T> Q1_;
        auto Q1 = new_matrix(Q1_, m, n);
        std::vector<T> P1_;
        auto P1 = new_matrix(P1_, n, n);

        std::vector<T> A_copy_;
        auto A_copy = new_matrix(A_copy_, m, n);

        lacpy(GENERAL, A, A_copy);
        bidiag(A, tauv, tauw);

        lacpy(LOWER_TRIANGLE, slice(A, range{0, m}, range{0, n}), Q1);
        ungbr_q(n, Q1, tauv);

        lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);
        ungbr_p(m, P1, tauw);

        std::vector<real_t> d(n);
        std::vector<real_t> e(n - 1);

        for (idx_t j = 0; j < n; ++j)
            d[j] = real(A(j, j));

        for (idx_t j = 0; j < n - 1; ++j)
            e[j] = real(A(j, j + 1));

        std::vector<T> Q2_;
        auto Q2 = new_matrix(Q2_, n, n);
        std::vector<T> P2t_;
        auto P2t = new_matrix(P2t_, n, n);
        const real_t zero(0);
        const real_t one(1);
        laset(Uplo::General, zero, one, Q2);
        laset(Uplo::General, zero, one, P2t);

        int err = svd_qr(Uplo::Upper, true, true, d, e, Q2, P2t);

        gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q1, b, xHat);
        std::vector<T> xHat2_;
        auto xHat2 = new_matrix(xHat2_, n, 1);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q2, xHat, xHat2);

        for (idx_t j = 0; j < n; ++j)
            xHat2(j, 0) /= d[j];

        std::vector<T> xHat3_;
        auto xHat3 = new_matrix(xHat3_, n, 1);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), P2t, xHat2, xHat3);

        std::vector<T> xHat4_;
        auto xHat4 = new_matrix(xHat4_, n, 1);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, xHat3, xHat4);

        lacpy(GENERAL, xHat4, xHat);

        lacpy(GENERAL, A_copy, A);
    }

    else {  // option == Tiknonov
        // Begin tikhonov regularization
        std::cout << std::endl
                  << "Using Tikhonov Regularization to solve Lsq Problem:"
                  << std::endl;

        // Initialize Gamma
        // type_t gammaScal;
        // for(idx_t
    }  // End of Tikhonov Regularization routine

    // Check to see if Lsq regression was implemented succesfully

    std::cout << std::endl
              << "Check to see if Lsq solved succesfully using " << option
              << " method:" << std::endl;

    // Compute A*xHat = bHat
    gemm(NO_TRANS, NO_TRANS, real_t(1), A, xHat, real_t(0), bHat);

    if (verbose) {
        std::cout << std::endl << "bHat =" << std::endl;
        for (idx_t i = 0; i < m; i++)
            std::cout << bHat(i, 0) << std::endl;
        std::cout << std::endl;
    }

    // Compute b - bHat -> b
    for (idx_t i = 0; i < m; i++)
        b(i, 0) -= bHat(i, 0);

    if (verbose) {
        std::cout << std::endl << "b - bHat =" << std::endl;
        for (idx_t i = 0; i < m; i++)
            std::cout << b(i, 0) << std::endl;
        std::cout << std::endl;
    }

    // Compute A.H*(b - bHat)
    // FIX LATER?: beta = real_t(0)
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), A, b, real_t(0), y);

    if (verbose) {
        std::cout << std::endl << "A.H*(b - bHat) =" << std::endl;
        for (idx_t i = 0; i < n; i++)
            std::cout << y(i, 0) << std::endl;
        std::cout << std::endl;
    }

    double dotProdNorm = lange(FROB_NORM, y);

    double normA = lange(FROB_NORM, A);

    double normb = lange(FROB_NORM, b);

    std::cout << std::endl
              << "(||A.H*(b - bHat)||_F) / (||A||_F*||b||_F) = " << std::endl
              << (dotProdNorm / (normA * normb)) << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << std::endl
              << "example_template executed" << std::endl
              << std::endl;

    using std::size_t;
    int m, n;

    // Default arguments
    m = 3;
    n = 2;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", m, n);
    run<long double>(m, n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )", m, n);
    run<std::complex<float>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )", m, n);
    run<std::complex<double>>(m, n);
    printf("-----------------------\n");
    return 0;
}
