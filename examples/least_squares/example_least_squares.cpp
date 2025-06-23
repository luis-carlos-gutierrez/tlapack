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
#include <tlapack/lapack/unmlq.hpp>
#include <tlapack/lapack/unmql.hpp>
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
void run(size_t m, size_t n, size_t k)
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
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
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
    auto xHat = new_matrix(xHat_, n, k);
    std::vector<T> b_;
    auto b = new_matrix(b_, m, k);
    std::vector<T> bHat_;
    auto bHat = new_matrix(bHat_, m, 1);
    std::vector<T> y_;
    auto y = new_matrix(y_, n, k);
    std::vector<T> tau_;
    auto tau = new_vector(tau_, std::min(m, n));
    std::vector<T> baug_;
    auto baug = new_vector(baug_, m + n);

    // FIX LATER?: Initialize ALL matrices/vectors to zero?
    // Ask Julien ^
    // FIX LATER: Figure out how to randomly seed matrix market
    MatrixMarket mm;

    // std::string option = "Cholesky";
    // std::string option = "QR";
    // std::string option = "SVD0";
    // std::string option = "SVD1";
    std::string option = "SVD2";
    // std::string option = "Tikhonov";

    // Initializing A randomly
    mm.random(A);

    // Create a copy of A for the check
    lacpy(GENERAL, A, A_copy);

    if (verbose) {
        std::cout << std::endl << "A_copy (randomly initialized)";
        printMatrix(A_copy);
        std::cout << std::endl;
    }

    // Initialize a random vectors b
    mm.random(b);

    if (verbose) {
        std::cout << std::endl << "b =" << std::endl;
        for (idx_t i = 0; i < m; i++)
            std::cout << b(i, 0) << std::endl;
        std::cout << std::endl;
    }

    // Begin solving the ordinary least squares linear regression

    if (option == "Cholesky") {
        // Solving the normal equations using Cholesky Factorization if A.H*A is
        // SPD
        // FIX LATER?: Delete info?

        // Fix later: Initialize H with garbage

        // Fix later?: Get rid of U

        std::cout << std::endl
                  << "Using Cholesky factorization to solve LSq Problem"
                  << std::endl;

        // Compute H := A.H*A
        herk(UPPER_TRIANGLE, CONJ_TRANS, real_t(1), A, H);

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
            std::cout << std::endl << "y initialized =";
            printMatrix(y);
            std::cout << std::endl;
        }

        // Let y~ = A.H*b => U.H*y = y~
        // Compute y~ = A.H*b
        // y~ -> y

        // FIX LATER: use a strong zero
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), A, b, real_t(0), y);

        if (verbose) {
            std::cout << std::endl << "y~ := A.H*b =";
            printMatrix(y);
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
                      << "y =";
            printMatrix(y);
            std::cout << std::endl;
        }

        // Finally solve for xHat in U*xHat = y
        // xHat -> y
        // FIX LATER?: I feel unsure about alpha = real_t(1). Not sure if y
        // should be unnaffedted by the 1
        trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), U,
             y);

        // Write y into xHat for clarity
        lacpy(GENERAL, y, xHat);

        if (verbose) {
            std::cout << std::endl
                      << "After solving for xHat in U*xHat = y," << std::endl
                      << std::endl
                      << "xhat = ";
            printMatrix(xHat);
            std::cout << std::endl;
        }
    }
    else if (option == "QR") {  // A.H*Ais not SPD => Solve Lsq using QR

        // Factor A into A = Q * R (QR factorization stored compactly in AHH and
        // tau)
        lacpy(GENERAL, A, AHH);
        geqrf(AHH, tau);

        // Apply Q^H to b using the compact QR representation
        std::vector<T> xHat2_;
        auto xHat2 = new_matrix(xHat2_, m, k);

        lacpy(GENERAL, b, xHat2);
        unmqr(LEFT_SIDE, CONJ_TRANS, AHH, tau, xHat2);

        // Extract upper triangular R from AHH
        lacpy(UPPER_TRIANGLE, slice(AHH, range{0, n}, range{0, n}), R);

        // Keep only the first n rows of Q^H * b as xHat
        lacpy(GENERAL, slice(xHat2, range{0, n}, range{0, k}), xHat);

        // Solve R * xHat = Q^H * b (now in xHat)
        trsm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, real_t(1), R,
             xHat);

    }  // End QR factorization routine
    else if (option == "SVD0") {
        std::vector<T> tauv(n);
        std::vector<T> tauw(n);

        std::vector<T> Q1_;
        auto Q1 = new_matrix(Q1_, m, n);
        std::vector<T> P1_;
        auto P1 = new_matrix(P1_, n, n);

        // Bidiagonal decomposition
        bidiag(A, tauv, tauw);

        // Reconstruct Q1 and P1
        lacpy(LOWER_TRIANGLE, slice(A, range{0, m}, range{0, n}), Q1);
        ungbr_q(n, Q1, tauv);

        lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);
        ungbr_p(n, P1, tauw);

        // Diagonal and off-diagonal of bidiagonal matrix
        std::vector<real_t> d(n);
        std::vector<real_t> e(n - 1);
        for (idx_t j = 0; j < n; ++j)
            d[j] = real(A(j, j));
        for (idx_t j = 0; j < n - 1; ++j)
            e[j] = real(A(j, j + 1));

        // Compute SVD of bidiagonal
        std::vector<T> Q2_;
        auto Q2 = new_matrix(Q2_, n, n);
        std::vector<T> P2t_;
        auto P2t = new_matrix(P2t_, n, n);
        const real_t zero(0);
        const real_t one(1);
        laset(Uplo::General, zero, one, Q2);
        laset(Uplo::General, zero, one, P2t);

        int err = svd_qr(Uplo::Upper, true, true, d, e, Q2, P2t);

        // Allocate temporaries for multiple RHS (k columns)
        std::vector<T> xHat2_;
        auto xHat2 = new_matrix(xHat2_, n, k);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q1, b, real_t(0), xHat2);

        std::vector<T> xHat3_;
        auto xHat3 = new_matrix(xHat3_, n, k);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q2, xHat2, real_t(0), xHat3);

        // Scale each row of xHat3 by 1/d[j]
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < k; ++i)
                xHat3(j, i) /= d[j];

        std::vector<T> xHat4_;
        auto xHat4 = new_matrix(xHat4_, n, k);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), P2t, xHat3, real_t(0), xHat4);

        // Final result: xHat = P1ᵀ * P2ᵀ * Σ⁻¹ * Q2ᵀ * Q1ᵀ * b
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, xHat4, real_t(0), xHat);
    }
    else if (option == "SVD1") {
        std::vector<T> tauv(n);
        std::vector<T> tauw(n);

        std::vector<T> Q1_;
        auto Q1 = new_matrix(Q1_, m, n);
        std::vector<T> P1_;
        auto P1 = new_matrix(P1_, n, n);

        // Bidiag decomposition
        bidiag(A, tauv, tauw);

        // Bidiag matrix reconstruction
        lacpy(LOWER_TRIANGLE, slice(A, range{0, m}, range{0, n}), Q1);
        ungbr_q(n, Q1, tauv);

        lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);
        ungbr_p(n, P1, tauw);

        // Diagonal and off-diagonal extraction
        std::vector<real_t> d(n);
        std::vector<real_t> e(n - 1);
        for (idx_t j = 0; j < n; ++j)
            d[j] = real(A(j, j));
        for (idx_t j = 0; j < n - 1; ++j)
            e[j] = real(A(j, j + 1));

        // SVD of bidiagonal matrix
        int err = svd_qr(Uplo::Upper, true, true, d, e, Q1, P1);

        // Apply Q1ᵀ to b
        std::vector<T> xHat2_;
        auto xHat2 = new_matrix(xHat2_, n, k);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q1, b, real_t(0), xHat2);

        // Scale rows by 1/d[j]
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < k; ++i)
                xHat2(j, i) /= d[j];

        // Apply P1ᵀ to scaled result
        std::vector<T> xHat3_;
        auto xHat3 = new_matrix(xHat3_, n, k);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, xHat2, real_t(0), xHat3);

        // Store final solution
        lacpy(GENERAL, xHat3, xHat);
    }
    else if (option == "SVD2") {
        std::vector<T> tauv(n);
        std::vector<T> tauw(n);

        std::vector<T> Q1_;
        auto Q1 = new_matrix(Q1_, m, n);
        std::vector<T> P1_;
        auto P1 = new_matrix(P1_, n, n);

        // Bidiagonal decomposition
        bidiag(A, tauv, tauw);

        // Reconstruct Q1 and P1
        lacpy(LOWER_TRIANGLE, slice(A, range{0, m}, range{0, n}), Q1);
        ungbr_q(n, Q1, tauv);

        lacpy(UPPER_TRIANGLE, slice(A, range{0, n}, range{0, n}), P1);
        ungbr_p(n, P1, tauw);

        // Extract diagonal and superdiagonal
        std::vector<real_t> d(n);
        std::vector<real_t> e(n - 1);
        for (idx_t j = 0; j < n; ++j)
            d[j] = real(A(j, j));
        for (idx_t j = 0; j < n - 1; ++j)
            e[j] = real(A(j, j + 1));

        // Allocate Q2 and P2t
        std::vector<T> Q2_;
        auto Q2 = new_matrix(Q2_, n, n);
        std::vector<T> P2t_;
        auto P2t = new_matrix(P2t_, n, n);
        const real_t zero(0);
        const real_t one(1);
        laset(Uplo::General, zero, one, Q2);
        laset(Uplo::General, zero, one, P2t);

        // Apply Q1ᵀ to b using unmqr
        std::vector<T> btmp1_;
        auto btmp1 = new_matrix(btmp1_, m, k);
        lacpy(GENERAL, b, btmp1);
        unmqr(LEFT_SIDE, CONJ_TRANS, A, tauv, btmp1);

        // Slice top n rows: Q1ᵀ b
        lacpy(GENERAL, slice(btmp1, range{0, n}, range{0, k}), xHat);

        // Apply Q2ᵀ
        std::vector<T> xHat2_;
        auto xHat2 = new_matrix(xHat2_, n, k);
        int err = svd_qr(Uplo::Upper, true, true, d, e, Q2, P2t);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), Q2, xHat, real_t(0), xHat2);

        // Scale each row by 1/d[j]
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < k; ++i)
                xHat2(j, i) /= d[j];

        // Apply P2tᵀ
        std::vector<T> xHat3_;
        auto xHat3 = new_matrix(xHat3_, n, k);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), P2t, xHat2, real_t(0), xHat3);

        // Apply P1ᵀ
        std::vector<T> xHat4_;
        auto xHat4 = new_matrix(xHat4_, n, k);
        gemm(CONJ_TRANS, NO_TRANS, real_t(1), P1, xHat3, real_t(0), xHat4);

        // Final result
        lacpy(GENERAL, xHat4, xHat);
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

    ///////////////////////////// check starts her ////////////////////

    std::cout << std::endl
              << "Check to see if Lsq solved succesfully using " << option
              << " method:" << std::endl;

    // lacpy(GENERAL, A_copy, A);

    // Compute b - A *xHat
    gemm(NO_TRANS, NO_TRANS, real_t(-1), A_copy, xHat, real_t(1), b);

    if (verbose) {
        std::cout << std::endl << "r =";
        printMatrix(b);
        std::cout << std::endl;
    }

    // Compute A.H*(b - A xHat)
    // FIX LATER?: beta = real_t(0)
    gemm(CONJ_TRANS, NO_TRANS, real_t(1), A_copy, b, real_t(0), y);

    if (verbose) {
        std::cout << std::endl << "A.H*(b - bHat) =";
        printMatrix(y);
        std::cout << std::endl;
    }

    double dotProdNorm = lange(FROB_NORM, y);

    double normAcopy = lange(FROB_NORM, A_copy);

    double normb = lange(FROB_NORM, b);

    std::cout << std::endl
              << "(||A.H*(b - bHat)||_F) / (||A||_F) = " << std::endl
              << (dotProdNorm / normAcopy) << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << std::endl
              << "example_template executed" << std::endl
              << std::endl;

    using std::size_t;
    int m, n, k;

    // Default arguments
    m = 3;
    n = 2;
    k = 10;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d, %d )", m, n, k);
    run<float>(m, n, k);
    printf("-----------------------\n");

    printf("run< double >( %d, %d, %d )", m, n, k);
    run<double>(m, n, k);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d, %d )", m, n, k);
    run<long double>(m, n, k);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d, %d )", m, n, k);
    run<std::complex<float>>(m, n, k);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d, %d )", m, n, k);
    run<std::complex<double>>(m, n, k);
    printf("-----------------------\n");
    return 0;
}
