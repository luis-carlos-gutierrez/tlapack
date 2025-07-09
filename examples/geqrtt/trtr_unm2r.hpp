#include "tlapack/base/utils.hpp"

using namespace tlapack;

template <TLAPACK_SMATRIX matrixA1_t,
          TLAPACK_VECTOR tau_t,
          TLAPACK_SMATRIX matrixC0_t,
          TLAPACK_SMATRIX matrixC1_t>
void trtr_unm2r(const matrixA1_t& A1,
                const tau_t& tau,
                matrixC0_t& C0,
                matrixC1_t& C1)
{
    using idx_t = size_type<matrixA1_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrixA1_t>;
    using real_t = real_type<T>;

    // Functors for creating new matrices
    Create<matrixA1_t> new_matrix;

    idx_t n = nrows(C0);
    idx_t k = ncols(C0);

    std::vector<T> work_;
    auto work = new_matrix(work_, k, 1);

    for (idx_t i = 0; i < n; ++i) {
        auto view_A1 = slice(A1, range{0, n}, i);
        auto view_C0 = slice(C0, i, range{0, k});
        auto view_C1 = slice(C1, range{0, n}, range{0, k});

        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, view_A1, conj(tau[i]), view_C0,
                  view_C1, work);
    }
}