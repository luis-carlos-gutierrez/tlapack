#include "tlapack/base/utils.hpp"

using namespace tlapack;

template <TLAPACK_SMATRIX matrixA_t,
          TLAPACK_VECTOR tau_t,
          TLAPACK_SMATRIX matrixC0_t,
          TLAPACK_SMATRIX matrixC1_t>
void trge_unm2r(const matrixA_t& A,
                const tau_t& tau,
                matrixC0_t& C0,
                matrixC1_t& C1)
{
    using idx_t = size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrixA_t>;
    using real_t = real_type<T>;

    // Functors for creating new matrices
    Create<matrixA_t> new_matrix;

    idx_t n = nrows(C0);
    idx_t m = nrows(C1);
    idx_t k = ncols(C0);

    std::vector<T> work_;
    auto work = new_matrix(work_, k, 1);

    for (idx_t i = 0; i < n; ++i) {
        auto v = slice(A, range{0, m}, i);
        auto C0_view = slice(C0, i, range{0, k});
        auto C1_view = slice(C1, range{0, m}, range{0, k});

        larf_work(LEFT_SIDE, COLUMNWISE_STORAGE, v, conj(tau[i]), C0_view,
                  C1_view, work);
    }
}