#include "sheaf.hpp"

#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

// An OpenMP directive that vanishes from a serial build, where a bare #pragma omp would warn.
#ifdef _OPENMP
#define SHEAF_OMP(directive) _Pragma(#directive)
#else
#define SHEAF_OMP(directive)
#endif

namespace sheaf {
namespace {

using Eigen::Index;

// Runs fn with the stalk dimension as a compile-time constant for the small d used in practice,
// so Eigen unrolls the d x d products; any other d runs the Eigen::Dynamic instantiation.
template <class Fn>
void dispatch(int d, Fn&& fn) {
    switch (d) {
        case 1: fn(std::integral_constant<int, 1>{}); break;
        case 2: fn(std::integral_constant<int, 2>{}); break;
        case 3: fn(std::integral_constant<int, 3>{}); break;
        case 4: fn(std::integral_constant<int, 4>{}); break;
        default: fn(std::integral_constant<int, Eigen::Dynamic>{}); break;
    }
}

// The d rows of cochain x over cell i.
template <int D, class X>
auto stalk(X& x, Index i, int d) {
    return x.template middleRows<D>(i * d, d);
}

// The d x d row-major matrix at p, writable unless T is const.
template <int D, class T>
auto square(T* p, int d) {
    using M = Eigen::Matrix<std::remove_const_t<T>, D, D, Eigen::RowMajor>;
    return Eigen::Map<std::conditional_t<std::is_const_v<T>, const M, M>>(p, d, d);
}

// Edge ids grouped by vertex: ids[ptr[v] .. ptr[v + 1]) are the e with endpoint[e] == v, ascending.
void group_by_vertex(const std::vector<int>& endpoint, int n, std::vector<int>& ptr,
                     std::vector<int>& ids) {
    ptr.assign(n + 1, 0);
    for (int v : endpoint) ++ptr[v + 1];
    std::partial_sum(ptr.begin(), ptr.end(), ptr.begin());
    std::vector<int> next(ptr.begin(), ptr.end() - 1);
    ids.resize(endpoint.size());
    for (int e = 0; e < static_cast<int>(endpoint.size()); ++e) ids[next[endpoint[e]]++] = e;
}

}  // namespace

template <class T>
Coboundary<T>::Coboundary(int num_vertices, int d, std::span<const Edge> edges)
    : n_(num_vertices), d_(d) {
    if (num_vertices < 0 || d < 1 || edges.size() > std::numeric_limits<int>::max() / 2)
        throw std::invalid_argument("sheaf::Coboundary: invalid sizes");
    tail_.reserve(edges.size());
    head_.reserve(edges.size());
    for (const Edge& edge : edges) {
        if (edge.tail < 0 || edge.tail >= n_ || edge.head < 0 || edge.head >= n_ ||
            edge.tail == edge.head)
            throw std::invalid_argument(
                "sheaf::Coboundary: edge endpoints must be distinct vertices in [0, num_vertices)");
        tail_.push_back(edge.tail);
        head_.push_back(edge.head);
    }
    group_by_vertex(head_, n_, in_ptr_, in_edges_);
    group_by_vertex(tail_, n_, out_ptr_, out_edges_);
}

template <class T>
void Coboundary<T>::apply(const In& maps, const In& x, Out y) const {
    const Index m = num_edges();
    const int d = d_;
    assert(maps.rows() == 2 * m && maps.cols() == d * d);
    assert(x.rows() == Index(n_) * d && y.rows() == m * d && y.cols() == x.cols());
    dispatch(d, [&]<int D>(std::integral_constant<int, D>) {
        SHEAF_OMP(omp parallel for schedule(static))
        for (Index e = 0; e < m; ++e) {
            stalk<D>(y, e, d).noalias() =
                square<D>(maps.row(2 * e + 1).data(), d).lazyProduct(stalk<D>(x, head_[e], d)) -
                square<D>(maps.row(2 * e).data(), d).lazyProduct(stalk<D>(x, tail_[e], d));
        }
    });
}

// A scatter over edges would race on shared endpoints, so each vertex gathers from its incident
// edges instead: every stalk of x has one writer, and the summation order is fixed.
//
// Each x_v += F^T y_e is written as x_v^T += y_e^T F. Spelled directly, the column-major F^T makes
// Eigen evaluate the product column-major, which it cannot store vectorized into the row-major x_v,
// so it goes scalar. Transposed, it vectorizes along f (about 2.5x faster for d > 4).
template <class T>
void Coboundary<T>::apply_transpose(const In& maps, const In& y, Out x) const {
    const Index m = num_edges();
    const int d = d_;
    assert(maps.rows() == 2 * m && maps.cols() == d * d);
    assert(y.rows() == m * d && x.rows() == Index(n_) * d && x.cols() == y.cols());
    dispatch(d, [&]<int D>(std::integral_constant<int, D>) {
        SHEAF_OMP(omp parallel for schedule(dynamic, 256))
        for (Index v = 0; v < n_; ++v) {
            auto xv = stalk<D>(x, v, d).transpose();
            xv.setZero();
            for (int k = in_ptr_[v]; k < in_ptr_[v + 1]; ++k) {
                const Index e = in_edges_[k];
                xv.noalias() += stalk<D>(y, e, d).transpose().lazyProduct(
                    square<D>(maps.row(2 * e + 1).data(), d));
            }
            for (int k = out_ptr_[v]; k < out_ptr_[v + 1]; ++k) {
                const Index e = out_edges_[k];
                xv.noalias() -= stalk<D>(y, e, d).transpose().lazyProduct(
                    square<D>(maps.row(2 * e).data(), d));
            }
        }
    });
}

template <class T>
void Coboundary<T>::map_gradient(const In& x, const In& gy, Out gmaps, bool accumulate) const {
    const Index m = num_edges();
    const int d = d_;
    assert(x.rows() == Index(n_) * d && gy.rows() == m * d && gy.cols() == x.cols());
    assert(gmaps.rows() == 2 * m && gmaps.cols() == d * d);
    dispatch(d, [&]<int D>(std::integral_constant<int, D>) {
        SHEAF_OMP(omp parallel for schedule(static))
        for (Index e = 0; e < m; ++e) {
            const auto ge = stalk<D>(gy, e, d);
            auto g_head = square<D>(gmaps.row(2 * e + 1).data(), d);
            auto g_tail = square<D>(gmaps.row(2 * e).data(), d);
            if (accumulate) {
                g_head.noalias() += ge.lazyProduct(stalk<D>(x, head_[e], d).transpose());
                g_tail.noalias() -= ge.lazyProduct(stalk<D>(x, tail_[e], d).transpose());
            } else {
                g_head.noalias() = ge.lazyProduct(stalk<D>(x, head_[e], d).transpose());
                g_tail.noalias() = -ge.lazyProduct(stalk<D>(x, tail_[e], d).transpose());
            }
        }
    });
}

// Expanding δ^T δ at vertex v, with u the other endpoint of e, gives L's diagonal and off-diagonal
// blocks:
//   (Lx)_v = D_v x_v - Σ_{e ∋ v} F_{v ⊴ e}^T F_{u ⊴ e} x_u,   D_v = Σ_{e ∋ v} F_{v ⊴ e}^T F_{v ⊴ e}.
// Forming each d x d block costs d^3, small next to the d x f products, and the (m*d) x f cochain
// δx is never written or read back. As in apply_transpose, every stalk of lx has one writer.
//
// The d x d products are plain loops. As lazy products, Eigen evaluates each entry of a runtime-
// sized result as its own reduction, whose setup outweighs the d multiply-adds: 1.6x slower for
// d = 5..8, and no faster for the unrolled sizes.
template <class T>
void Coboundary<T>::laplacian_apply(const In& maps, const In& x, Out lx) const {
    const Index m = num_edges();
    const int d = d_;
    assert(maps.rows() == 2 * m && maps.cols() == d * d);
    assert(x.rows() == Index(n_) * d && lx.rows() == x.rows() && lx.cols() == x.cols());
    dispatch(d, [&]<int D>(std::integral_constant<int, D>) {
        SHEAF_OMP(omp parallel)
        {
            Eigen::Matrix<T, D, D, Eigen::RowMajor> diag(d, d), block(d, d);
            SHEAF_OMP(omp for schedule(dynamic, 256))
            for (Index v = 0; v < n_; ++v) {
                auto lv = stalk<D>(lx, v, d);
                lv.setZero();
                diag.setZero();
                // Edge e as seen from v: F_{v ⊴ e} is row row_v of maps, F_{u ⊴ e} is row row_u.
                const auto add_edge = [&](Index row_v, Index row_u, Index u) {
                    const auto Fv = square<D>(maps.row(row_v).data(), d);
                    const auto Fu = square<D>(maps.row(row_u).data(), d);
                    for (Index i = 0; i < Fv.rows(); ++i) {
                        for (Index j = 0; j < Fv.rows(); ++j) {
                            T vv = 0, vu = 0;
                            for (Index k = 0; k < Fv.rows(); ++k) {
                                vv += Fv(k, i) * Fv(k, j);
                                vu += Fv(k, i) * Fu(k, j);
                            }
                            diag(i, j) += vv;
                            block(i, j) = vu;
                        }
                    }
                    lv.noalias() -= block.lazyProduct(stalk<D>(x, u, d));
                };
                for (int k = in_ptr_[v]; k < in_ptr_[v + 1]; ++k) {
                    const Index e = in_edges_[k];
                    add_edge(2 * e + 1, 2 * e, tail_[e]);
                }
                for (int k = out_ptr_[v]; k < out_ptr_[v + 1]; ++k) {
                    const Index e = out_edges_[k];
                    add_edge(2 * e, 2 * e + 1, head_[e]);
                }
                lv.noalias() += diag.lazyProduct(stalk<D>(x, v, d));
            }
        }
    });
}

template <class T>
Eigen::SparseMatrix<T, Eigen::RowMajor> Coboundary<T>::to_sparse(const In& maps) const {
    const Index m = num_edges();
    const int d = d_;
    assert(maps.rows() == 2 * m && maps.cols() == d * d);
    std::vector<Eigen::Triplet<T>> entries;
    entries.reserve(2 * m * d * d);
    for (Index e = 0; e < m; ++e) {
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                entries.emplace_back(e * d + i, head_[e] * d + j, maps(2 * e + 1, i * d + j));
                entries.emplace_back(e * d + i, tail_[e] * d + j, -maps(2 * e, i * d + j));
            }
        }
    }
    Eigen::SparseMatrix<T, Eigen::RowMajor> delta(m * d, Index(n_) * d);
    delta.setFromTriplets(entries.begin(), entries.end());
    return delta;
}

template class Coboundary<float>;
template class Coboundary<double>;

}  // namespace sheaf
