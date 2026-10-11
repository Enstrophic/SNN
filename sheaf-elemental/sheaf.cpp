#include "sheaf.hpp"

#include <cassert>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(dynamic, 256)
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
void Coboundary<T>::map_gradient(const In& x, const In& gy, Out gmaps) const {
    const Index m = num_edges();
    const int d = d_;
    assert(x.rows() == Index(n_) * d && gy.rows() == m * d && gy.cols() == x.cols());
    assert(gmaps.rows() == 2 * m && gmaps.cols() == d * d);
    dispatch(d, [&]<int D>(std::integral_constant<int, D>) {
#pragma omp parallel for schedule(static)
        for (Index e = 0; e < m; ++e) {
            const auto ge = stalk<D>(gy, e, d);
            square<D>(gmaps.row(2 * e + 1).data(), d).noalias() =
                ge.lazyProduct(stalk<D>(x, head_[e], d).transpose());
            square<D>(gmaps.row(2 * e).data(), d).noalias() =
                -ge.lazyProduct(stalk<D>(x, tail_[e], d).transpose());
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
