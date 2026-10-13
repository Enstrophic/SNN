#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <span>
#include <vector>

namespace sheaf {

template <class T>
using RowMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// An edge with an arbitrary but fixed orientation tail -> head. Store each undirected edge once;
// listing both u -> v and v -> u counts it twice in the Laplacian.
struct Edge {
    int tail;
    int head;
};

// The coboundary of a cellular sheaf with stalks R^d over a fixed graph with n vertices and m edges,
//   (δx)_e = F_{head(e) ⊴ e} x_head(e) - F_{tail(e) ⊴ e} x_tail(e).
// The restriction maps are an argument, not state, since a sheaf network predicts them afresh on
// every forward pass. Arrays are row-major, as RowMat<T> or a view of one with unit inner stride:
//   x      (n*d) x f     0-cochain; rows [v*d, v*d + d) are the stalk at vertex v
//   y      (m*d) x f     1-cochain; rows [e*d, e*d + d) are the stalk at edge e
//   maps   (2m) x (d*d)  row 2e is F_{tail(e) ⊴ e} and row 2e + 1 is F_{head(e) ⊴ e}, each
//                        flattened row-major; map gradients have the same layout
// f is the number of feature channels. Outputs must be presized and must not share memory with an
// input, so in-place calls such as laplacian_apply(maps, x, x) are not allowed; debug builds assert
// this. Instantiated for float and double.
template <class T>
class Coboundary {
public:
    using In = Eigen::Ref<const RowMat<T>>;
    using Out = Eigen::Ref<RowMat<T>>;

    // Throws std::invalid_argument on a self-loop or an endpoint outside [0, num_vertices).
    Coboundary(int num_vertices, int d, std::span<const Edge> edges);

    int num_vertices() const { return n_; }
    int num_edges() const { return static_cast<int>(tail_.size()); }
    int stalk_dim() const { return d_; }

    // y = δx.
    void apply(const In& maps, const In& x, Out y) const;

    // x = δ^T y. For y = δx this is also the backward pass to x: dLoss/dx = δ^T dLoss/dy.
    void apply_transpose(const In& maps, const In& y, Out x) const;

    // Backward pass to the maps for y = δx, given gy = dLoss/dy:
    //   dLoss/dF_{head(e) ⊴ e} = gy_e x_head(e)^T,   dLoss/dF_{tail(e) ⊴ e} = -gy_e x_tail(e)^T.
    // By default this overwrites gmaps. With accumulate = true it adds to gmaps instead, so when
    // the same maps are used more than once, as in δ^T δ, their contributions sum into one buffer.
    void map_gradient(const In& x, const In& gy, Out gmaps, bool accumulate = false) const;

    // lx = Lx for the sheaf Laplacian L = δ^T δ, in one pass that never forms δx:
    //   (Lx)_v = Σ_{e ∋ v} F_{v ⊴ e}^T (F_{v ⊴ e} x_v - F_{u ⊴ e} x_u),  u the other endpoint of e.
    // L is symmetric, so the backward pass to x is laplacian_apply(maps, dLoss/dlx, gx). Since
    // ⟨G, Lx⟩ = ⟨δG, δx⟩, the backward pass to the maps given G = dLoss/dlx is map_gradient(G, δx)
    // plus map_gradient(x, δG), the second with accumulate = true. Usually faster than apply then
    // apply_transpose, but not on one thread with small f or for d > 8; sheaf.cpp has the numbers.
    void laplacian_apply(const In& maps, const In& x, Out lx) const;

    // L = δ^T δ as a dense (n*d) x (n*d) matrix, so that applying it is one matrix product run by
    // Eigen's blocked GEMM: lx.noalias() = L * x. Rebuild it whenever the maps change. It holds
    // (n*d)^2 scalars, and a product costs (n*d)^2 f multiply-adds against about 2 m d^2 f for
    // laplacian_apply: n / (average degree) times the arithmetic, which GEMM's speed repays only
    // when L is far from sparse. Measured at d = 3: with 100 vertices of average degree 50 (L half
    // full) the product is 2-3x faster than laplacian_apply, and still about 2x counting assembly
    // over a forward and backward pass; at 250 vertices the two tie, and at 500 vertices, or at
    // average degree 10, laplacian_apply wins every time. The other use is analysis: L comes out
    // exactly symmetric, so Eigen's self-adjoint solvers accept it, e.g. for the sheaf's spectrum.
    // The backward passes are those of laplacian_apply, with L * G as the one to x.
    void dense_laplacian(const In& maps, Out L) const;

    // δ as an explicit (m*d) x (n*d) sparse matrix, for analysis and testing.
    Eigen::SparseMatrix<T, Eigen::RowMajor> to_sparse(const In& maps) const;

private:
    // Calls visit(row_v, row_u, u) for each edge e at vertex v, where rows row_v and row_u of maps
    // hold F_{v ⊴ e} and F_{u ⊴ e} and u is the other endpoint of e.
    template <class Visit>
    void for_each_incident(Eigen::Index v, Visit&& visit) const;

    int n_;
    int d_;
    std::vector<int> tail_, head_;
    std::vector<int> in_ptr_, in_edges_;    // edges with head v: in_edges_[in_ptr_[v] .. in_ptr_[v + 1])
    std::vector<int> out_ptr_, out_edges_;  // edges with tail v, likewise
};

}  // namespace sheaf
