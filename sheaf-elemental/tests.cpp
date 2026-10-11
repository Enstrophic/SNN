// Build and run:
//   g++ -std=c++20 -O3 -march=native -fopenmp -isystem /usr/include/eigen3 sheaf.cpp tests.cpp -o tests && ./tests

#include "sheaf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using sheaf::Edge;
using Mat = sheaf::RowMat<double>;
using Delta = sheaf::Coboundary<double>;

int failures = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                                   \
        }                                                                                 \
    } while (0)

bool near(const Mat& a, const Mat& b, double tol = 1e-10) {
    return (a - b).norm() <= tol * (1.0 + b.norm());
}

// Two triangles sharing vertex 2: connected, 5 vertices, 6 edges, two independent cycles.
constexpr int n = 5;
const std::vector<Edge> bowtie = {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 4}, {4, 2}};

// Restriction map k as a writable d x d matrix.
Eigen::Map<Mat> map_at(Mat& maps, int k, int d) { return Eigen::Map<Mat>(maps.row(k).data(), d, d); }

Mat identity_maps(int m, int d) {
    Mat maps(2 * m, d * d);
    for (int k = 0; k < 2 * m; ++k) map_at(maps, k, d).setIdentity();
    return maps;
}

Mat apply(const Delta& delta, const Mat& maps, const Mat& x) {
    Mat y(delta.num_edges() * delta.stalk_dim(), x.cols());
    delta.apply(maps, x, y);
    return y;
}

Mat apply_transpose(const Delta& delta, const Mat& maps, const Mat& y) {
    Mat x(delta.num_vertices() * delta.stalk_dim(), y.cols());
    delta.apply_transpose(maps, y, x);
    return x;
}

Mat map_gradient(const Delta& delta, const Mat& x, const Mat& gy) {
    Mat gmaps(2 * delta.num_edges(), delta.stalk_dim() * delta.stalk_dim());
    delta.map_gradient(x, gy, gmaps);
    return gmaps;
}

// dim ker A, from a rank-revealing LU.
int nullity(const Mat& A) {
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
    lu.setThreshold(1e-9);
    return int(lu.dimensionOfKernel());
}

// With d = 1 and identity maps the sheaf is just the graph: δ is the signed incidence matrix
// (+1 at the head, -1 at the tail) and δ^T δ is the graph Laplacian D - A.
void test_constant_sheaf_is_graph() {
    const Delta delta(n, 1, bowtie);
    const int m = delta.num_edges();
    Mat incidence = Mat::Zero(m, n);
    Mat laplacian = Mat::Zero(n, n);
    for (int e = 0; e < m; ++e) {
        const int t = bowtie[e].tail, h = bowtie[e].head;
        incidence(e, h) = 1;
        incidence(e, t) = -1;
        laplacian(t, t) += 1;
        laplacian(h, h) += 1;
        laplacian(t, h) -= 1;
        laplacian(h, t) -= 1;
    }
    const Mat dense = delta.to_sparse(identity_maps(m, 1)).toDense();
    CHECK(near(dense, incidence));
    CHECK(near(dense.transpose() * dense, laplacian));
}

// Every dispatch path (d = 1..4 unrolled, 5 and 6 dynamic) agrees with the assembled matrix,
// including feature counts that are not a multiple of the SIMD width.
void test_kernels_match_matrix() {
    for (int d = 1; d <= 6; ++d) {
        const Delta delta(n, d, bowtie);
        const Mat maps = Mat::Random(2 * delta.num_edges(), d * d);
        const Mat dense = delta.to_sparse(maps).toDense();
        for (int f : {1, 3, 8, 17}) {
            const Mat x = Mat::Random(n * d, f);
            const Mat y = Mat::Random(delta.num_edges() * d, f);
            CHECK(near(apply(delta, maps, x), dense * x));
            CHECK(near(apply_transpose(delta, maps, y), dense.transpose() * y));
        }
    }
}

// Inputs and outputs may be views with an outer stride, such as a column range of a wider matrix,
// and nothing outside an output view is written.
void test_strided_views() {
    const int d = 3, f = 5;
    const Delta delta(n, d, bowtie);
    const int m = delta.num_edges();
    const Mat maps_buf = Mat::Random(2 * m, d * d + 2);
    const Mat x_buf = Mat::Random(n * d, f + 3);
    const Mat y_buf = Mat::Random(m * d, f + 3);
    Mat y_out = Mat::Zero(m * d, f + 2), x_out = Mat::Zero(n * d, f + 2);
    Mat g_out = Mat::Zero(2 * m, d * d + 2);

    delta.apply(maps_buf.leftCols(d * d), x_buf.middleCols(1, f), y_out.middleCols(1, f));
    delta.apply_transpose(maps_buf.leftCols(d * d), y_buf.middleCols(2, f), x_out.middleCols(1, f));
    delta.map_gradient(x_buf.middleCols(1, f), y_buf.middleCols(2, f), g_out.middleCols(1, d * d));

    const Mat maps = maps_buf.leftCols(d * d), x = x_buf.middleCols(1, f), y = y_buf.middleCols(2, f);
    CHECK(near(y_out.middleCols(1, f), apply(delta, maps, x)));
    CHECK(near(x_out.middleCols(1, f), apply_transpose(delta, maps, y)));
    CHECK(near(g_out.middleCols(1, d * d), map_gradient(delta, x, y)));
    for (const Mat* out : {&y_out, &x_out, &g_out})
        CHECK(out->col(0).isZero(0) && out->col(out->cols() - 1).isZero(0));
}

// Gauge-transform the constant sheaf: F_{v ⊴ e} = Q_e P_v^{-1}. Then x_v = P_v c is a global
// section for every c in R^d, and on a connected graph there are no others, so dim ker δ = d.
// A generic sheaf on this graph has nontrivial holonomy around the cycles and no sections.
void test_global_sections() {
    const int d = 3, f = 2;
    const Delta delta(n, d, bowtie);
    const int m = delta.num_edges();
    std::vector<Mat> P(n);
    for (Mat& Pv : P) Pv = Mat::Random(d, d) + 2.0 * d * Mat::Identity(d, d);
    Mat maps(2 * m, d * d);
    for (int e = 0; e < m; ++e) {
        const Mat Q = Mat::Random(d, d) + 2.0 * d * Mat::Identity(d, d);
        map_at(maps, 2 * e, d) = Q * P[bowtie[e].tail].inverse();
        map_at(maps, 2 * e + 1, d) = Q * P[bowtie[e].head].inverse();
    }
    const Mat c = Mat::Random(d, f);
    Mat x(n * d, f);
    for (int v = 0; v < n; ++v) x.middleRows(v * d, d) = P[v] * c;
    CHECK(apply(delta, maps, x).norm() < 1e-10 * x.norm());
    CHECK(nullity(delta.to_sparse(maps).toDense()) == d);
    CHECK(nullity(delta.to_sparse(Mat::Random(2 * m, d * d)).toDense()) == 0);
}

// Reversing an edge (swapping its endpoints and their maps) negates that edge's rows of δ and
// leaves δ^T δ unchanged, so the choice of orientation never reaches the Laplacian.
void test_orientation_independence() {
    const int d = 3;
    const Delta delta(n, d, bowtie);
    const Mat maps = Mat::Random(2 * delta.num_edges(), d * d);
    std::vector<Edge> flipped = bowtie;
    Mat flipped_maps = maps;
    for (int e = 0; e < delta.num_edges(); e += 2) {
        std::swap(flipped[e].tail, flipped[e].head);
        flipped_maps.row(2 * e).swap(flipped_maps.row(2 * e + 1));
    }
    const Mat dF = delta.to_sparse(maps).toDense();
    const Mat dG = Delta(n, d, flipped).to_sparse(flipped_maps).toDense();
    for (int e = 0; e < delta.num_edges(); ++e) {
        const double sign = e % 2 == 0 ? -1.0 : 1.0;
        CHECK(near(dG.middleRows(e * d, d), sign * dF.middleRows(e * d, d)));
    }
    CHECK(near(dG.transpose() * dG, dF.transpose() * dF));
}

// The largest gap between grad and the central difference of loss in each entry of p, which loss
// must read. Exact up to rounding for any step h when loss is at most quadratic in each entry.
template <class Loss>
double gradient_error(const Loss& loss, Mat& p, const Mat& grad, double h) {
    double worst = 0;
    for (Eigen::Index i = 0; i < p.size(); ++i) {
        const double p0 = p(i);
        p(i) = p0 + h;
        const double up = loss();
        p(i) = p0 - h;
        const double down = loss();
        p(i) = p0;
        worst = std::max(worst, std::abs((up - down) / (2 * h) - grad(i)));
    }
    return worst;
}

// Gradient check on the sheaf Dirichlet energy E = ½‖δx‖², whose gradient with respect to δx is
// δx itself. E is quadratic in each parameter, so central differences are exact up to rounding.
void test_gradients() {
    for (int d : {2, 5}) {  // one unrolled and one dynamic stalk dimension
        const int f = 3;
        const Delta delta(n, d, bowtie);
        Mat maps = Mat::Random(2 * delta.num_edges(), d * d);
        Mat x = Mat::Random(n * d, f);
        const Mat y = apply(delta, maps, x);
        const Mat gx = apply_transpose(delta, maps, y);
        const Mat gmaps = map_gradient(delta, x, y);

        const auto energy = [&] { return 0.5 * apply(delta, maps, x).squaredNorm(); };
        CHECK(gradient_error(energy, x, gx, 1e-3) < 1e-8);
        CHECK(gradient_error(energy, maps, gmaps, 1e-3) < 1e-8);
    }
}

// The upstream gradient δx above lies in the image of δ, a proper subspace of the 1-cochains. A
// random G reaches every direction. ⟨G, δx⟩ is linear in each parameter, so even a unit step is
// exact up to rounding.
void test_gradients_linear() {
    for (int d : {2, 5}) {
        const int f = 3;
        const Delta delta(n, d, bowtie);
        Mat maps = Mat::Random(2 * delta.num_edges(), d * d);
        Mat x = Mat::Random(n * d, f);
        const Mat G = Mat::Random(delta.num_edges() * d, f);
        const Mat gx = apply_transpose(delta, maps, G);
        const Mat gmaps = map_gradient(delta, x, G);

        const auto loss = [&] { return G.cwiseProduct(apply(delta, maps, x)).sum(); };
        CHECK(gradient_error(loss, x, gx, 1.0) < 1e-12);
        CHECK(gradient_error(loss, maps, gmaps, 1.0) < 1e-12);
    }
}

// ⟨H, δ^T δ x⟩ uses every map twice. The δ step y = δx contributes map_gradient(x, gy) with
// gy = δH. The δ^T step z = δ^T y adds F^T y_e at the head and -F^T y_e at the tail, so given
// gz = H it contributes y_e H_head^T and -y_e H_tail^T: map_gradient with the vertex and edge
// arguments trading places, map_gradient(H, y). The two must sum, which exercises accumulate.
void test_gradients_laplacian() {
    for (int d : {2, 5}) {
        const int f = 3;
        const Delta delta(n, d, bowtie);
        Mat maps = Mat::Random(2 * delta.num_edges(), d * d);
        Mat x = Mat::Random(n * d, f);
        const Mat H = Mat::Random(n * d, f);
        const Mat y = apply(delta, maps, x);
        const Mat gx = apply_transpose(delta, maps, apply(delta, maps, H));
        Mat gmaps = Mat::Random(2 * delta.num_edges(), d * d);  // stale; the first call overwrites
        delta.map_gradient(H, y, gmaps);
        delta.map_gradient(x, apply(delta, maps, H), gmaps, /*accumulate=*/true);

        const auto loss = [&] {
            return H.cwiseProduct(apply_transpose(delta, maps, apply(delta, maps, x))).sum();
        };
        CHECK(gradient_error(loss, x, gx, 1e-3) < 1e-8);
        CHECK(gradient_error(loss, maps, gmaps, 1e-3) < 1e-8);
    }
}

// The float instantiation agrees with double to single precision.
void test_float() {
    using MatF = sheaf::RowMat<float>;
    const int d = 3, f = 7;
    const Delta delta(n, d, bowtie);
    const sheaf::Coboundary<float> delta_f(n, d, bowtie);
    const int m = delta.num_edges();
    const Mat maps = Mat::Random(2 * m, d * d), x = Mat::Random(n * d, f), y = Mat::Random(m * d, f);
    MatF yf(m * d, f), xf(n * d, f), gf(2 * m, d * d);
    delta_f.apply(maps.cast<float>(), x.cast<float>(), yf);
    delta_f.apply_transpose(maps.cast<float>(), y.cast<float>(), xf);
    delta_f.map_gradient(x.cast<float>(), y.cast<float>(), gf);
    CHECK(near(yf.cast<double>(), apply(delta, maps, x), 1e-5));
    CHECK(near(xf.cast<double>(), apply_transpose(delta, maps, y), 1e-5));
    CHECK(near(gf.cast<double>(), map_gradient(delta, x, y), 1e-5));
}

void test_rejects_bad_edges() {
    const auto rejects = [](std::vector<Edge> edges) {
        try {
            (void)Delta(n, 2, edges);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(rejects({{0, 0}}));
    CHECK(rejects({{0, n}}));
    CHECK(rejects({{-1, 1}}));
    CHECK(!rejects({{0, 1}}));
}

}  // namespace

int main() {
    test_constant_sheaf_is_graph();
    test_kernels_match_matrix();
    test_strided_views();
    test_global_sections();
    test_orientation_independence();
    test_gradients();
    test_gradients_linear();
    test_gradients_laplacian();
    test_float();
    test_rejects_bad_edges();
    if (failures > 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("all coboundary checks passed");
}
