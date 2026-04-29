#pragma once
#include <gipc/type_define.h>
#include <Eigen/Dense>
#include <vector>

namespace gipc
{

// ---------------------------------------------------------------------------
// Surface-integral utilities for ABD bodies backed by closed triangle meshes.
// Uses the Divergence theorem to compute volume, dyadic mass, and gravity
// force directly from surface triangles — no tetrahedralization needed.
// Ported from rbs-uipc (compute_mesh_volume.cpp, compute_dyadic_mass.cpp,
// compute_body_force.cpp).
// ---------------------------------------------------------------------------

/// Compute the enclosed volume of a closed triangle mesh using the
/// scalar-triple-product formula (Divergence theorem):
///   V = (1/6) * sum_i  p0_i . (p1_i x p2_i)
///
/// Optional `orient` (libuipc-style): per-triangle in {-1, 0, +1}.
/// Negative values flip the contribution sign, equivalent to swapping
/// two of the triangle's vertices but without mutating topology. Pass
/// nullptr or empty vector to use face winding as-is.
inline double compute_trimesh_volume(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    const std::vector<int>*             orient = nullptr)
{
    const bool has_orient = orient && orient->size() == triangles.size();
    double volume = 0.0;
    for(size_t i = 0; i < triangles.size(); i++)
    {
        const auto& tri = triangles[i];
        const auto& p0  = vertices[tri[0]];
        const auto& p1  = vertices[tri[1]];
        const auto& p2  = vertices[tri[2]];
        const double s  = (has_orient && (*orient)[i] < 0) ? -1.0 : 1.0;
        volume += s * p0.cross(p1).dot(p2) / 6.0;
    }
    return volume;
}

/// Compute dyadic mass quantities (m, m*x_bar, m*x_bar*x_bar^T) for a closed
/// triangle mesh body using Divergence theorem surface integrals.
///
/// These three quantities are exactly what ABDJacobiDyadicMass stores:
///   m                = total mass
///   m_x_bar          = first moment (mass * center_of_mass)
///   m_x_bar_x_bar    = second moment (mass * inertia-related tensor)
///
/// After calling, use ABDJacobiDyadicMass::from_dyadic_mass(m, m_x_bar, m_x_bar_x_bar).
inline void compute_trimesh_dyadic_mass(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    double                              density,
    double&                             out_m,
    Eigen::Vector3d&                    out_m_x_bar,
    Eigen::Matrix3d&                    out_m_x_bar_x_bar,
    const std::vector<int>*             orient = nullptr)
{
    out_m = 0.0;
    out_m_x_bar.setZero();
    out_m_x_bar_x_bar.setZero();

    const bool has_orient = orient && orient->size() == triangles.size();
    for(size_t i = 0; i < triangles.size(); i++)
    {
        const auto& tri = triangles[i];
        const auto& p0  = vertices[tri[0]];
        const auto& p1  = vertices[tri[1]];
        const auto& p2  = vertices[tri[2]];

        Eigen::Vector3d N = (p1 - p0).cross(p2 - p0);
        if(has_orient && (*orient)[i] < 0)
            N = -N;  // libuipc-style: flip integration sign without mutating topology

        // Mass: m += rho * p0 . N / 6
        out_m += density * p0.dot(N) / 6.0;

        // First moment: m_x_bar(a) += (rho/2) * N(a) * Q_first(a)
        // where Q_first(a) = (1/12) * sum_{i<=j} p_i(a)*p_j(a)
        {
            auto Q = [&](int a) -> double
            {
                double V = 0.0;
                V += p0(a) * p0(a) / 12.0;
                V += p0(a) * p1(a) / 12.0;
                V += p0(a) * p2(a) / 12.0;
                V += p1(a) * p1(a) / 12.0;
                V += p1(a) * p2(a) / 12.0;
                V += p2(a) * p2(a) / 12.0;
                return density / 2.0 * N(a) * V;
            };
            for(int a = 0; a < 3; a++)
                out_m_x_bar(a) += Q(a);
        }

        // Second moment diagonal: m_x_bar_x_bar(a,a) += (rho/3) * N(a) * Q_diag(a)
        {
            auto Q = [&](int a) -> double
            {
                double p0a = p0(a), p1a = p1(a), p2a = p2(a);
                double p0a2 = p0a * p0a, p1a2 = p1a * p1a, p2a2 = p2a * p2a;
                double V = 0.0;
                V += p0a2 * p0a / 20.0;
                V += p0a2 * p1a / 20.0;
                V += p0a2 * p2a / 20.0;
                V += p0a * p1a2 / 20.0;
                V += p0a * p1a * p2a / 20.0;
                V += p0a * p2a2 / 20.0;
                V += p1a2 * p1a / 20.0;
                V += p1a2 * p2a / 20.0;
                V += p1a * p2a2 / 20.0;
                V += p2a2 * p2a / 20.0;
                return density / 3.0 * N(a) * V;
            };
            for(int a = 0; a < 3; a++)
                out_m_x_bar_x_bar(a, a) += Q(a);
        }

        // Second moment off-diagonal: m_x_bar_x_bar(a,b) += (rho/2) * N(a) * Q_off(a,b)
        {
            auto Q = [&](int a, int b) -> double
            {
                double p0a = p0(a), p1a = p1(a), p2a = p2(a);
                double p0b = p0(b), p1b = p1(b), p2b = p2(b);
                double p0a2 = p0a * p0a, p1a2 = p1a * p1a, p2a2 = p2a * p2a;
                double V = 0.0;
                V += p0a2 * p0b / 20.0;
                V += p0a2 * p1b / 60.0;
                V += p0a2 * p2b / 60.0;
                V += p0a * p0b * p1a / 30.0;
                V += p0a * p0b * p2a / 30.0;
                V += p0a * p1a * p1b / 30.0;
                V += p0a * p1a * p2b / 60.0;
                V += p0a * p1b * p2a / 60.0;
                V += p0a * p2a * p2b / 30.0;
                V += p0b * p1a2 / 60.0;
                V += p0b * p1a * p2a / 60.0;
                V += p0b * p2a2 / 60.0;
                V += p1a2 * p1b / 20.0;
                V += p1a2 * p2b / 60.0;
                V += p1a * p1b * p2a / 30.0;
                V += p1a * p2a * p2b / 30.0;
                V += p1b * p2a2 / 60.0;
                V += p2a2 * p2b / 20.0;
                return density / 2.0 * N(a) * V;
            };
            out_m_x_bar_x_bar(0, 1) += Q(0, 1);
            out_m_x_bar_x_bar(0, 2) += Q(0, 2);
            out_m_x_bar_x_bar(1, 2) += Q(1, 2);
        }

        // Symmetric fill
        out_m_x_bar_x_bar(1, 0) = out_m_x_bar_x_bar(0, 1);
        out_m_x_bar_x_bar(2, 0) = out_m_x_bar_x_bar(0, 2);
        out_m_x_bar_x_bar(2, 1) = out_m_x_bar_x_bar(1, 2);
    }
}

/// Compute the ABD generalized gravity force for a closed triangle mesh
/// using Divergence theorem surface integrals.
///
/// Returns a Vector12:
///   [0:3]  = translational force  (= m * g)
///   [3:6]  = row-1 affine force   (= g_x * m_x_bar_col)
///   [6:9]  = row-2 affine force   (= g_y * m_x_bar_col)
///   [9:12] = row-3 affine force   (= g_z * m_x_bar_col)
inline Vector12 compute_trimesh_body_force(
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles,
    const Eigen::Vector3d&              gravity,
    const std::vector<int>*             orient = nullptr)
{
    Vector12 body_force = Vector12::Zero();

    const bool has_orient = orient && orient->size() == triangles.size();
    for(size_t i = 0; i < triangles.size(); i++)
    {
        const auto& tri = triangles[i];
        const auto& p0  = vertices[tri[0]];
        const auto& p1  = vertices[tri[1]];
        const auto& p2  = vertices[tri[2]];

        Eigen::Vector3d N = (p1 - p0).cross(p2 - p0);
        if(has_orient && (*orient)[i] < 0)
            N = -N;
        double V = p0.dot(N) / 6.0;

        auto Q_p = [&](int a) -> double
        {
            double val = 0.0;
            val += p0(a) * p0(a) / 12.0;
            val += p0(a) * p1(a) / 12.0;
            val += p0(a) * p2(a) / 12.0;
            val += p1(a) * p1(a) / 12.0;
            val += p1(a) * p2(a) / 12.0;
            val += p2(a) * p2(a) / 12.0;
            return 0.5 * N(a) * val;
        };

        Eigen::Vector3d Qs(Q_p(0), Q_p(1), Q_p(2));

        body_force.segment<3>(0) += gravity * V;
        body_force.segment<3>(3) += gravity.x() * Qs;
        body_force.segment<3>(6) += gravity.y() * Qs;
        body_force.segment<3>(9) += gravity.z() * Qs;
    }

    return body_force;
}

}  // namespace gipc
