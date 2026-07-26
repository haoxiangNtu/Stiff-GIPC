// ============================================================================
// device_common/matrix_pd.cuh — makePDGeneral PSD projection template (E3.5
// hoist from gipc_modules/00 for per-term TUs). Template inline: ODR-safe.
// ============================================================================
#pragma once
#include <Eigen/Eigen>

template <typename Scalar, int size>
__device__ __host__ void makePDGeneral(Eigen::Matrix<Scalar, size, size>& symMtr)
{
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigen_solver;

    if constexpr(size <= 3)
        eigen_solver.computeDirect(symMtr);
    else
        eigen_solver.compute(symMtr);
    Eigen::Vector<Scalar, size> eigen_values = eigen_solver.eigenvalues();
    Eigen::Matrix<Scalar, size, size> eigen_vectors = eigen_solver.eigenvectors();


    if(eigen_values[0] >= 0.0)
    {
        return;
    }

    for(int i = 0; i < size; ++i)
    {
        if(eigen_values(i) < 0)
        {
            eigen_values(i) = 0;
        }
    }
    symMtr = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}
