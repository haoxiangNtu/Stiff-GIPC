#include <cstdlib>
#include <abd_system/abd_system.h>
#include <muda/launch.h>
#include <muda/ext/eigen/evd.h>
#include <muda/ext/eigen/atomic.h>
#include <muda/ext/eigen/inverse.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
#include <abd_system/abd_energy.h>
#include <abd_system/abd_joint_constraint.h>
#include <abd_system/abd_driving_joint.h>
#include <gipc/utils/math.h>
#include <gipc/utils/timer.h>
#include "cuda_tools/cuda_tools.h"
#include <linear_system/utils/binned_reduce.cuh>
// [multi-env determinism 4.3] device globals for the ABD binned accumulators (set via
// cudaMemcpyToSymbol from ABDSystem::m_abd_* before assembly; -rdc on). cal_q_tilde.cu uses
// g_abd_wrenchbin via extern. The assembly kernels' atomic_add → bin_add* into these.
__device__ double* g_abd_sysbin    = nullptr;
__device__ double* g_abd_hessbin   = nullptr;
__device__ double* g_abd_wrenchbin = nullptr;
#include <fstream>
#include <vector>
namespace gipc
{
// [4.3] combine binned accumulators back into the ABD buffers (+= onto the body-func init).
__global__ void _abd_sysbin_combine_k(double* g, const double* bin, int n)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if(d >= n) return;
    g[d] += binned_combine(bin + (size_t)d * BINNED_K);
}
__global__ void _abd_hessbin_combine_k(Matrix12x12* H, const double* bin, int nb)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if(b >= nb) return;
    for(int i = 0; i < 12; ++i)
        for(int j = 0; j < 12; ++j)
            H[b](i, j) += binned_combine(bin + ((size_t)b * 144 + i * 12 + j) * BINNED_K);
}
void ABDSystem::_abd_binned_open(ABDSimData& sim_data)
{
    int N = sim_data.abd_fem_count_info().abd_body_num;
    if(N < 1) return;
    size_t sg = (size_t)N * 12 * BINNED_K, hb = (size_t)N * 144 * BINNED_K;
    if(sg > m_abd_sysbin_cap)
    { if(m_abd_sysbin) cudaFree(m_abd_sysbin); cudaMalloc((void**)&m_abd_sysbin, sg * sizeof(double)); m_abd_sysbin_cap = sg; }
    if(hb > m_abd_hessbin_cap)
    { if(m_abd_hessbin) cudaFree(m_abd_hessbin); cudaMalloc((void**)&m_abd_hessbin, hb * sizeof(double)); m_abd_hessbin_cap = hb; }
    cudaMemset(m_abd_sysbin, 0, sg * sizeof(double));
    cudaMemset(m_abd_hessbin, 0, hb * sizeof(double));
    cudaMemcpyToSymbol(g_abd_sysbin, &m_abd_sysbin, sizeof(double*));
    cudaMemcpyToSymbol(g_abd_hessbin, &m_abd_hessbin, sizeof(double*));
}
void ABDSystem::_abd_binned_close(ABDSimData& sim_data)
{
    int N = sim_data.abd_fem_count_info().abd_body_num;
    if(N < 1) return;
    { int n = N * 12;
      muda::ParallelFor(256).apply(n,
          [g = system_gradient.viewer(), bin = m_abd_sysbin] __device__(int d) mutable
          { g(d) += binned_combine(bin + (size_t)d * BINNED_K); }); }
    { int bs = 256, gs = (N + bs - 1) / bs;
      if(gs > 0)  // [zero-ABD guard] gridDim=0 launch = cudaErrorInvalidConfiguration
      _abd_hessbin_combine_k<<<gs, bs>>>(abd_body_hessian.data(), m_abd_hessbin, N); }
}
void ABDSystem::couple_bin_open(int n_dofs)
{
    if(n_dofs < 1) return;
    size_t sg = (size_t)n_dofs * BINNED_K;
    if(sg > m_abd_sysbin_cap)
    { if(m_abd_sysbin) cudaFree(m_abd_sysbin); cudaMalloc((void**)&m_abd_sysbin, sg * sizeof(double)); m_abd_sysbin_cap = sg; }
    cudaMemset(m_abd_sysbin, 0, sg * sizeof(double));
    cudaMemcpyToSymbol(g_abd_sysbin, &m_abd_sysbin, sizeof(double*));
}
void ABDSystem::couple_bin_close(int n_dofs)
{
    if(n_dofs < 1) return;
    muda::ParallelFor(256).apply(n_dofs,
        [g = system_gradient.viewer(), bin = m_abd_sysbin] __device__(int d) mutable
        { g(d) += binned_combine(bin + (size_t)d * BINNED_K); });
}

struct DrivingCtrlPacked  { Float target_angle;    Float strength_ratio; Float ext_torque; };
struct PrisCtrlPacked     { Float target_distance; Float strength_ratio; Float ext_force; };

//template <int ROWS, int COLS>
__device__ inline void write_triplet_cv(Eigen::Matrix3d* triplet_value,
                                        int*             row_ids,
                                        int*             col_ids,
                                        unsigned int*    node_index,
                                        const Eigen::Matrix<double, 12, 12>& input,
                                        const int& offset)
{
    int rown = 4;
    int coln = 4;
    int kk    = 0;
    for(int ii = 0; ii < rown; ii++)
    {
        for(int jj = ii; jj < coln; jj++)
        {
            row_ids[offset + kk]       = node_index[ii];
            col_ids[offset + kk]       = node_index[jj];
            triplet_value[offset + kk] = input.block<3, 3>(ii * 3, jj * 3);
            kk++;
        }
    }
}




template <int ROWS, int COLS>
__device__ inline void write_triplet_cv2(Eigen::Matrix3d* triplet_value,
                                         int*             row_ids,
                                         int*             col_ids,
                                         unsigned int*    node_index_rows,
                                         unsigned int*    node_index_cols,
                                         const Eigen::Matrix<double, ROWS, COLS>& input,
                                         const int& offset)
{
    int rown = ROWS / 3;
    int coln = COLS / 3;
    for(int ii = 0; ii < rown; ii++)
    {
        for(int jj = 0; jj < coln; jj++)
        {
            int kk  = ii * coln + jj;
            int row = node_index_rows[ii];
            int col = node_index_cols[jj];


            if(row <= col)
            {
                row_ids[offset + kk]       = row;
                col_ids[offset + kk]       = col;
                triplet_value[offset + kk] = input.block<3, 3>(ii * 3, jj * 3);
            }
            else
            {
                row_ids[offset + kk]       = col;
                col_ids[offset + kk]       = row;
                triplet_value[offset + kk].setZero();
            }
        }
    }
}


__device__ inline void write_cross_body_triplet_cv2(
    Eigen::Matrix3d* triplet_value,
    int*             row_ids,
    int*             col_ids,
    int              body_i,
    int              body_j,
    const Matrix12x12& input,
    const int&        offset)
{
    int          output_body_i = body_i;
    int          output_body_j = body_j;
    Matrix12x12  output         = input;

    if(body_i == body_j)
    {
        Matrix12x12 input_transpose = input.transpose();
        output += input_transpose;
    }
    else if(body_i > body_j)
    {
        output_body_i = body_j;
        output_body_j = body_i;
        output         = input.transpose();
    }

    unsigned int output_base_i = static_cast<unsigned int>(output_body_i * 4);
    unsigned int output_base_j = static_cast<unsigned int>(output_body_j * 4);
    unsigned int index_row[4] = {
        output_base_i, output_base_i + 1, output_base_i + 2, output_base_i + 3};
    unsigned int index_col[4] = {
        output_base_j, output_base_j + 1, output_base_j + 2, output_base_j + 3};

    write_triplet_cv2<12, 12>(
        triplet_value, row_ids, col_ids, index_row, index_col, output, offset);
}


template <typename T>
__global__ inline void moveMemory_0(T* data, int output_start, int input_start, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= length)
        return;
    data[output_start + idx] = data[input_start + idx];
}



__global__ void write_barrier_hessian(//muda::TripletMatrixViewer<double, 12> tripletViewer,
                                      Eigen::Matrix3d*        triplet,
                                      int*              rows,
                                      int*              cols,
                                      ABDJacobi*        abd_J,
                                      const int*          body_id,
                                      const BodyBoundaryType* is_fixed,
                                      int               start_input,
                                      int               start_output,
                                      int               number)
{
    int vI = blockIdx.x * blockDim.x + threadIdx.x;
    if(vI >= number)
        return;

    auto H = triplet[vI + start_input];
    auto i = rows[vI + start_input];
    auto j = cols[vI + start_input];
    auto body_id_i = body_id[i];
    auto body_id_j = body_id[j];

    int output_body_i = body_id_i;
    int output_body_j = body_id_j;
    int offset = vI * 16 + start_output;

    if(output_body_i > output_body_j)
    {
        int temp = output_body_i;
        output_body_i = output_body_j;
        output_body_j = temp;
    }

    unsigned int output_base_i = static_cast<unsigned int>(output_body_i * 4);
    unsigned int output_base_j = static_cast<unsigned int>(output_body_j * 4);
    unsigned int index_row[4] = {
        output_base_i, output_base_i + 1, output_base_i + 2, output_base_i + 3};

    unsigned int index_col[4] = {
        output_base_j, output_base_j + 1, output_base_j + 2, output_base_j + 3};

    if(is_fixed[body_id_i] == BodyBoundaryType::Fixed
       || is_fixed[body_id_j] == BodyBoundaryType::Fixed)
    {
        Matrix12x12 zero12 = Matrix12x12::Zero();
        write_triplet_cv2<12, 12>(triplet, rows, cols, index_row, index_col, zero12, offset);
    }
    else
    {
        auto ABD_H = ABDJacobi::JT_H_J(abd_J[i].T(), H, abd_J[j]);
        Matrix12x12 ABD_H_transpose = ABD_H.transpose();
        if(body_id_i == body_id_j)
        {
            if(i == j)
                ABD_H = (ABD_H + ABD_H_transpose) * 0.5;
            else
                ABD_H += ABD_H_transpose;
        }
        else if(body_id_i > body_id_j)
        {
            ABD_H = ABD_H_transpose;
        }
        write_triplet_cv2<12, 12>(triplet, rows, cols, index_row, index_col, ABD_H, offset);
    }

}



__global__ void write_abd_body_hessian(
    Matrix12x12* matrix_input, Matrix3x3* triplet, int* row, int* col, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int          offset   = idx * 10;
    unsigned int index[4] = {idx * 4, idx * 4 + 1, idx * 4 + 2, idx * 4 + 3};
    write_triplet_cv(triplet, row, col, index, matrix_input[idx], offset);
}


void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  GIPCTripletMatrix& global_triplets,
                                                  muda::CBufferView<double3> vertex_barrier_gradient)
{
    setup_abd_non_contact_gradient(sim_data);
    add_abd_contact_gradient(sim_data, vertex_barrier_gradient);
    _setup_abd_system_hessian(sim_data, global_triplets);
}

void ABDSystem::setup_abd_non_contact_gradient(ABDSimData& sim_data)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _abd_binned_open(sim_data);   // [4.3] zero binned accumulators + bind globals (after per-body init)
    _cal_abd_joint_gradient_and_hessian(sim_data);
    _cal_abd_revolute_driving_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_driving_gradient_and_hessian(sim_data);
    _cal_abd_stitch_gradient_and_hessian(sim_data);
    _abd_binned_close(sim_data);  // [4.3] combine binned coupling gradient/Hessian into ABD buffers
}

void ABDSystem::add_abd_contact_gradient(
    ABDSimData& sim_data, muda::CBufferView<double3> vertex_contact_gradient)
{
    _abd_binned_open(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_contact_gradient);
    _abd_binned_close(sim_data);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  GIPCTripletMatrix& global_triplets,
                                                  muda::CBufferView<Vector3> vertex_barrier_gradient)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _abd_binned_open(sim_data);   // [4.3] zero binned accumulators + bind globals (after per-body init)
    _cal_abd_joint_gradient_and_hessian(sim_data);
    _cal_abd_revolute_driving_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_driving_gradient_and_hessian(sim_data);
    _cal_abd_stitch_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
    _abd_binned_close(sim_data);  // [4.3] combine binned coupling gradient/Hessian into ABD buffers
    _setup_abd_system_hessian(sim_data, global_triplets);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  int*        fbtype,
                                                  muda::CBufferView<double3> vertex_barrier_gradient,
                                                  GIPCTripletMatrix& global_triplets)
{
    fem_boundary_type = fbtype;
    setup_abd_system_gradient_hessian(sim_data, global_triplets, vertex_barrier_gradient);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    converter3x3.convert(global_triplets,
                         global_triplets.h_abd_abd_contact_start_id,
                         global_triplets.abd_abd_contact_num,
                         global_triplets.global_collision_triplet_offset);
    global_triplets.global_collision_triplet_offset =
        global_triplets.global_collision_triplet_offset
        - global_triplets.abd_abd_contact_num + global_triplets.h_unique_key_number;
    global_triplets.global_triplet_offset = global_triplets.global_collision_triplet_offset;
    global_triplets.abd_abd_contact_num = global_triplets.h_unique_key_number;
}


// file local function, make the matrix positive definite
__device__ __host__ void make_pd(Matrix9x9& mat)
{
    Vector9   eigen_values;
    Matrix9x9 eigen_vectors;
    muda::eigen::evd<Float, 9>(mat, eigen_values, eigen_vectors);
    for(int i = 0; i < 9; ++i)
    {
        if(eigen_values(i) < 0)
        {
            eigen_values(i) = 0;
        }
    }
    mat = eigen_vectors * eigen_values.asDiagonal() * eigen_vectors.transpose();
}


void ABDSystem::_cal_abd_body_gradient_and_hessian(ABDSimData& sim_data)
{
    gipc::Timer timer("_cal_abd_body_gradient_and_hessian");
    using namespace muda;
    auto& abd       = sim_data.device;
    auto  N         = sim_data.abd_fem_count_info().abd_body_num;
    auto  parameter = parms;
    abd_body_hessian.resize(N);
    abd_gradient.resize(N);
    system_gradient.resize(N * 12);

    auto boundary_type = sim_data.body_id_to_boundary_type();

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(N,
               [boundary_type = boundary_type.cviewer().name("btype"),
                qs            = abd.body_id_to_q.cviewer().name("q"),
                q_tildes = abd.body_id_to_q_tilde.cviewer().name("affine_q_tilde"),
                q_prev    = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                Ms        = abd.body_id_to_abd_mass.cviewer().name("M"),
                volumes   = abd.body_id_to_volume.cviewer().name("volumes"),
                gradients = abd_gradient.viewer().name("abd_gradient"),
                system_gradient = system_gradient.viewer().name("system_gradient"),
                body_hessian = abd_body_hessian.viewer().name("shape_hessian"),
                kappa        = parameter.kappa,
                dt           = parameter.dt,
                body_motor_data  = sim_data.body_motor_params(),
                motor_speed  = parms.motor_speed,
                motor_strength = parms.motor_strength] __device__(int i) mutable
               {
                   if(boundary_type(i) == BodyBoundaryType::Fixed)
                   {
                       gradients(i) = Vector12::Zero();
                       system_gradient.segment<12>(i * 12).as_eigen().setZero();
                       body_hessian(i) = Ms(i).to_mat();
                       // body_hessian(i) = Matrix12x12::Zero();
                   }
                   else
                   {
                       Matrix12x12 H = Matrix12x12::Zero();
                       Vector12    G = Vector12::Zero();

                       const auto& q       = qs(i);
                       const auto& q_tilde = q_tildes(i);
                       const auto& M       = Ms(i);

                       {  // kinetic energy
                           Vector12 dq               = (q - q_tilde);
                           Vector12 kinetic_gradient = M * dq;
                           H                         = M.to_mat();
                           G                         = kinetic_gradient;
                       }

                       {  // shape energy
                           const auto& volume = volumes(i);
                           auto        kvt2   = kappa * volume * dt * dt;
                           Vector9 shape_gradient = kvt2 * shape_energy_gradient(q);

                           Matrix9x9 shape_H = kvt2 * shape_energy_hessian(q);

                           // make H positive definite
                           make_pd(shape_H);
                           H.block<9, 9>(3, 3) += shape_H;
                           G.segment<9>(3) += shape_gradient;
                       }

                       gradients(i)                                   = G;
                       system_gradient.segment<12>(i * 12).as_eigen() = G;
                       body_hessian(i)                                = H;

                       if(boundary_type(i) == BodyBoundaryType::Animated)
                       {
                           // Soft drive with ABSOLUTE target:
                           // body_motor_params = [target_x, target_y, target_z, strength, 0]
                           // Constrain translation toward target, affine A toward identity.
                           Vector3 aim_pos = q_tilde.segment<3>(0);  // fallback to q_tilde
                           double  anim_strength = 1e6;
                           if(body_motor_data)
                           {
                               aim_pos(0) = body_motor_data[i * 5 + 0];
                               aim_pos(1) = body_motor_data[i * 5 + 1];
                               aim_pos(2) = body_motor_data[i * 5 + 2];
                               double st = body_motor_data[i * 5 + 3];
                               if(st > 0.0) anim_strength = st;
                           }

                           // q_aim: target position + identity affine matrix
                           Vector12 q_aim;
                           q_aim.segment<3>(0) = aim_pos;
                           q_aim(3) = 1.0; q_aim(4) = 0.0; q_aim(5) = 0.0;
                           q_aim(6) = 0.0; q_aim(7) = 1.0; q_aim(8) = 0.0;
                           q_aim(9) = 0.0; q_aim(10) = 0.0; q_aim(11) = 1.0;

                           Vector12 dq = q - q_aim;
                           // Penalize all 12 DOFs: translation + affine
                           Matrix12x12 PowMass = anim_strength * Matrix12x12::Identity();

                           system_gradient.segment<12>(i * 12).as_eigen() += PowMass * dq;
                           gradients(i) += PowMass * dq;
                           body_hessian(i) += PowMass;
                       }

                       if(boundary_type(i) == BodyBoundaryType::Motor)
                       {
                           // Read per-body motor params: [axis_x, axis_y, axis_z, speed, strength]
                           Vector3 rot_axis = Vector3::UnitX();
                           double  body_speed    = motor_speed;
                           double  body_strength = motor_strength;
                           if(body_motor_data)
                           {
                               double ax = body_motor_data[i * 5 + 0];
                               double ay = body_motor_data[i * 5 + 1];
                               double az = body_motor_data[i * 5 + 2];
                               double sp = body_motor_data[i * 5 + 3];
                               double st = body_motor_data[i * 5 + 4];
                               double len = sqrt(ax * ax + ay * ay + az * az);
                               if(len > 1e-10)
                                   rot_axis = Vector3{ax, ay, az} / len;
                               if(sp > 0.0)
                                   body_speed = sp;
                               if(st > 0.0)
                                   body_strength = st;
                           }

                           Vector3 bar_x0 = Vector3::Zero();
                           Vector3 bar_x1 = Vector3::UnitX();
                           Vector3 bar_x2 = Vector3::UnitY();
                           Vector3 bar_x3 = Vector3::UnitZ();

                           auto mat0 = ABDJacobi{bar_x0}.to_mat();
                           auto mat1 = ABDJacobi{bar_x1}.to_mat();
                           auto mat2 = ABDJacobi{bar_x2}.to_mat();
                           auto mat3 = ABDJacobi{bar_x3}.to_mat();

                           Matrix12x12 J;
                           J.block<3, 12>(0, 0) = mat0;
                           J.block<3, 12>(3, 0) = mat1;
                           J.block<3, 12>(6, 0) = mat2;
                           J.block<3, 12>(9, 0) = mat3;

                           Matrix12x12 inv_J = eigen::inverse(J);

                           auto theta = body_speed * dt;
                           // rotate around per-body axis
                           auto R = Eigen::AngleAxisd(theta, rot_axis);

                           Vector3 x1_P = R * bar_x1;
                           Vector3 x2_P = R * bar_x2;
                           Vector3 x3_P = R * bar_x3;

                           auto mat0_delta = ABDJacobi{Vector3::Zero()}.to_mat();
                           auto mat1_delta = ABDJacobi{x1_P - bar_x1}.to_mat();
                           auto mat2_delta = ABDJacobi{x2_P - bar_x2}.to_mat();
                           auto mat3_delta = ABDJacobi{x3_P - bar_x3}.to_mat();

                           Matrix12x12 J_delta;
                           J_delta.block<3, 12>(0, 0) = mat0_delta;
                           J_delta.block<3, 12>(3, 0) = mat1_delta;
                           J_delta.block<3, 12>(6, 0) = mat2_delta;
                           J_delta.block<3, 12>(9, 0) = mat3_delta;

                           // Vector12 q_p = inv_J * J_delta * q_prev(i) + q_prev(i);
                           Vector12 q_p = inv_J * J_delta * q_tilde + q_tilde;
                           q_p.segment<3>(3).normalize();
                           q_p.segment<3>(6).normalize();
                           q_p.segment<3>(9).normalize();

                           Vector12 dq      = q - q_p;
                           dq.segment<3>(0) = Vector3::Zero();

                           Matrix12x12 PowMass = Matrix12x12::Zero();
                           PowMass.block<9, 9>(3, 3) =
                               body_strength * Ms(i).to_mat().block<9, 9>(3, 3);


                           system_gradient.segment<12>(i * 12).as_eigen() += PowMass * dq;
                           gradients(i) += PowMass * dq;

                           // Power Mass
                           body_hessian(i) += PowMass;
                       }
                   }
               });
}


void ABDSystem::_cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                                 muda::CBufferView<double3> vertex_barrier_gradient)
{
    gipc::Timer timer("_cal_abd_system_barrier_gradient");
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();
    auto  body_id_is_fixed   = sim_data.body_id_to_boundary_type();


    // Barrier Part
    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(vertex_barrier_gradient.size(),
               [unique_point_id_to_body_id = body_id.cviewer().name("unique_point_id_to_body_id"),
                gradient = vertex_barrier_gradient.cviewer().name("gradient"),
                affine_gradient = abd_gradient.viewer().name("abd_gradient"),
                system_gradient = system_gradient.viewer().name("system_gradient"),
                is_fixed = body_id_is_fixed.cviewer().name("is_fixed"),
                J = abd.unique_point_id_to_J.cviewer().name("J")] __device__(int i) mutable
               {
                   auto  body_id = unique_point_id_to_body_id(i);
                   auto& dst     = affine_gradient(body_id);
                   auto& g       = gradient(i);

                   if(is_fixed(body_id) == BodyBoundaryType::Fixed)
                       return;

                   //printf("barrier gradient[%d]=%f %f %f\n", i, g.x, g.y, g.z);

                   Vector12 G = J(i).T() * Vector3{g.x, g.y, g.z};
                   eigen::atomic_add(dst, G);
                   bin_add12_off(g_abd_sysbin, body_id * 12, G);
               });
}

void ABDSystem::_cal_abd_system_barrier_gradient(ABDSimData& sim_data,
                                                 muda::CBufferView<Vector3> vertex_barrier_gradient)
{
    using namespace muda;
    auto& abd                = sim_data.device;
    auto  abd_count          = sim_data.abd_fem_count_info().abd_body_num;
    auto  unique_point_count = sim_data.abd_fem_count_info().abd_point_num;
    auto  body_id            = sim_data.unique_point_id_to_body_id();
    auto  body_id_is_fixed   = sim_data.body_id_to_boundary_type();


    // Barrier Part
    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(vertex_barrier_gradient.size(),
               [unique_point_id_to_body_id = body_id.cviewer().name("unique_point_id_to_body_id"),
                gradient = vertex_barrier_gradient.cviewer().name("gradient"),
                affine_gradient = abd_gradient.viewer().name("abd_gradient"),
                system_gradient = system_gradient.viewer().name("system_gradient"),
                J = abd.unique_point_id_to_J.cviewer().name("J"),
                is_fixed = body_id_is_fixed.cviewer().name("is_fixed")] __device__(int i) mutable
               {
                   auto  body_id = unique_point_id_to_body_id(i);
                   auto& dst     = affine_gradient(body_id);
                   auto& g       = gradient(i);


                   if(is_fixed(body_id) == BodyBoundaryType::Fixed)
                       return;

                   Vector12 G = J(i).T() * g;

                   eigen::atomic_add(dst, G);
                   bin_add12_off(g_abd_sysbin, body_id * 12, G);
               });
}

void ABDSystem::_setup_abd_system_hessian(ABDSimData& sim_data,
                                          GIPCTripletMatrix& global_triplets)
{
    gipc::Timer timer("_setup_abd_system_hessian");
    using namespace muda;

    if(global_triplets.abd_abd_contact_num)
    {
        converter3x3.convert(global_triplets,
                             global_triplets.h_abd_abd_contact_start_id,
                             global_triplets.abd_abd_contact_num,
                             global_triplets.global_collision_triplet_offset);
    }
    else
    {
        global_triplets.h_unique_key_number = 0;
    }

    int bcooNum = global_triplets.h_unique_key_number; 

    auto abd_body_count   = sim_data.abd_fem_count_info().abd_body_num;
    auto body_id_is_fixed = sim_data.body_id_to_boundary_type();

    auto unique_point_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto body_hessian_size          = abd_body_count;

    global_triplets.abd_abd_contact_num = bcooNum;
    int joint_triplet_blocks = m_num_joints * 16;
    int revolute_driving_triplet_blocks = m_num_revolute_driving * 16;
    int prismatic_triplet_blocks = m_num_prismatic * 16;
    int prismatic_driving_triplet_blocks = m_num_prismatic_driving * 16;
    int stitch_triplet_blocks = m_stitch_count * 4;
    int new_triplet_offset =
        global_triplets.fem_fem_contact_num + global_triplets.abd_fem_contact_num * 4
        + (global_triplets.abd_abd_contact_num * 16 + abd_body_count * 10
           + joint_triplet_blocks + revolute_driving_triplet_blocks
           + prismatic_triplet_blocks + prismatic_driving_triplet_blocks
           + stitch_triplet_blocks);

    int h_abd_fem_contact_start_id = global_triplets.fem_fem_contact_num;
    int h_abd_abd_contact_start_id =
        h_abd_fem_contact_start_id + global_triplets.abd_fem_contact_num * 4;


    int write_offset = 0;

    if(getenv("STIFF_TRIPLET_DBG"))
        printf("[triplet-dbg] cap=%zu  ff=%d abdfem=%d abdabd=%d  abd_bodies=%d "
               "stitch=%d joints=%d  new_off=%d abd_abd_start=%d  first_write=%d\n",
               global_triplets.triplet_capacity(),
               global_triplets.fem_fem_contact_num, global_triplets.abd_fem_contact_num,
               global_triplets.abd_abd_contact_num, (int)abd_body_count,
               m_stitch_count, m_num_joints, new_triplet_offset,
               h_abd_abd_contact_start_id,
               h_abd_abd_contact_start_id + new_triplet_offset + write_offset);

    // [abd-assembly capacity] Everything below indexes off new_triplet_offset:
    // the block groups are written at (h_abd_abd_contact_start_id +
    // new_triplet_offset + ...), and the memcpy at the end of this function
    // reads the scratch region [new_triplet_offset, 2*new_triplet_offset) back
    // down over [fem_fem_contact_num, new_triplet_offset). So the buffer must
    // hold 2*new_triplet_offset blocks. GIPC.cu's pre-assembly bound only
    // estimates that from h_cpNum, which under-shoots on contact-dense scenes
    // (measured: a grasped 6720-face tile wanted 11.3M blocks against a 10.7M
    // capacity -> illegal write in thread 0 of _setup_abd_system_hessian, and
    // an invalid-argument memcpy here). Grow to the exact figure instead.
    // reserve_triplets (not ensure_capacity_discard): the contact triplets in
    // [0, new_triplet_offset) were written earlier this step and must survive.
    {
        const size_t need = 2ull * static_cast<size_t>(new_triplet_offset) + 65536ull;
        if(global_triplets.triplet_capacity() < need)
            global_triplets.reserve_triplets(need);
    }


    int number    = body_hessian_size;
    int threadNum = 256;
    //int blockNum  = (number + threadNum - 1) / threadNum;

    LaunchCudaKernal_default(
        body_hessian_size,
        threadNum,
        0,
        write_abd_body_hessian,
        abd_body_hessian.viewer().data(),
        global_triplets.block_values(h_abd_abd_contact_start_id + new_triplet_offset + write_offset),
        global_triplets.block_row_indices(h_abd_abd_contact_start_id
                                          + new_triplet_offset + write_offset),
        global_triplets.block_col_indices(h_abd_abd_contact_start_id
                                          + new_triplet_offset + write_offset),
        (int)body_hessian_size);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(bcooNum)
    {
        {

        gipc::Timer timer("barrier_hessian");
        auto abd_J = sim_data.device.unique_point_id_to_J.viewer().data();
        auto is_fixed     = body_id_is_fixed.viewer().data();
        auto my_triplet   = global_triplets.block_values();
        auto my_rows      = global_triplets.block_row_indices();
        auto my_cols      = global_triplets.block_col_indices();
        auto body_id      = unique_point_id_to_body_id.viewer().data();
        int  start_input  = global_triplets.h_abd_abd_contact_start_id;
        int  start_output = new_triplet_offset + h_abd_abd_contact_start_id
                           + write_offset + 10 * body_hessian_size;

        LaunchCudaKernal_default(bcooNum,
                                 threadNum,
                                 0,
                                 write_barrier_hessian,
                                 my_triplet,
                                 my_rows,
                                 my_cols,
                                 abd_J,
                                 body_id,
                                 is_fixed,
                                 start_input,
                                 start_output,
                                 bcooNum);
        }


       
    }

    if(global_triplets.abd_fem_contact_num)
    {
        auto fem_point_offset = sim_data.abd_fem_count_info().abd_point_num;
        auto fem_count        = sim_data.abd_fem_count_info().fem_point_num;
        auto femb_type =
            muda::CBufferView<int>(fem_boundary_type, fem_point_offset, fem_count);

        ParallelFor()
            .kernel_name(__FUNCTION__)
            .apply(global_triplets.abd_fem_contact_num,
                   [point_to_body = unique_point_id_to_body_id.cviewer().name("body_id"),
                    btype = femb_type.cviewer().name("boundary_type"),
                    Js = sim_data.device.unique_point_id_to_J.cviewer().name("Js"),
                    triplet_out = global_triplets.block_values() + new_triplet_offset,
                    row_out = global_triplets.block_row_indices() + new_triplet_offset,
                    col_out = global_triplets.block_col_indices() + new_triplet_offset,
                    abd_fem_contact = global_triplets.block_values()
                                      + global_triplets.h_abd_fem_contact_start_id,

                    abd_fem_rows = global_triplets.block_row_indices()
                                   + global_triplets.h_abd_fem_contact_start_id,
                    abd_fem_cols = global_triplets.block_col_indices()
                                   + global_triplets.h_abd_fem_contact_start_id,
                    h_abd_fem_contact_start_id,
                    is_fixed = body_id_is_fixed.cviewer().name("is_fixed"),
                    abd_body_count,
                    fem_point_offset] __device__(int I) mutable
                   {
                       // 1. process upper : ABD-FEM
                       {
                           auto abd_fem_H3x3 = abd_fem_contact[I];
                           auto i_abd = abd_fem_rows[I];  // global point id
                           auto j_fem = abd_fem_cols[I];  // global point id

                           auto local_fem_point_id = j_fem - fem_point_offset;

                           auto body_id = point_to_body(i_abd);  // global body id

                           auto local_abd_body_id = body_id;  // - abd_body_offset;
                           auto local_abd_point_id = i_abd;  // - abd_point_offset;
                           //tex: $\mathbf{J}_{3\times 12}$
                           gipc::ABDJacobi J = Js(local_abd_point_id);
                           gipc::Matrix12x3 H = J.to_mat().transpose() * abd_fem_H3x3;
                           //tex:
                           //$$
                           // \mathbf{H} = \begin{bmatrix}
                           //  \mathbf{H}_{1} \\ \mathbf{H}_{2} \\ \mathbf{H}_{3} \\ \mathbf{H}_{4}
                           //\end{bmatrix}
                           //$$
                           auto offset = 4 * I;
                           if(btype(local_fem_point_id) != 0
                              || is_fixed(body_id) == BodyBoundaryType::Fixed)
                           {
                               H.setZero();
                           }
                           for(int i = 0; i < 4; ++i)
                           {
                               triplet_out[h_abd_fem_contact_start_id + I * 4 + i] =
                                   H.block<3, 3>(i * 3, 0);
                               row_out[h_abd_fem_contact_start_id + I * 4 + i] =
                                   body_id * 4 + i;

                               col_out[h_abd_fem_contact_start_id + I * 4 + i] =
                                   abd_body_count * 4 + local_fem_point_id;
                           }
                       }
                   });
    }

    global_triplets.h_abd_abd_contact_start_id = h_abd_abd_contact_start_id;
    global_triplets.abd_abd_contact_num =
        16 * global_triplets.abd_abd_contact_num + abd_body_count * 10
        + joint_triplet_blocks + revolute_driving_triplet_blocks
        + prismatic_triplet_blocks + prismatic_driving_triplet_blocks
        + stitch_triplet_blocks;

    // Write joint cross-body Hessian triplets
    if(m_num_joints > 0)
    {
        int joint_output_start = new_triplet_offset + h_abd_abd_contact_start_id
                                 + write_offset + 10 * body_hessian_size;
        if(bcooNum > 0)
            joint_output_start += 16 * bcooNum;

        ParallelFor(256)
            .kernel_name("write_joint_cross_hessian")
            .apply(m_num_joints,
                   [joints        = m_joint_data.cviewer().name("joint_data"),
                    cross_hessian = m_joint_cross_hessian.cviewer().name("joint_cross_hessian"),
                    triplet_out   = global_triplets.block_values(),
                    row_out       = global_triplets.block_row_indices(),
                    col_out       = global_triplets.block_col_indices(),
                    joint_output_start] __device__(int j) mutable
                   {
                       auto& joint = joints(j);
                       int   pid   = joint.parent_body_id;
                       int   cid   = joint.child_body_id;

                       auto H_pc = cross_hessian(j);

                       int offset = joint_output_start + j * 16;
                       write_cross_body_triplet_cv2(
                           triplet_out, row_out, col_out, pid, cid, H_pc, offset);
                   });
    }

    // Write revolute driving cross-body Hessian triplets
    if(m_num_revolute_driving > 0)
    {
        int drv_output_start = new_triplet_offset + h_abd_abd_contact_start_id
                               + write_offset + 10 * body_hessian_size;
        if(bcooNum > 0)
            drv_output_start += 16 * bcooNum;
        drv_output_start += joint_triplet_blocks;  // after joint cross-hessians

        ParallelFor(256)
            .kernel_name("write_revolute_driving_cross_hessian")
            .apply(m_num_revolute_driving,
                   [drvs          = m_revolute_driving_data.cviewer().name("drv_data"),
                    cross_hessian = m_revolute_driving_cross_hessian.cviewer().name("drv_cross_hessian"),
                    triplet_out   = global_triplets.block_values(),
                    row_out       = global_triplets.block_row_indices(),
                    col_out       = global_triplets.block_col_indices(),
                    drv_output_start] __device__(int j) mutable
                   {
                       auto& drv = drvs(j);
                       int   pid = drv.parent_body_id;
                       int   cid = drv.child_body_id;

                       auto H_pc = cross_hessian(j);

                       int offset = drv_output_start + j * 16;
                       write_cross_body_triplet_cv2(
                           triplet_out, row_out, col_out, pid, cid, H_pc, offset);
                   });
    }

    // Write prismatic constraint cross-body Hessian triplets
    if(m_num_prismatic > 0)
    {
        int pris_output_start = new_triplet_offset + h_abd_abd_contact_start_id
                                + write_offset + 10 * body_hessian_size;
        if(bcooNum > 0)
            pris_output_start += 16 * bcooNum;
        pris_output_start += joint_triplet_blocks;
        pris_output_start += revolute_driving_triplet_blocks;

        ParallelFor(256)
            .kernel_name("write_prismatic_cross_hessian")
            .apply(m_num_prismatic,
                   [prisms        = m_prismatic_data.cviewer().name("prism_data"),
                    cross_hessian = m_prismatic_cross_hessian.cviewer().name("pris_cross_hessian"),
                    triplet_out   = global_triplets.block_values(),
                    row_out       = global_triplets.block_row_indices(),
                    col_out       = global_triplets.block_col_indices(),
                    pris_output_start] __device__(int j) mutable
                   {
                       auto& pj  = prisms(j);
                       int   pid = pj.parent_body_id;
                       int   cid = pj.child_body_id;

                       auto H_pc = cross_hessian(j);

                       int offset = pris_output_start + j * 16;
                       write_cross_body_triplet_cv2(
                           triplet_out, row_out, col_out, pid, cid, H_pc, offset);
                   });
    }

    // Write prismatic driving cross-body Hessian triplets
    if(m_num_prismatic_driving > 0)
    {
        int pris_drv_output_start = new_triplet_offset + h_abd_abd_contact_start_id
                                    + write_offset + 10 * body_hessian_size;
        if(bcooNum > 0)
            pris_drv_output_start += 16 * bcooNum;
        pris_drv_output_start += joint_triplet_blocks;
        pris_drv_output_start += revolute_driving_triplet_blocks;
        pris_drv_output_start += prismatic_triplet_blocks;

        ParallelFor(256)
            .kernel_name("write_prismatic_driving_cross_hessian")
            .apply(m_num_prismatic_driving,
                   [drvs          = m_prismatic_driving_data.cviewer().name("pris_drv_data"),
                    cross_hessian = m_prismatic_driving_cross_hessian.cviewer().name("pris_drv_cross_hessian"),
                    triplet_out   = global_triplets.block_values(),
                    row_out       = global_triplets.block_row_indices(),
                    col_out       = global_triplets.block_col_indices(),
                    pris_drv_output_start] __device__(int j) mutable
                   {
                       auto& drv = drvs(j);
                       int   pid = drv.parent_body_id;
                       int   cid = drv.child_body_id;

                       auto H_pc = cross_hessian(j);

                       int offset = pris_drv_output_start + j * 16;
                       write_cross_body_triplet_cv2(
                           triplet_out, row_out, col_out, pid, cid, H_pc, offset);
                   });
    }

    // Write bilateral stitch spring ABD-FEM cross-Hessian triplets
    if(m_stitch_count > 0 && m_d_stitch_paired_vertex)
    {
        auto fem_point_offset = sim_data.abd_fem_count_info().abd_point_num;
        auto abd_pt_offset    = sim_data.abd_fem_count_info().abd_point_offset;
        int stitch_output_start = new_triplet_offset + h_abd_abd_contact_start_id
                                  + write_offset + 10 * body_hessian_size;
        if(bcooNum > 0)
            stitch_output_start += 16 * bcooNum;
        stitch_output_start += joint_triplet_blocks;
        stitch_output_start += revolute_driving_triplet_blocks;
        stitch_output_start += prismatic_triplet_blocks;
        stitch_output_start += prismatic_driving_triplet_blocks;

        ParallelFor(256)
            .kernel_name("write_stitch_cross_hessian")
            .apply(m_stitch_count,
                   [stitch_paired_vertex = m_d_stitch_paired_vertex,
                    stitch_abd_body_id   = m_d_stitch_abd_body_id,
                    stitch_fem_vertex_id = m_d_stitch_fem_vertex_id,
                    Js                   = sim_data.device.unique_point_id_to_J.cviewer().name("Js"),
                    is_fixed  = body_id_is_fixed.cviewer().name("is_fixed"),
                    triplet_out = global_triplets.block_values(),
                    row_out     = global_triplets.block_row_indices(),
                    col_out     = global_triplets.block_col_indices(),
                    abd_body_count,
                    fem_point_offset,
                    abd_pt_offset,
                    motionRate     = m_stitch_motion_rate,
                    rate           = m_stitch_rate,
                    stitch_output_start] __device__(int i) mutable
                   {
                       int abd_point_id = stitch_paired_vertex[i];
                       if(abd_point_id < 0)
                       {
                           // Not a stitch spring — zero out the 4 reserved blocks
                           int offset = stitch_output_start + i * 4;
                           for(int b = 0; b < 4; ++b)
                           {
                               triplet_out[offset + b].setZero();
                               row_out[offset + b] = 0;
                               col_out[offset + b] = 0;
                           }
                           return;
                       }

                       int body_id = stitch_abd_body_id[i];
                       double k = motionRate * rate * rate;

                       // J^T: 12x3 matrix
                       int local_abd_point_id = abd_point_id - static_cast<int>(abd_pt_offset);
                       gipc::ABDJacobi J = Js(local_abd_point_id);
                       gipc::Matrix3x12 Jmat = J.to_mat();

                       // Cross-Hessian = -k * J^T (12x3)
                       // Written as 4 blocks of 3x3: H_block[b] = -k * Jmat^T rows [3b..3b+2]
                       uint32_t fem_vid = stitch_fem_vertex_id[i];
                       int local_fem_id = static_cast<int>(fem_vid) - static_cast<int>(fem_point_offset);
                       int col_idx = abd_body_count * 4 + local_fem_id;

                       bool is_zero = (is_fixed(body_id) == BodyBoundaryType::Fixed);

                       int offset = stitch_output_start + i * 4;
                       for(int b = 0; b < 4; ++b)
                       {
                           if(is_zero)
                           {
                               triplet_out[offset + b].setZero();
                           }
                           else
                           {
                               // Block (b, 0) of J^T: rows [3b..3b+2], cols [0..2]
                               // J^T = Jmat^T, so J^T[3b+r][c] = Jmat[c][3b+r]
                               Eigen::Matrix3d block;
                               for(int r = 0; r < 3; ++r)
                                   for(int c = 0; c < 3; ++c)
                                       block(r, c) = -k * Jmat(c, b * 3 + r);
                               triplet_out[offset + b] = block;
                           }
                           row_out[offset + b] = body_id * 4 + b;
                           col_out[offset + b] = col_idx;
                       }
                   });
    }


    global_triplets.global_collision_triplet_offset = new_triplet_offset;
    global_triplets.global_triplet_offset = global_triplets.global_collision_triplet_offset;


    CUDA_SAFE_CALL(cudaMemcpy(
        global_triplets.block_values() + global_triplets.fem_fem_contact_num,
        global_triplets.block_values() + new_triplet_offset + global_triplets.fem_fem_contact_num,
        (new_triplet_offset - global_triplets.fem_fem_contact_num) * sizeof(Eigen::Matrix3d),
        cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(
        global_triplets.block_col_indices() + global_triplets.fem_fem_contact_num,
        global_triplets.block_col_indices() + new_triplet_offset
            + global_triplets.fem_fem_contact_num,
        (new_triplet_offset - global_triplets.fem_fem_contact_num) * sizeof(int),
        cudaMemcpyDeviceToDevice));

    CUDA_SAFE_CALL(cudaMemcpy(
        global_triplets.block_row_indices() + global_triplets.fem_fem_contact_num,
        global_triplets.block_row_indices() + new_triplet_offset
            + global_triplets.fem_fem_contact_num,
        (new_triplet_offset - global_triplets.fem_fem_contact_num) * sizeof(int),
        cudaMemcpyDeviceToDevice));
}

// ============================================================================
// Joint Constraint Gradient & Hessian
// ============================================================================

void ABDSystem::_cal_abd_joint_gradient_and_hessian(ABDSimData& sim_data)
{
    if(m_num_joints == 0)
        return;

    gipc::Timer timer("_cal_abd_joint_gradient_and_hessian");
    using namespace muda;

    auto& abd = sim_data.device;
    auto  kappa_fallback = parms.joint_strength_ratio;  // fallback (per-joint kappa takes priority)
    auto  body_id_is_fixed = sim_data.body_id_to_boundary_type();

    m_joint_cross_hessian.resize(m_num_joints);

    ParallelFor(256)
        .kernel_name(__FUNCTION__)
        .apply(m_num_joints,
               [joints          = m_joint_data.cviewer().name("joint_data"),
                qs              = abd.body_id_to_q.cviewer().name("qs"),
                affine_gradient = abd_gradient.viewer().name("abd_gradient"),
                system_gradient = system_gradient.viewer().name("system_gradient"),
                body_hessian    = abd_body_hessian.viewer().name("abd_body_hessian"),
                cross_hessian   = m_joint_cross_hessian.viewer().name("joint_cross_hessian"),
                is_fixed        = body_id_is_fixed.cviewer().name("is_fixed"),
                kappa_fallback] __device__(int j) mutable
               {
                   auto& joint = joints(j);
                   int   pid   = joint.parent_body_id;
                   int   cid   = joint.child_body_id;

                   auto& q_parent = qs(pid);
                   auto& q_child  = qs(cid);

                   bool parent_fixed = (is_fixed(pid) == BodyBoundaryType::Fixed);
                   bool child_fixed  = (is_fixed(cid) == BodyBoundaryType::Fixed);

                   Vector12 grad_parent, grad_child;
                   joint_constraint_gradient(joint, q_parent, q_child, kappa_fallback,
                                             grad_parent, grad_child);

                   Matrix12x12 H_pp, H_cc, H_pc;
                   joint_constraint_hessian(joint, kappa_fallback, H_pp, H_cc, H_pc);

                   // Add gradient to parent body (skip if fixed)
                   if(!parent_fixed)
                   {
                       eigen::atomic_add(affine_gradient(pid), grad_parent);
                       bin_add12_off(g_abd_sysbin, pid * 12, grad_parent);
                       // Add self-body Hessian
                       bin_add144(g_abd_hessbin, pid, H_pp);
                   }

                   // Add gradient to child body (skip if fixed)
                   if(!child_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad_child);
                       bin_add12_off(g_abd_sysbin, cid * 12, grad_child);
                       // Add self-body Hessian
                       bin_add144(g_abd_hessbin, cid, H_cc);
                   }

                   // Store cross-body Hessian for triplet writing
                   // If either body is fixed, zero out the cross term
                   if(parent_fixed || child_fixed)
                       cross_hessian(j) = Matrix12x12::Zero();
                   else
                       cross_hessian(j) = H_pc;
               });
}


// ============================================================================
// Joint Constraint Initialization
// ============================================================================

void ABDSystem::init_joint_constraints(
    ABDSimData& sim_data,
    const std::vector<JointConstraintHostInfo>& host_joints)
{
    m_num_joints = static_cast<int>(host_joints.size());
    if(m_num_joints == 0)
        return;

    auto& abd = sim_data.device;

    // Build host-side GPU data: world-space positions for now
    std::vector<JointConstraintGPUData> host_gpu_data(m_num_joints);

    for(int j = 0; j < m_num_joints; j++)
    {
        auto& hj     = host_joints[j];
        auto& gj     = host_gpu_data[j];
        gj.parent_body_id = hj.parent_body_id;
        gj.child_body_id  = hj.child_body_id;
        gj.num_points     = hj.num_points;
        gj.kappa      = 0.0;

        for(int k = 0; k < kMaxJointConstraintPoints; k++)
        {
            if(k < hj.num_points)
            {
                gj.parent_xbar[k] = Vector3(hj.world_anchor[k].x(),
                                              hj.world_anchor[k].y(),
                                              hj.world_anchor[k].z());
                gj.child_xbar[k]  = gj.parent_xbar[k];
                gj.point_weight[k] = static_cast<Float>(hj.point_weight[k]);
            }
            else
            {
                gj.parent_xbar[k] = Vector3::Zero();
                gj.child_xbar[k]  = Vector3::Zero();
                gj.point_weight[k] = 0.0;
            }
        }

        gj.has_direction_constraint = hj.has_direction_constraint ? 1 : 0;
        if(hj.has_direction_constraint)
        {
            // Store world-space directions temporarily; converted to material below
            gj.parent_t_bar = Vector3(hj.world_tangent.x(), hj.world_tangent.y(), hj.world_tangent.z());
            gj.child_t_bar  = gj.parent_t_bar;
            gj.parent_n_bar = Vector3(hj.world_normal.x(), hj.world_normal.y(), hj.world_normal.z());
            gj.child_n_bar  = gj.parent_n_bar;
            gj.parent_b_bar = Vector3(hj.world_bitangent.x(), hj.world_bitangent.y(), hj.world_bitangent.z());
            gj.child_b_bar  = gj.parent_b_bar;
        }
        else
        {
            gj.parent_t_bar = gj.child_t_bar = Vector3::Zero();
            gj.parent_n_bar = gj.child_n_bar = Vector3::Zero();
            gj.parent_b_bar = gj.child_b_bar = Vector3::Zero();
        }
    }

    m_joint_data.resize(m_num_joints);
    m_joint_data.view().copy_from(host_gpu_data.data());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Convert world-space positions to material coordinates and compute
    // mass-based stiffness: kappa = strength_ratio * (m_parent + m_child).
    // Matches rbs-uipc: joint energies have NO dt² factor — they act as stiff
    // penalty terms relative to kinetic energy in the IP formulation.
    using namespace muda;
    Float sr  = parms.joint_strength_ratio;

    ParallelFor(256)
        .kernel_name("convert_joint_world_to_material")
        .apply(m_num_joints,
               [joints    = m_joint_data.viewer().name("joint_data"),
                qs        = abd.body_id_to_q.cviewer().name("qs"),
                masses    = body_mass.cviewer().name("body_mass"),
                sr] __device__(int j) mutable
               {
                   auto& joint = joints(j);
                   int   pid   = joint.parent_body_id;
                   int   cid   = joint.child_body_id;

                   Float mass_sum = masses(pid) + masses(cid);
                   joint.kappa = sr * mass_sum;

                   Vector3 p_parent = qs(pid).segment<3>(0);
                   Vector3 p_child  = qs(cid).segment<3>(0);

                   Matrix3x3 A_parent;
                   A_parent.row(0) = qs(pid).segment<3>(3).transpose();
                   A_parent.row(1) = qs(pid).segment<3>(6).transpose();
                   A_parent.row(2) = qs(pid).segment<3>(9).transpose();

                   Matrix3x3 A_child;
                   A_child.row(0) = qs(cid).segment<3>(3).transpose();
                   A_child.row(1) = qs(cid).segment<3>(6).transpose();
                   A_child.row(2) = qs(cid).segment<3>(9).transpose();

                   Matrix3x3 A_parent_inv = eigen::inverse(A_parent);
                   Matrix3x3 A_child_inv  = eigen::inverse(A_child);

                   for(int k = 0; k < joint.num_points; k++)
                   {
                       Vector3 world_pos = joint.parent_xbar[k];
                       joint.parent_xbar[k] = A_parent_inv * (world_pos - p_parent);
                       joint.child_xbar[k]  = A_child_inv * (world_pos - p_child);
                   }

                   if(joint.has_direction_constraint)
                   {
                       // Direction vectors: d_bar = A_inv * d_world (rotation only)
                       joint.parent_t_bar = A_parent_inv * joint.parent_t_bar;
                       joint.child_t_bar  = A_child_inv  * joint.child_t_bar;
                       joint.parent_n_bar = A_parent_inv * joint.parent_n_bar;
                       joint.child_n_bar  = A_child_inv  * joint.child_n_bar;
                       joint.parent_b_bar = A_parent_inv * joint.parent_b_bar;
                       joint.child_b_bar  = A_child_inv  * joint.child_b_bar;
                   }
               });

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Log computed stiffness for the first joint
    if(m_num_joints > 0)
    {
        std::vector<JointConstraintGPUData> dbg(1);
        m_joint_data.view().subview(0, 1).copy_to(dbg.data());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        std::cout << "[ABDSystem] Initialized " << m_num_joints << " joint constraints "
                  << "(strength_ratio=" << sr
                  << ", kappa[0]=" << dbg[0].kappa << ")." << std::endl;
    }
}


// ============================================================================
// Revolute Driving Joint Initialization
// ============================================================================

void ABDSystem::init_revolute_driving(
    ABDSimData& sim_data,
    const std::vector<JointAngleControlInfo>& controls,
    const std::vector<JointConstraintHostInfo>& host_joints)
{
    m_num_revolute_driving = static_cast<int>(controls.size());
    if(m_num_revolute_driving == 0)
        return;

    auto& abd = sim_data.device;

    std::vector<RevoluteDrivingGPUData> host_data(m_num_revolute_driving);

    for(int i = 0; i < m_num_revolute_driving; i++)
    {
        auto& ctrl = controls[i];
        auto& drv  = host_data[i];

        int ji = ctrl.constraint_index;
        if(ji < 0 || ji >= static_cast<int>(host_joints.size()))
            continue;

        auto& hj = host_joints[ji];
        drv.parent_body_id = hj.parent_body_id;
        drv.child_body_id  = hj.child_body_id;

        Eigen::Vector3d axis = ctrl.axis_dir.normalized();
        Eigen::Vector3d n    = ctrl.n_dir.normalized();
        Eigen::Vector3d m    = axis.cross(n).normalized();

        drv.p_bar  = Vector3(n.x(), n.y(), n.z());
        drv.pN_bar = Vector3(m.x(), m.y(), m.z());
        drv.q_bar  = Vector3(n.x(), n.y(), n.z());
        drv.qN_bar = Vector3(m.x(), m.y(), m.z());

        drv.stiffness              = 0.0;  // computed on GPU using body masses
        drv.initial_angle_offset   = static_cast<Float>(ctrl.initial_angle_offset);
        drv.target_angle           = static_cast<Float>(ctrl.target_angle - ctrl.initial_angle_offset);
        drv.ext_torque             = static_cast<Float>(ctrl.ext_torque);  // [force-control]
        // [joint limit FIX] the ctrl's URDF/API limits were NEVER copied into
        // the GPU driving data — drv.lower/upper stayed at their +-1e30 "none"
        // defaults, so the limit penalty could never fire (a limited passive
        // hinge swung straight through its bounds; regression
        // test_passive_revolute case B). Same relative frame as target_angle:
        // shift by initial_angle_offset.
        drv.lower_limit = static_cast<Float>(ctrl.lower_limit - ctrl.initial_angle_offset);
        drv.upper_limit = static_cast<Float>(ctrl.upper_limit - ctrl.initial_angle_offset);
        // [audit v0.8.5.1] the runtime angle is atan2-based, range (-pi, pi].
        // A REAL limit (not the +-1e30 "none" sentinel) shifted outside that
        // range by a large initial_angle_offset can never fire — warn loudly at
        // init instead of silently disabling one side of the limit.
        {
            const double kPi = 3.14159265358979323846;
            const bool real_lo = std::abs(ctrl.lower_limit) < 1e29;
            const bool real_up = std::abs(ctrl.upper_limit) < 1e29;
            if((real_lo && drv.lower_limit < -kPi) || (real_up && drv.upper_limit > kPi))
                fprintf(stderr,
                        "[revolute-limit][WARN] joint %d: limit(s) shifted by "
                        "initial_angle_offset=%.6f land outside the atan2 angle "
                        "range (-pi, pi]: lower=%.6f upper=%.6f — the out-of-range "
                        "side(s) can NEVER trigger. Re-express the limits relative "
                        "to the initial pose.\n",
                        ji, ctrl.initial_angle_offset,
                        (double)drv.lower_limit, (double)drv.upper_limit);
        }
    }

    m_revolute_driving_data.resize(m_num_revolute_driving);
    m_revolute_driving_data.view().copy_from(host_data.data());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Compute mass-based stiffness and convert directions to material space on GPU.
    // K = strength_ratio * ctrl.strength_ratio * (m_parent + m_child)
    // Matches rbs-uipc: kappa = strength_ratio * (m_i + m_j), NO dt² factor.
    // Joint energies are intentionally NOT scaled by dt² so they act as very
    // stiff penalty terms relative to kinetic energy in the IP formulation.
    using namespace muda;
    Float sr  = parms.revolute_driving_strength_ratio;
    Float lsr = parms.joint_limit_strength_ratio;  // [joint limit] mass-scaled penalty stiffness

    // Upload per-joint strength ratios to a temp buffer
    std::vector<Float> host_ctrl_sr(m_num_revolute_driving, 1.0);
    for(int i = 0; i < m_num_revolute_driving && i < static_cast<int>(controls.size()); i++)
        host_ctrl_sr[i] = static_cast<Float>(controls[i].strength_ratio);

    muda::DeviceBuffer<Float> d_ctrl_sr(m_num_revolute_driving);
    d_ctrl_sr.view().copy_from(host_ctrl_sr.data());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    ParallelFor(256)
        .kernel_name("init_driving_stiffness_and_material_dirs")
        .apply(m_num_revolute_driving,
               [drvs     = m_revolute_driving_data.viewer().name("revolute_driving"),
                qs       = abd.body_id_to_q.cviewer().name("qs"),
                masses   = body_mass.cviewer().name("body_mass"),
                ctrl_sr  = d_ctrl_sr.cviewer().name("ctrl_sr"),
                sr, lsr] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrl_sr(i) * mass_sum;
                   drv.limit_stiffness = lsr * mass_sum;  // [joint limit] mass-scaled penalty

                   Matrix3x3 A_parent;
                   A_parent.row(0) = qs(pid).segment<3>(3).transpose();
                   A_parent.row(1) = qs(pid).segment<3>(6).transpose();
                   A_parent.row(2) = qs(pid).segment<3>(9).transpose();

                   Matrix3x3 A_child;
                   A_child.row(0) = qs(cid).segment<3>(3).transpose();
                   A_child.row(1) = qs(cid).segment<3>(6).transpose();
                   A_child.row(2) = qs(cid).segment<3>(9).transpose();

                   Matrix3x3 A_parent_inv = eigen::inverse(A_parent);
                   Matrix3x3 A_child_inv  = eigen::inverse(A_child);

                   drv.p_bar  = (A_parent_inv * drv.p_bar).normalized();
                   drv.pN_bar = (A_parent_inv * drv.pN_bar).normalized();
                   drv.q_bar  = (A_child_inv  * drv.q_bar).normalized();
                   drv.qN_bar = (A_child_inv  * drv.qN_bar).normalized();
               });

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    if(m_num_revolute_driving > 0)
    {
        std::vector<RevoluteDrivingGPUData> dbg(1);
        m_revolute_driving_data.view().subview(0, 1).copy_to(dbg.data());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        std::cout << "[ABDSystem] Initialized " << m_num_revolute_driving
                  << " revolute driving joints (strength_ratio=" << sr
                  << ", K[0]=" << dbg[0].stiffness << ")." << std::endl;
    }
}


// ============================================================================
// Update Revolute Driving Targets (per frame from UI)
// ============================================================================

void ABDSystem::update_revolute_driving_targets(
    ABDSimData& sim_data,
    const std::vector<JointAngleControlInfo>& controls,
    double substep_ratio)
{
    if(controls.empty() || m_num_revolute_driving == 0)
        return;

    using namespace muda;
    auto& abd = sim_data.device;
    int n = m_num_revolute_driving;

    std::vector<DrivingCtrlPacked> host_ctrl(n);
    for(int i = 0; i < n && i < static_cast<int>(controls.size()); i++)
    {
        host_ctrl[i].target_angle   = static_cast<Float>(controls[i].target_angle);
        host_ctrl[i].strength_ratio = static_cast<Float>(controls[i].strength_ratio);
        host_ctrl[i].ext_torque     = static_cast<Float>(controls[i].ext_torque);  // [force-control]
    }

    muda::DeviceBuffer<DrivingCtrlPacked> d_ctrl(n);
    d_ctrl.view().copy_from(host_ctrl.data());

    Float kMaxStepPerFrame = parms.max_revolute_step_per_frame;
    Float sr = parms.revolute_driving_strength_ratio;

    ParallelFor(256)
        .kernel_name("update_revolute_driving_targets_gpu")
        .apply(n,
               [drvs    = m_revolute_driving_data.viewer().name("revolute_driving"),
                q_prev  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                masses  = body_mass.cviewer().name("body_mass"),
                ctrls   = d_ctrl.cviewer().name("ctrls"),
                sr, kMaxStepPerFrame,
                ratio = static_cast<Float>(substep_ratio)] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   const auto& q1 = q_prev(pid);
                   const auto& q2 = q_prev(cid);

                   Matrix3x3 A1, A2;
                   A1.row(0) = q1.segment<3>(3).transpose();
                   A1.row(1) = q1.segment<3>(6).transpose();
                   A1.row(2) = q1.segment<3>(9).transpose();
                   A2.row(0) = q2.segment<3>(3).transpose();
                   A2.row(1) = q2.segment<3>(6).transpose();
                   A2.row(2) = q2.segment<3>(9).transpose();

                   Vector3 p  = A1 * drv.p_bar;
                   Vector3 pN = A1 * drv.pN_bar;
                   Vector3 q  = A2 * drv.q_bar;
                   Vector3 qN = A2 * drv.qN_bar;

                   Float cos_prev = Float(0.5) * (p.dot(q) + pN.dot(qN));
                   Float sin_prev = Float(0.5) * (q.dot(pN) - qN.dot(p));
                   Float theta_prev = atan2(sin_prev, cos_prev);

                   Float desired_goal = ctrls(i).target_angle - drv.initial_angle_offset;
                   Float diff = desired_goal - theta_prev;
                   const Float TWO_PI = Float(2.0 * 3.14159265358979323846);
                   diff = diff - TWO_PI * round(diff / TWO_PI);
                   diff = (diff >  kMaxStepPerFrame) ?  kMaxStepPerFrame :
                          (diff < -kMaxStepPerFrame) ? -kMaxStepPerFrame : diff;

                   drv.target_angle = theta_prev + diff * ratio;  // [drive-substep]

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrls(i).strength_ratio * mass_sum;
                   drv.ext_torque = ctrls(i).ext_torque;  // [force-control] live sync

                   // [limit lagged active-set] once-per-frame state machine on
                   // theta_prev (last frame's CONVERGED angle):
                   //   engage  : theta actually crossed a bound;
                   //   release : the converged angle sits back INSIDE the bound
                   //             — with a two-sided frame spring that can only
                   //             happen when the spring ended up PULLING the
                   //             joint toward the bound (multiplier sign test:
                   //             a one-sided limit must never pull). A pressed
                   //             joint converges slightly OUTSIDE the bound
                   //             (depth ~ torque/K_lim), so it stays engaged
                   //             with no flip-flop.
                   if(drv.limit_stiffness > Float(0))
                   {
                       if(drv.limit_active == 0)
                       {
                           if(theta_prev < drv.lower_limit)      drv.limit_active = -1;
                           else if(theta_prev > drv.upper_limit) drv.limit_active = +1;
                       }
                       else if(drv.limit_active < 0)
                       {
                           if(theta_prev > drv.lower_limit) drv.limit_active = 0;
                       }
                       else
                       {
                           if(theta_prev < drv.upper_limit) drv.limit_active = 0;
                       }
                   }
               });
}


// ============================================================================
// Revolute Driving Energy
// ============================================================================

Float ABDSystem::cal_abd_revolute_driving_energy(ABDSimData& sim_data,
                                                  bool copy_to_host)
{
    using namespace muda;
    if(m_num_revolute_driving == 0)
        return 0;

    auto& abd = sim_data.device;
    m_revolute_driving_energy_per.resize(m_num_revolute_driving);

    ParallelFor()
        .kernel_name("cal_revolute_driving_energy")
        .apply(m_num_revolute_driving,
               [energies = m_revolute_driving_energy_per.viewer().name("energies"),
                drvs     = m_revolute_driving_data.cviewer().name("drvs"),
                qs       = abd.body_id_to_q.cviewer().name("qs")] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   energies(i) = revolute_driving_energy(drv, qs(drv.parent_body_id),
                                                              qs(drv.child_body_id))
                                 // [joint limit FIX] the one-sided limit energy
                                 // was implemented but NEVER wired in — add it
                                 // so limits act (incl. passive joints, whose
                                 // driving term is zero).
                                 + revolute_limit_energy(drv, qs(drv.parent_body_id),
                                                              qs(drv.child_body_id));
               });

    muda::DeviceReduce().Sum(
        m_revolute_driving_energy_per.data(), m_revolute_driving_energy.data(),
        m_num_revolute_driving);

    return copy_to_host ? Float(m_revolute_driving_energy) : 0.0;
}


// ============================================================================
// Revolute Driving Gradient & Hessian
// ============================================================================

void ABDSystem::_cal_abd_revolute_driving_gradient_and_hessian(ABDSimData& sim_data)
{
    if(m_num_revolute_driving == 0)
        return;

    using namespace muda;
    auto& abd = sim_data.device;
    auto  body_id_is_fixed = sim_data.body_id_to_boundary_type();

    m_revolute_driving_cross_hessian.resize(m_num_revolute_driving);

    ParallelFor(256)
        .kernel_name("cal_revolute_driving_grad_hess")
        .apply(m_num_revolute_driving,
               [drvs            = m_revolute_driving_data.cviewer().name("drvs"),
                qs               = abd.body_id_to_q.cviewer().name("qs"),
                affine_gradient  = abd_gradient.viewer().name("abd_gradient"),
                sys_gradient     = system_gradient.viewer().name("system_gradient"),
                body_hessian     = abd_body_hessian.viewer().name("abd_body_hessian"),
                cross_hessian    = m_revolute_driving_cross_hessian.viewer().name("drv_cross_hessian"),
                is_fixed         = body_id_is_fixed.cviewer().name("is_fixed")] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   auto& q1 = qs(pid);
                   auto& q2 = qs(cid);

                   bool p_fixed = (is_fixed(pid) == BodyBoundaryType::Fixed);
                   bool c_fixed = (is_fixed(cid) == BodyBoundaryType::Fixed);

                   Vector12 grad1, grad2;
                   revolute_driving_gradient(drv, q1, q2, grad1, grad2);

                   Matrix12x12 H_11, H_22, H_12;
                   revolute_driving_hessian(drv, q1, q2, H_11, H_22, H_12);

                   // [joint limit FIX] wire the (previously dead) one-sided
                   // limit gradient + SPD Gauss-Newton Hessian into the same
                   // scatter. Zero when within [lower, upper].
                   {
                       Vector12    lg1, lg2;
                       Matrix12x12 lH11, lH22, lH12;
                       revolute_limit_gradient_hessian(drv, q1, q2, lg1, lg2,
                                                       lH11, lH22, lH12);
                       grad1 += lg1; grad2 += lg2;
                       H_11 += lH11; H_22 += lH22; H_12 += lH12;
                   }

                   if(!p_fixed)
                   {
                       eigen::atomic_add(affine_gradient(pid), grad1);
                       bin_add12_off(g_abd_sysbin, pid * 12, grad1);
                       bin_add144(g_abd_hessbin, pid, H_11);
                   }

                   if(!c_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad2);
                       bin_add12_off(g_abd_sysbin, cid * 12, grad2);
                       bin_add144(g_abd_hessbin, cid, H_22);
                   }

                   if(p_fixed || c_fixed)
                       cross_hessian(i) = Matrix12x12::Zero();
                   else
                       cross_hessian(i) = H_12;
               });
}


// ============================================================================
// Prismatic Joint Constraint Initialization
// ============================================================================

void ABDSystem::init_prismatic_constraints(
    ABDSimData& sim_data,
    const std::vector<PrismaticJointHostInfo>& host_prismatic)
{
    m_num_prismatic = static_cast<int>(host_prismatic.size());
    if(m_num_prismatic == 0)
        return;

    auto& abd = sim_data.device;

    std::vector<PrismaticJointGPUData> host_gpu(m_num_prismatic);

    for(int i = 0; i < m_num_prismatic; i++)
    {
        auto& hp = host_prismatic[i];
        auto& gp = host_gpu[i];

        gp.parent_body_id = hp.parent_body_id;
        gp.child_body_id  = hp.child_body_id;

        gp.Cp_bar = Vector3(hp.world_center.x(), hp.world_center.y(), hp.world_center.z());
        gp.Cq_bar = gp.Cp_bar;

        Eigen::Vector3d t = hp.world_axis.normalized();
        Eigen::Vector3d n = hp.world_normal.normalized();
        Eigen::Vector3d b = hp.world_bitangent.normalized();

        gp.tp_bar = Vector3(t.x(), t.y(), t.z());
        gp.tq_bar = gp.tp_bar;
        gp.np_bar = Vector3(n.x(), n.y(), n.z());
        gp.nq_bar = gp.np_bar;
        gp.bp_bar = Vector3(b.x(), b.y(), b.z());
        gp.bq_bar = gp.bp_bar;
    }

    m_prismatic_data.resize(m_num_prismatic);
    m_prismatic_data.view().copy_from(host_gpu.data());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    Float sr = parms.prismatic_strength_ratio;

    using namespace muda;
    ParallelFor(256)
        .kernel_name("init_prismatic_constraints_material")
        .apply(m_num_prismatic,
               [prisms = m_prismatic_data.viewer().name("prismatic_data"),
                qs     = abd.body_id_to_q.cviewer().name("qs"),
                masses = body_mass.cviewer().name("body_mass"),
                sr] __device__(int i) mutable
               {
                   auto& pj = prisms(i);
                   int pid = pj.parent_body_id;
                   int cid = pj.child_body_id;

                   Matrix3x3 Ap, Ac;
                   Ap.row(0) = qs(pid).segment<3>(3).transpose();
                   Ap.row(1) = qs(pid).segment<3>(6).transpose();
                   Ap.row(2) = qs(pid).segment<3>(9).transpose();
                   Ac.row(0) = qs(cid).segment<3>(3).transpose();
                   Ac.row(1) = qs(cid).segment<3>(6).transpose();
                   Ac.row(2) = qs(cid).segment<3>(9).transpose();

                   Matrix3x3 Ap_inv = eigen::inverse(Ap);
                   Matrix3x3 Ac_inv = eigen::inverse(Ac);

                   Vector3 pp = qs(pid).segment<3>(0);
                   Vector3 pc = qs(cid).segment<3>(0);

                   pj.Cp_bar = Ap_inv * (pj.Cp_bar - pp);
                   pj.Cq_bar = Ac_inv * (pj.Cq_bar - pc);

                   pj.tp_bar = (Ap_inv * pj.tp_bar).normalized();
                   pj.tq_bar = (Ac_inv * pj.tq_bar).normalized();
                   pj.np_bar = (Ap_inv * pj.np_bar).normalized();
                   pj.nq_bar = (Ac_inv * pj.nq_bar).normalized();
                   pj.bp_bar = (Ap_inv * pj.bp_bar).normalized();
                   pj.bq_bar = (Ac_inv * pj.bq_bar).normalized();
               });
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::cout << "[ABDSystem] Initialized " << m_num_prismatic
              << " prismatic joint constraints (strength_ratio=" << sr << ")." << std::endl;
}


// ============================================================================
// Prismatic Joint Constraint Energy
// ============================================================================

Float ABDSystem::cal_abd_prismatic_energy(ABDSimData& sim_data, bool copy_to_host)
{
    using namespace muda;
    if(m_num_prismatic == 0)
        return 0;

    auto& abd = sim_data.device;
    Float kappa = parms.prismatic_strength_ratio;
    m_prismatic_energy_per.resize(m_num_prismatic);

    ParallelFor()
        .kernel_name("cal_prismatic_energy")
        .apply(m_num_prismatic,
               [energies = m_prismatic_energy_per.viewer().name("energies"),
                prisms   = m_prismatic_data.cviewer().name("prisms"),
                qs       = abd.body_id_to_q.cviewer().name("qs"),
                masses   = body_mass.cviewer().name("body_mass"),
                kappa] __device__(int i) mutable
               {
                   auto& pj = prisms(i);
                   Float K = kappa * (masses(pj.parent_body_id) + masses(pj.child_body_id));
                   energies(i) = prismatic_constraint_energy(pj, qs(pj.parent_body_id),
                                                                  qs(pj.child_body_id), K);
               });

    muda::DeviceReduce().Sum(
        m_prismatic_energy_per.data(), m_prismatic_energy.data(), m_num_prismatic);

    return copy_to_host ? Float(m_prismatic_energy) : 0.0;
}


// ============================================================================
// Prismatic Joint Constraint Gradient & Hessian
// ============================================================================

void ABDSystem::_cal_abd_prismatic_gradient_and_hessian(ABDSimData& sim_data)
{
    if(m_num_prismatic == 0)
        return;

    using namespace muda;
    auto& abd = sim_data.device;
    auto  body_id_is_fixed = sim_data.body_id_to_boundary_type();
    Float kappa = parms.prismatic_strength_ratio;

    m_prismatic_cross_hessian.resize(m_num_prismatic);

    ParallelFor(256)
        .kernel_name("cal_prismatic_grad_hess")
        .apply(m_num_prismatic,
               [prisms          = m_prismatic_data.cviewer().name("prisms"),
                qs              = abd.body_id_to_q.cviewer().name("qs"),
                affine_gradient = abd_gradient.viewer().name("abd_gradient"),
                sys_gradient    = system_gradient.viewer().name("system_gradient"),
                body_hessian    = abd_body_hessian.viewer().name("abd_body_hessian"),
                cross_hessian   = m_prismatic_cross_hessian.viewer().name("pris_cross_hessian"),
                is_fixed        = body_id_is_fixed.cviewer().name("is_fixed"),
                masses          = body_mass.cviewer().name("body_mass"),
                kappa] __device__(int i) mutable
               {
                   auto& pj = prisms(i);
                   int pid = pj.parent_body_id;
                   int cid = pj.child_body_id;

                   Float K = kappa * (masses(pid) + masses(cid));

                   auto& q1 = qs(pid);
                   auto& q2 = qs(cid);

                   bool p_fixed = (is_fixed(pid) == BodyBoundaryType::Fixed);
                   bool c_fixed = (is_fixed(cid) == BodyBoundaryType::Fixed);

                   Vector12 grad1, grad2;
                   Matrix12x12 H_pp, H_qq, H_pq;
                   prismatic_constraint_gradient_hessian(pj, q1, q2, K,
                                                         grad1, grad2, H_pp, H_qq, H_pq);

                   if(!p_fixed)
                   {
                       eigen::atomic_add(affine_gradient(pid), grad1);
                       bin_add12_off(g_abd_sysbin, pid * 12, grad1);
                       bin_add144(g_abd_hessbin, pid, H_pp);
                   }

                   if(!c_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad2);
                       bin_add12_off(g_abd_sysbin, cid * 12, grad2);
                       bin_add144(g_abd_hessbin, cid, H_qq);
                   }

                   if(p_fixed || c_fixed)
                       cross_hessian(i) = Matrix12x12::Zero();
                   else
                       cross_hessian(i) = H_pq;
               });
}


// ============================================================================
// Prismatic Driving Joint Initialization
// ============================================================================

void ABDSystem::init_prismatic_driving(
    ABDSimData& sim_data,
    const std::vector<PrismaticDrivingControlInfo>& controls,
    const std::vector<PrismaticJointHostInfo>& host_prismatic)
{
    m_num_prismatic_driving = static_cast<int>(controls.size());
    if(m_num_prismatic_driving == 0)
        return;

    auto& abd = sim_data.device;

    std::vector<PrismaticDrivingGPUData> host_data(m_num_prismatic_driving);

    for(int i = 0; i < m_num_prismatic_driving; i++)
    {
        auto& ctrl = controls[i];
        auto& drv  = host_data[i];

        int pi = ctrl.prismatic_constraint_index;
        if(pi < 0 || pi >= static_cast<int>(host_prismatic.size()))
            continue;

        auto& hp = host_prismatic[pi];
        drv.parent_body_id = hp.parent_body_id;
        drv.child_body_id  = hp.child_body_id;

        Eigen::Vector3d c = hp.world_center;
        Eigen::Vector3d t = hp.world_axis.normalized();

        drv.Cp_bar = Vector3(c.x(), c.y(), c.z());
        drv.Cq_bar = drv.Cp_bar;
        drv.tq_bar = Vector3(t.x(), t.y(), t.z());
        // [force-control] parent material axis starts as the same world axis;
        // the kernel below maps it into parent material frame.
        drv.tp_bar = Vector3(t.x(), t.y(), t.z());

        drv.stiffness       = 0.0;
        drv.target_distance = static_cast<Float>(ctrl.target_distance);
        drv.ext_force       = static_cast<Float>(ctrl.ext_force);
    }

    m_prismatic_driving_data.resize(m_num_prismatic_driving);
    m_prismatic_driving_data.view().copy_from(host_data.data());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    Float sr = parms.prismatic_driving_strength_ratio;
    Float lsr = parms.joint_limit_strength_ratio;  // [joint limit] mass-scaled penalty stiffness

    std::vector<Float> host_ctrl_sr(m_num_prismatic_driving, 1.0);
    for(int i = 0; i < m_num_prismatic_driving && i < static_cast<int>(controls.size()); i++)
        host_ctrl_sr[i] = static_cast<Float>(controls[i].strength_ratio);

    muda::DeviceBuffer<Float> d_ctrl_sr(m_num_prismatic_driving);
    d_ctrl_sr.view().copy_from(host_ctrl_sr.data());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    using namespace muda;
    ParallelFor(256)
        .kernel_name("init_prismatic_driving_material")
        .apply(m_num_prismatic_driving,
               [drvs     = m_prismatic_driving_data.viewer().name("prismatic_driving"),
                qs       = abd.body_id_to_q.cviewer().name("qs"),
                masses   = body_mass.cviewer().name("body_mass"),
                ctrl_sr  = d_ctrl_sr.cviewer().name("ctrl_sr"),
                sr, lsr] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrl_sr(i) * mass_sum;
                   drv.pen_limit_stiffness = lsr * mass_sum;  // [joint limit] mass-scaled penalty

                   Matrix3x3 Ap, Ac;
                   Ap.row(0) = qs(pid).segment<3>(3).transpose();
                   Ap.row(1) = qs(pid).segment<3>(6).transpose();
                   Ap.row(2) = qs(pid).segment<3>(9).transpose();
                   Ac.row(0) = qs(cid).segment<3>(3).transpose();
                   Ac.row(1) = qs(cid).segment<3>(6).transpose();
                   Ac.row(2) = qs(cid).segment<3>(9).transpose();

                   Matrix3x3 Ap_inv = eigen::inverse(Ap);
                   Matrix3x3 Ac_inv = eigen::inverse(Ac);

                   Vector3 pp = qs(pid).segment<3>(0);
                   Vector3 pc = qs(cid).segment<3>(0);

                   drv.Cp_bar = Ap_inv * (drv.Cp_bar - pp);
                   drv.Cq_bar = Ac_inv * (drv.Cq_bar - pc);
                   drv.tq_bar = (Ac_inv * drv.tq_bar).normalized();
                   // [force-control] parent material axis (per-body tangent)
                   drv.tp_bar = (Ap_inv * drv.tp_bar).normalized();
               });
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::cout << "[ABDSystem] Initialized " << m_num_prismatic_driving
              << " prismatic driving joints (strength_ratio=" << sr << ")." << std::endl;
}


// ============================================================================
// Update Prismatic Driving Targets (per frame from UI)
// ============================================================================

void ABDSystem::update_prismatic_driving_targets(
    ABDSimData& sim_data,
    const std::vector<PrismaticDrivingControlInfo>& controls,
    double substep_ratio)
{
    if(controls.empty() || m_num_prismatic_driving == 0)
        return;

    using namespace muda;
    auto& abd = sim_data.device;
    int n = m_num_prismatic_driving;

    std::vector<PrisCtrlPacked> host_ctrl(n);
    for(int i = 0; i < n && i < static_cast<int>(controls.size()); i++)
    {
        host_ctrl[i].target_distance = static_cast<Float>(controls[i].target_distance);
        host_ctrl[i].strength_ratio  = static_cast<Float>(controls[i].strength_ratio);
        host_ctrl[i].ext_force       = static_cast<Float>(controls[i].ext_force);
    }

    muda::DeviceBuffer<PrisCtrlPacked> d_ctrl(n);
    d_ctrl.view().copy_from(host_ctrl.data());

    Float kMaxStepPerFrame = parms.max_prismatic_step_per_frame;
    Float sr = parms.prismatic_driving_strength_ratio;

    ParallelFor(256)
        .kernel_name("update_prismatic_driving_targets_gpu")
        .apply(n,
               [drvs    = m_prismatic_driving_data.viewer().name("prismatic_driving"),
                q_prev  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                masses  = body_mass.cviewer().name("body_mass"),
                ctrls   = d_ctrl.cviewer().name("ctrls"),
                sr, kMaxStepPerFrame,
                ratio = static_cast<Float>(substep_ratio)] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   const auto& q1 = q_prev(pid);
                   const auto& q2 = q_prev(cid);

                   Vector3 Cp = ABDJacobi(drv.Cp_bar) * q1;
                   Vector3 Cq = ABDJacobi(drv.Cq_bar) * q2;
                   Matrix3x3 Aq;
                   Aq.row(0) = q2.segment<3>(3).transpose();
                   Aq.row(1) = q2.segment<3>(6).transpose();
                   Aq.row(2) = q2.segment<3>(9).transpose();
                   Vector3 tq = Aq * drv.tq_bar;

                   Float d_prev = (Cq - Cp).dot(tq);

                   Float desired_goal = ctrls(i).target_distance;
                   Float diff = desired_goal - d_prev;
                   diff = (diff >  kMaxStepPerFrame) ?  kMaxStepPerFrame :
                          (diff < -kMaxStepPerFrame) ? -kMaxStepPerFrame : diff;

                   drv.target_distance = d_prev + diff * ratio;  // [drive-substep]

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrls(i).strength_ratio * mass_sum;
                   // [force-control] sync external prismatic force each step
                   drv.ext_force = ctrls(i).ext_force;
               });
}


// ============================================================================
// Prismatic Driving Energy
// ============================================================================

Float ABDSystem::cal_abd_prismatic_driving_energy(ABDSimData& sim_data,
                                                   bool copy_to_host)
{
    using namespace muda;
    if(m_num_prismatic_driving == 0)
        return 0;

    auto& abd = sim_data.device;
    m_prismatic_driving_energy_per.resize(m_num_prismatic_driving);

    ParallelFor()
        .kernel_name("cal_prismatic_driving_energy")
        .apply(m_num_prismatic_driving,
               [energies = m_prismatic_driving_energy_per.viewer().name("energies"),
                drvs     = m_prismatic_driving_data.cviewer().name("drvs"),
                qs       = abd.body_id_to_q.cviewer().name("qs")] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   energies(i) = prismatic_driving_energy(drv, qs(drv.parent_body_id),
                                                               qs(drv.child_body_id));
               });

    muda::DeviceReduce().Sum(
        m_prismatic_driving_energy_per.data(), m_prismatic_driving_energy.data(),
        m_num_prismatic_driving);

    return copy_to_host ? Float(m_prismatic_driving_energy) : 0.0;
}


// ============================================================================
// Prismatic Driving Gradient & Hessian
// ============================================================================

void ABDSystem::_cal_abd_prismatic_driving_gradient_and_hessian(ABDSimData& sim_data)
{
    if(m_num_prismatic_driving == 0)
        return;

    using namespace muda;
    auto& abd = sim_data.device;
    auto  body_id_is_fixed = sim_data.body_id_to_boundary_type();

    m_prismatic_driving_cross_hessian.resize(m_num_prismatic_driving);

    ParallelFor(256)
        .kernel_name("cal_prismatic_driving_grad_hess")
        .apply(m_num_prismatic_driving,
               [drvs            = m_prismatic_driving_data.cviewer().name("drvs"),
                qs              = abd.body_id_to_q.cviewer().name("qs"),
                affine_gradient = abd_gradient.viewer().name("abd_gradient"),
                sys_gradient    = system_gradient.viewer().name("system_gradient"),
                body_hessian    = abd_body_hessian.viewer().name("abd_body_hessian"),
                cross_hessian   = m_prismatic_driving_cross_hessian.viewer().name("pris_drv_cross_hessian"),
                is_fixed        = body_id_is_fixed.cviewer().name("is_fixed")] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   auto& q1 = qs(pid);
                   auto& q2 = qs(cid);

                   bool p_fixed = (is_fixed(pid) == BodyBoundaryType::Fixed);
                   bool c_fixed = (is_fixed(cid) == BodyBoundaryType::Fixed);

                   Vector12 grad1, grad2;
                   Matrix12x12 H_pp, H_qq, H_pq;
                   prismatic_driving_gradient_hessian(drv, q1, q2,
                                                      grad1, grad2, H_pp, H_qq, H_pq);

                   if(!p_fixed)
                   {
                       eigen::atomic_add(affine_gradient(pid), grad1);
                       bin_add12_off(g_abd_sysbin, pid * 12, grad1);
                       bin_add144(g_abd_hessbin, pid, H_pp);
                   }

                   if(!c_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad2);
                       bin_add12_off(g_abd_sysbin, cid * 12, grad2);
                       bin_add144(g_abd_hessbin, cid, H_qq);
                   }

                   if(p_fixed || c_fixed)
                       cross_hessian(i) = Matrix12x12::Zero();
                   else
                       cross_hessian(i) = H_pq;
               });
}


void ABDSystem::_cal_abd_system_preconditioner(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd                        = sim_data.device;
    auto  unique_point_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  body_hessian_size = sim_data.abd_fem_count_info().abd_body_num;

    abd_system_diag_preconditioner.resize(body_hessian_size);
    // Must zero-init: the scatter loop below only touches bodies that appear
    // in the contact triplet range.  Bodies with no barrier/joint coupling
    // (e.g. an isolated free rigid body) would otherwise receive uninitialized
    // stack memory, and a later inverse(P(i)) propagates NaN.  Build the
    // per-body block accumulator on a clean zero buffer, then add the mass
    // diagonal so every body has at least a valid preconditioner.
    abd_system_diag_preconditioner.fill(Matrix12x12::Zero());
    {
        using namespace muda;
        auto Ms = sim_data.device.body_id_to_abd_mass.cviewer().name("M");
        ParallelFor(256)
            .kernel_name("seed_preconditioner_with_mass")
            .apply(body_hessian_size,
                   [P = abd_system_diag_preconditioner.viewer().name("P"),
                    Ms] __device__(int i) mutable
                   { P(i) = Ms(i).to_mat(); });
    }
    auto triplet = global_triplet->block_values(global_triplet->h_abd_abd_contact_start_id);
    auto rows = global_triplet->block_row_indices(global_triplet->h_abd_abd_contact_start_id);
    auto cols = global_triplet->block_col_indices(global_triplet->h_abd_abd_contact_start_id);
    {
        // [kick root-cause fix] The old scatter ASSIGNED (=) each contact
        // triplet's 3x3 into the body block: (a) it WIPED the dyadic-mass seed
        // (the kinetic regularization) from every touched sub-block, and
        // (b) duplicate triplets targeting the same sub-block (one per contact
        // pair — the common case) raced and all but one were lost. Under
        // ground contact a light body's preconditioner block degenerated to a
        // rank-deficient contact-only matrix, so inverse(P) EXPLODED along the
        // soft rotation mode — the very amplifier behind the "resting rigid
        // body flip/kick" pathology (libuipc's abd_diag_preconditioner, which
        // accumulates the full block, is immune; validated 0.075 vs 16.65 m/s
        // on the same pusher-vs-toy scene). Fix: ACCUMULATE onto the mass seed
        // with atomics. (The old racy assignment was already order-
        // nondeterministic, so this does not regress determinism.)
        // STIFF_ABD_PRECOND_LEGACY=1 restores the old behavior for A/B.
        static const bool s_legacy = [] {
            const char* v = getenv("STIFF_ABD_PRECOND_LEGACY");
            return v && v[0] && v[0] != '0';
        }();
        const bool legacy = s_legacy;  // locals are capturable by device lambdas
        ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(global_triplet->abd_abd_contact_num,
                   [P = abd_system_diag_preconditioner.viewer().name("P"),
                    triplet, rows, cols, legacy] __device__(int i) mutable
                   {
                       auto row = rows[i];
                       auto H   = triplet[i];
                       auto col = cols[i];
                       if(row / 4 != col / 4)
                           return;
                       if(legacy)
                       {
                           P(row / 4).block<3, 3>((row % 4) * 3, (col % 4) * 3) = H;
                           if(row != col)
                               P(row / 4).block<3, 3>((col % 4) * 3, (row % 4) * 3) =
                                   H.transpose();
                           return;
                       }
                       const int b  = row / 4;
                       const int r0 = (row % 4) * 3;
                       const int c0 = (col % 4) * 3;
                       for(int a = 0; a < 3; ++a)
                           for(int c = 0; c < 3; ++c)
                           {
                               atomicAdd(&P(b)(r0 + a, c0 + c), H(a, c));
                               if(row != col)
                                   atomicAdd(&P(b)(c0 + c, r0 + a), H(a, c));
                           }
                   });
        int count = sim_data.abd_fem_count_info().abd_body_num;
        ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(count,
                   [P = abd_system_diag_preconditioner.viewer().name("P")] __device__(int i) mutable
                   {
                       auto H = P(i);
                       P(i)   = inverse(H);
                   });
    }
}

// ============================================================================
// Bilateral stitch spring: ABD-side gradient and Hessian
//
// Energy: E = 0.5 * k * |x_fem - J*q - r|^2
// where x_fem = vertex position of FEM point, J*q = ABD vertex position,
// r = rest offset, k = motionRate * rate^2.
//
// ABD gradient: dE/dq = -k * J^T * d   where d = x_fem - J*q - r
// ABD Hessian:  d2E/dq2 = k * J^T * J  (12x12, positive semi-definite)
// ============================================================================
void ABDSystem::_cal_abd_stitch_gradient_and_hessian(ABDSimData& sim_data)
{
    using namespace muda;
    if(m_stitch_count <= 0 || !m_d_stitch_paired_vertex)
        return;

    auto& abd = sim_data.device;
    auto  abd_body_count = sim_data.abd_fem_count_info().abd_body_num;
    auto  abd_point_offset = sim_data.abd_fem_count_info().abd_point_offset;
    auto  body_id_is_fixed = sim_data.body_id_to_boundary_type();

    ParallelFor(256)
        .kernel_name("abd_stitch_gradient_hessian")
        .apply(m_stitch_count,
               [stitch_paired_vertex = m_d_stitch_paired_vertex,
                stitch_rest_offset   = m_d_stitch_rest_offset,
                stitch_abd_body_id   = m_d_stitch_abd_body_id,
                stitch_fem_vertex_id = m_d_stitch_fem_vertex_id,
                all_vertexes         = m_d_all_vertexes,
                Js             = abd.unique_point_id_to_J.cviewer().name("Js"),
                qs             = abd.body_id_to_q.cviewer().name("q"),
                abd_gradient   = abd_gradient.viewer().name("abd_gradient"),
                sys_gradient   = system_gradient.viewer().name("system_gradient"),
                body_hessian   = abd_body_hessian.viewer().name("body_hessian"),
                is_fixed       = body_id_is_fixed.cviewer().name("is_fixed"),
                motionRate     = m_stitch_motion_rate,
                rate           = m_stitch_rate,
                abd_point_offset] __device__(int i) mutable
               {
                   int abd_point_id = stitch_paired_vertex[i];
                   if(abd_point_id < 0)
                       return;  // not a stitch spring

                   int body_id = stitch_abd_body_id[i];
                   if(is_fixed(body_id) == BodyBoundaryType::Fixed)
                       return;

                   double k = motionRate * rate * rate;

                   // Get current positions
                   uint32_t fem_vid = stitch_fem_vertex_id[i];
                   Vector3 x_fem{all_vertexes[fem_vid].x,
                                 all_vertexes[fem_vid].y,
                                 all_vertexes[fem_vid].z};

                   // Compute ABD position: x_abd = J * q
                   int local_abd_point_id = abd_point_id - static_cast<int>(abd_point_offset);
                   gipc::ABDJacobi J = Js(local_abd_point_id);
                   const auto& q = qs(body_id);
                   Vector3 x_abd = J.point_x(q);

                   // Rest offset
                   Vector3 r{stitch_rest_offset[i].x,
                             stitch_rest_offset[i].y,
                             stitch_rest_offset[i].z};

                   // d = x_fem - x_abd - r
                   Vector3 d = x_fem - x_abd - r;

                   // ABD gradient: -k * J^T * d
                   Vector12 G = -(k) * (J.T() * d);
                   eigen::atomic_add(abd_gradient(body_id), G);
                   bin_add12_off(g_abd_sysbin, body_id * 12, G);

                   // ABD Hessian: k * J^T * J (12x12)
                   Matrix3x3 kI = k * Matrix3x3::Identity();
                   Matrix12x12 H = gipc::ABDJacobi::JT_H_J(J.T(), kI, J);
                   bin_add144(g_abd_hessbin, body_id, H);
               });
}

}  // namespace gipc
