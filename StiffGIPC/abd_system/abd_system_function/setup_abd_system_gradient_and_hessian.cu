#include <abd_system/abd_system.h>
#include <muda/launch.h>
#include <muda/ext/eigen/evd.h>
#include <muda/ext/eigen/atomic.h>
#include <muda/ext/eigen/inverse.h>
#include <gipc/utils/cuda_vec_to_eigen.h>
#include <abd_system/abd_energy.h>
#include <abd_system/abd_joint_constraint.h>
#include <gipc/utils/math.h>
#include <gipc/utils/timer.h>
#include "cuda_tools/cuda_tools.h"
#include <fstream>
#include <vector>
namespace gipc
{
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


    int          offset       = vI * 16 + start_output;
    unsigned int index_row[4] = {
        body_id_i * 4, body_id_i * 4 + 1, body_id_i * 4 + 2, body_id_i * 4 + 3};

    unsigned int index_col[4] = {
        body_id_j * 4, body_id_j * 4 + 1, body_id_j * 4 + 2, body_id_j * 4 + 3};

    if(is_fixed[body_id_i] == BodyBoundaryType::Fixed
       || is_fixed[body_id_j] == BodyBoundaryType::Fixed)
    {
        Matrix12x12 zero12 = Matrix12x12::Zero();
        write_triplet_cv2<12, 12>(triplet, rows, cols, index_row, index_col, zero12, offset);
    }
    else
    {
        auto ABD_H = ABDJacobi::JT_H_J(abd_J[i].T(), H, abd_J[j]);
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
    _cal_abd_body_gradient_and_hessian(sim_data);
    _cal_abd_joint_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
    _setup_abd_system_hessian(sim_data, global_triplets);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  GIPCTripletMatrix& global_triplets,
                                                  muda::CBufferView<Vector3> vertex_barrier_gradient)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _cal_abd_joint_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
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
                   system_gradient.segment<12>(body_id * 12).atomic_add(G);
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
                   system_gradient.segment<12>(body_id * 12).atomic_add(G);
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
    int joint_triplet_blocks = m_num_joints * 16;  // 16 block-3x3 per joint cross-body
    int new_triplet_offset =
        global_triplets.fem_fem_contact_num + global_triplets.abd_fem_contact_num * 4
        + (global_triplets.abd_abd_contact_num * 16 + abd_body_count * 10 + joint_triplet_blocks);

    int h_abd_fem_contact_start_id = global_triplets.fem_fem_contact_num;
    int h_abd_abd_contact_start_id =
        h_abd_fem_contact_start_id + global_triplets.abd_fem_contact_num * 4;


    int write_offset = 0;


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
        16 * global_triplets.abd_abd_contact_num + abd_body_count * 10 + joint_triplet_blocks;

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

                       unsigned int index_row[4] = {
                           (unsigned int)(pid * 4),
                           (unsigned int)(pid * 4 + 1),
                           (unsigned int)(pid * 4 + 2),
                           (unsigned int)(pid * 4 + 3)};

                       unsigned int index_col[4] = {
                           (unsigned int)(cid * 4),
                           (unsigned int)(cid * 4 + 1),
                           (unsigned int)(cid * 4 + 2),
                           (unsigned int)(cid * 4 + 3)};

                       int offset = joint_output_start + j * 16;
                       write_triplet_cv2<12, 12>(
                           triplet_out, row_out, col_out,
                           index_row, index_col, H_pc, offset);
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
    auto  kdt2 = parms.joint_stiffness * parms.dt * parms.dt;
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
                kdt2] __device__(int j) mutable
               {
                   auto& joint = joints(j);
                   int   pid   = joint.parent_body_id;
                   int   cid   = joint.child_body_id;

                   auto& q_parent = qs(pid);
                   auto& q_child  = qs(cid);

                   bool parent_fixed = (is_fixed(pid) == BodyBoundaryType::Fixed);
                   bool child_fixed  = (is_fixed(cid) == BodyBoundaryType::Fixed);

                   // Compute gradient
                   Vector12 grad_parent, grad_child;
                   joint_constraint_gradient(joint, q_parent, q_child, kdt2,
                                             grad_parent, grad_child);

                   // Compute Hessian blocks
                   Matrix12x12 H_pp, H_cc, H_pc;
                   joint_constraint_hessian(joint, kdt2, H_pp, H_cc, H_pc);

                   // Add gradient to parent body (skip if fixed)
                   if(!parent_fixed)
                   {
                       eigen::atomic_add(affine_gradient(pid), grad_parent);
                       system_gradient.segment<12>(pid * 12).atomic_add(grad_parent);
                       // Add self-body Hessian
                       eigen::atomic_add(body_hessian(pid), H_pp);
                   }

                   // Add gradient to child body (skip if fixed)
                   if(!child_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad_child);
                       system_gradient.segment<12>(cid * 12).atomic_add(grad_child);
                       // Add self-body Hessian
                       eigen::atomic_add(body_hessian(cid), H_cc);
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
    std::vector<Eigen::Vector3d> world_positions;  // all anchor positions flattened

    for(int j = 0; j < m_num_joints; j++)
    {
        auto& hj     = host_joints[j];
        auto& gj     = host_gpu_data[j];
        gj.parent_body_id = hj.parent_body_id;
        gj.child_body_id  = hj.child_body_id;
        gj.num_points     = hj.num_points;

        // Initialize with world positions (will be converted to material coords below)
        for(int k = 0; k < kMaxJointConstraintPoints; k++)
        {
            if(k < hj.num_points)
            {
                gj.parent_xbar[k] = Vector3(hj.world_anchor[k].x(),
                                              hj.world_anchor[k].y(),
                                              hj.world_anchor[k].z());
                gj.child_xbar[k]  = gj.parent_xbar[k];  // same world position
            }
            else
            {
                gj.parent_xbar[k] = Vector3::Zero();
                gj.child_xbar[k]  = Vector3::Zero();
            }
        }
    }

    // Upload to GPU
    m_joint_data.resize(m_num_joints);
    m_joint_data.view().copy_from(host_gpu_data.data());

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Convert world-space positions to material coordinates using initial q:
    // x_bar = world_pos - q.segment<3>(0)  (since initial A = I)
    using namespace muda;
    ParallelFor(256)
        .kernel_name("convert_joint_world_to_material")
        .apply(m_num_joints,
               [joints = m_joint_data.viewer().name("joint_data"),
                qs     = abd.body_id_to_q.cviewer().name("qs")] __device__(int j) mutable
               {
                   auto& joint = joints(j);
                   int   pid   = joint.parent_body_id;
                   int   cid   = joint.child_body_id;

                   // Get initial body centers from q
                   Vector3 p_parent = qs(pid).segment<3>(0);
                   Vector3 p_child  = qs(cid).segment<3>(0);

                   // Get initial rotation matrices from q (should be close to identity at init)
                   // A = [a1 a2 a3]^T, need A^{-1} to get material coords
                   // But at initialization, A = I, so x_bar = world_pos - p
                   // For robustness, use actual A:
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
                       Vector3 world_pos = joint.parent_xbar[k];  // stored as world pos initially
                       joint.parent_xbar[k] = A_parent_inv * (world_pos - p_parent);
                       joint.child_xbar[k]  = A_child_inv * (world_pos - p_child);
                   }
               });

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::cout << "[ABDSystem] Initialized " << m_num_joints << " joint constraints." << std::endl;
}


void ABDSystem::_cal_abd_system_preconditioner(ABDSimData& sim_data)
{
    using namespace muda;
    auto& abd                        = sim_data.device;
    auto  unique_point_id_to_body_id = sim_data.unique_point_id_to_body_id();
    auto  body_hessian_size = sim_data.abd_fem_count_info().abd_body_num;

    abd_system_diag_preconditioner.resize(body_hessian_size);
    //abd_system_diag_preconditioner.fill(Matrix12x12::Zero());
    auto triplet = global_triplet->block_values(global_triplet->h_abd_abd_contact_start_id);
    auto rows = global_triplet->block_row_indices(global_triplet->h_abd_abd_contact_start_id);
    auto cols = global_triplet->block_col_indices(global_triplet->h_abd_abd_contact_start_id);
    {
        ParallelFor(256)
            .kernel_name(__FUNCTION__)
            .apply(global_triplet->abd_abd_contact_num,
                   [P = abd_system_diag_preconditioner.viewer().name("P"), triplet, rows, cols] __device__(
                       int i) mutable
                   {
                       auto row = rows[i];
                       auto H   = triplet[i];
                       auto col = cols[i];
                       //auto&& [row, col, H] = bcoo(i);
                       if(row / 4 == col / 4)
                       {
                           P(row / 4).block<3, 3>((row % 4) * 3, (col % 4) * 3) = H;
                           if(row != col)
                           {
                               P(row / 4).block<3, 3>((col % 4) * 3, (row % 4) * 3) =
                                   H.transpose();
                           }
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
}  // namespace gipc