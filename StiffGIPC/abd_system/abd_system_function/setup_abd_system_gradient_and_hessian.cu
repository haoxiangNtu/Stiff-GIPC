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
#include <fstream>
#include <vector>
namespace gipc
{

struct DrivingCtrlPacked  { Float target_angle;    Float strength_ratio; };
struct PrisCtrlPacked     { Float target_distance; Float strength_ratio; };

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
    _cal_abd_revolute_driving_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_driving_gradient_and_hessian(sim_data);
    _cal_abd_stitch_gradient_and_hessian(sim_data);
    _cal_abd_system_barrier_gradient(sim_data, vertex_barrier_gradient);
    _setup_abd_system_hessian(sim_data, global_triplets);
}

void ABDSystem::setup_abd_system_gradient_hessian(ABDSimData& sim_data,
                                                  GIPCTripletMatrix& global_triplets,
                                                  muda::CBufferView<Vector3> vertex_barrier_gradient)
{
    _cal_abd_body_gradient_and_hessian(sim_data);
    _cal_abd_joint_gradient_and_hessian(sim_data);
    _cal_abd_revolute_driving_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_gradient_and_hessian(sim_data);
    _cal_abd_prismatic_driving_gradient_and_hessian(sim_data);
    _cal_abd_stitch_gradient_and_hessian(sim_data);
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

                       int offset = drv_output_start + j * 16;
                       write_triplet_cv2<12, 12>(
                           triplet_out, row_out, col_out,
                           index_row, index_col, H_pc, offset);
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

                       int offset = pris_output_start + j * 16;
                       write_triplet_cv2<12, 12>(
                           triplet_out, row_out, col_out,
                           index_row, index_col, H_pc, offset);
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

                       int offset = pris_drv_output_start + j * 16;
                       write_triplet_cv2<12, 12>(
                           triplet_out, row_out, col_out,
                           index_row, index_col, H_pc, offset);
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
            gj.parent_n_bar = Vector3(hj.world_normal.x(), hj.world_normal.y(), hj.world_normal.z());
            gj.child_n_bar  = gj.parent_n_bar;
            gj.parent_b_bar = Vector3(hj.world_bitangent.x(), hj.world_bitangent.y(), hj.world_bitangent.z());
            gj.child_b_bar  = gj.parent_b_bar;
        }
        else
        {
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

        drv.stiffness    = 0.0;  // computed on GPU using body masses
        drv.target_angle = static_cast<Float>(ctrl.target_angle);
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
                sr] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrl_sr(i) * mass_sum;

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
    const std::vector<JointAngleControlInfo>& controls)
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
    }

    muda::DeviceBuffer<DrivingCtrlPacked> d_ctrl(n);
    d_ctrl.view().copy_from(host_ctrl.data());

    constexpr Float kMaxStepPerFrame = 0.1;
    Float sr = parms.revolute_driving_strength_ratio;

    ParallelFor(256)
        .kernel_name("update_revolute_driving_targets_gpu")
        .apply(n,
               [drvs    = m_revolute_driving_data.viewer().name("revolute_driving"),
                q_prev  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                masses  = body_mass.cviewer().name("body_mass"),
                ctrls   = d_ctrl.cviewer().name("ctrls"),
                sr, kMaxStepPerFrame] __device__(int i) mutable
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

                   Float desired_goal = ctrls(i).target_angle;
                   Float diff = desired_goal - theta_prev;
                   diff = (diff >  kMaxStepPerFrame) ?  kMaxStepPerFrame :
                          (diff < -kMaxStepPerFrame) ? -kMaxStepPerFrame : diff;

                   drv.target_angle = theta_prev + diff;

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrls(i).strength_ratio * mass_sum;
               });
}


// ============================================================================
// Revolute Driving Energy
// ============================================================================

Float ABDSystem::cal_abd_revolute_driving_energy(ABDSimData& sim_data)
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
                                                              qs(drv.child_body_id));
               });

    muda::DeviceReduce().Sum(
        m_revolute_driving_energy_per.data(), m_revolute_driving_energy.data(),
        m_num_revolute_driving);

    return m_revolute_driving_energy;
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

                   if(!p_fixed)
                   {
                       eigen::atomic_add(affine_gradient(pid), grad1);
                       sys_gradient.segment<12>(pid * 12).atomic_add(grad1);
                       eigen::atomic_add(body_hessian(pid), H_11);
                   }

                   if(!c_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad2);
                       sys_gradient.segment<12>(cid * 12).atomic_add(grad2);
                       eigen::atomic_add(body_hessian(cid), H_22);
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

Float ABDSystem::cal_abd_prismatic_energy(ABDSimData& sim_data)
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

    return m_prismatic_energy;
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
                       sys_gradient.segment<12>(pid * 12).atomic_add(grad1);
                       eigen::atomic_add(body_hessian(pid), H_pp);
                   }

                   if(!c_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad2);
                       sys_gradient.segment<12>(cid * 12).atomic_add(grad2);
                       eigen::atomic_add(body_hessian(cid), H_qq);
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

        drv.stiffness       = 0.0;
        drv.target_distance = static_cast<Float>(ctrl.target_distance);
    }

    m_prismatic_driving_data.resize(m_num_prismatic_driving);
    m_prismatic_driving_data.view().copy_from(host_data.data());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    Float sr = parms.prismatic_driving_strength_ratio;

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
                sr] __device__(int i) mutable
               {
                   auto& drv = drvs(i);
                   int pid = drv.parent_body_id;
                   int cid = drv.child_body_id;

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrl_sr(i) * mass_sum;

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
    const std::vector<PrismaticDrivingControlInfo>& controls)
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
    }

    muda::DeviceBuffer<PrisCtrlPacked> d_ctrl(n);
    d_ctrl.view().copy_from(host_ctrl.data());

    constexpr Float kMaxStepPerFrame = 0.002;
    Float sr = parms.prismatic_driving_strength_ratio;

    ParallelFor(256)
        .kernel_name("update_prismatic_driving_targets_gpu")
        .apply(n,
               [drvs    = m_prismatic_driving_data.viewer().name("prismatic_driving"),
                q_prev  = abd.body_id_to_q_prev.cviewer().name("q_prev"),
                masses  = body_mass.cviewer().name("body_mass"),
                ctrls   = d_ctrl.cviewer().name("ctrls"),
                sr, kMaxStepPerFrame] __device__(int i) mutable
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

                   drv.target_distance = d_prev + diff;

                   Float mass_sum = masses(pid) + masses(cid);
                   drv.stiffness = sr * ctrls(i).strength_ratio * mass_sum;
               });
}


// ============================================================================
// Prismatic Driving Energy
// ============================================================================

Float ABDSystem::cal_abd_prismatic_driving_energy(ABDSimData& sim_data)
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

    return m_prismatic_driving_energy;
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
                       sys_gradient.segment<12>(pid * 12).atomic_add(grad1);
                       eigen::atomic_add(body_hessian(pid), H_pp);
                   }

                   if(!c_fixed)
                   {
                       eigen::atomic_add(affine_gradient(cid), grad2);
                       sys_gradient.segment<12>(cid * 12).atomic_add(grad2);
                       eigen::atomic_add(body_hessian(cid), H_qq);
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
                   sys_gradient.segment<12>(body_id * 12).atomic_add(G);

                   // ABD Hessian: k * J^T * J (12x12)
                   Matrix3x3 kI = k * Matrix3x3::Identity();
                   Matrix12x12 H = gipc::ABDJacobi::JT_H_J(J.T(), kI, J);
                   eigen::atomic_add(body_hessian(body_id), H);
               });
}

}  // namespace gipc