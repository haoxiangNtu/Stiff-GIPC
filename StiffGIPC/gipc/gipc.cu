#include <GIPC.cuh>
#include <gipc/gipc.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/json.h>
#include <load_mesh.h>
#include <fstream>
#include "device_common/debug_probes.h"  // g_gipc_log_level
#include <stdexcept>

void GIPC::build_gipc_system(device_TetraData& tet)
{
    std::cout << "* Building GIPC system:" << std::endl;
    gipc::Timer::disable_all();
    
    // set up debug
    muda::Debug::debug_sync_all(false);

    std::cout << "- create ABD system..." << std::endl;
    m_abd_sim_data            = std::make_unique<gipc::ABDSimData>(*this, tet);
    m_abd_system              = std::make_unique<gipc::ABDSystem>();
    m_abd_system->parms.kappa = 1e8;
    m_abd_system->parms.dt    = IPC_dt;
    // FIX: the ABD per-body gravity is PRECOMPUTED in init_system() below using
    // parms.gravity, but SimEngine::finalize() only sets parms.gravity AFTER this
    // build returns. So without this line the precompute uses the default
    // Vector3{0,-9.8,0} (Y-up) instead of the scene's real (Z-up) gravity, and
    // free articulated ABD bodies (e.g. a cartpole pole) never fall. Use the
    // GIPC-level `gravity`, already set from cfg by apply_config_to_ipc() pre-build.
    m_abd_system->parms.gravity[0] = gravity.x;
    m_abd_system->parms.gravity[1] = gravity.y;
    m_abd_system->parms.gravity[2] = gravity.z;

    // Transfer per-body density overrides (stashed before finalize) into the
    // freshly-created ABD system, BEFORE its per-body mass setup runs.
    for(const auto& kv : m_pending_abd_density)
        m_abd_system->set_body_density_override(kv.first, kv.second);

    for(const auto& kv : m_pending_abd_mass)
        m_abd_system->set_body_mass_override(kv.first, kv.second);

    // Transfer per-body inertial overrides (mass / COM / inertia) likewise.
    for(const auto& kv : m_pending_abd_inertia)
    {
        const auto& pi = kv.second;
        Eigen::Vector3d com(pi.com[0], pi.com[1], pi.com[2]);
        Eigen::Matrix3d I;
        I << pi.inertia[0], pi.inertia[1], pi.inertia[2],
             pi.inertia[3], pi.inertia[4], pi.inertia[5],
             pi.inertia[6], pi.inertia[7], pi.inertia[8];
        m_abd_system->set_body_inertia_override(kv.first, pi.mass, com, I);
    }

    std::string config_dir = assets_dir_cfg.empty()
        ? std::string(GIPC_ASSETS_DIR) + "scene/abd_system_config.json"
        : assets_dir_cfg + "scene/abd_system_config.json";

    // [phase4] fail with a nameable error instead of nlohmann's
    // "parse_error.101 ... unexpected end of input" when the file is absent
    // (the exact failure that burned an A800 deployment: bundles must ship
    // Assets/scene/abd_system_config.json).
    std::ifstream config_in{std::string{config_dir}};
    if(!config_in.is_open())
        throw std::runtime_error(
            "[finalize] required config file missing or unreadable: " + std::string{config_dir}
            + " — the deployment bundle must include Assets/scene/abd_system_config.json"
              " (or set assets_dir to a tree that has it)");
    gipc::Json json;
    try
    {
        json = gipc::Json::parse(config_in);
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("[finalize] failed to parse " + std::string{config_dir} + ": " + e.what());
    }
    

    m_abd_system->parms.motor_speed = json["motor_speed"].get<double>();
    m_abd_system->parms.motor_strength = json["motor_strength"].get<double>();
    if(json.contains("joint_strength_ratio"))
        m_abd_system->parms.joint_strength_ratio = json["joint_strength_ratio"].get<double>();
    if(json.contains("revolute_driving_strength_ratio"))
        m_abd_system->parms.revolute_driving_strength_ratio = json["revolute_driving_strength_ratio"].get<double>();

    // [C1] capture the mode snapshot AFTER the python resolver populated the
    // env (Engine.__init__ precedes finalize); print + half-config check.
    m_mode_config = ModeConfig::capture_from_env();
    m_mode_config.warn_if_incoherent();
    m_mode_config.print(g_gipc_log_level);

    std::cout << "- create Global Linear System ..." << std::endl;

    m_global_linear_system = std::make_unique<gipc::GlobalLinearSystem>();

    std::cout << "* Finished building GIPC system." << std::endl;
}

void GIPC::setup_surface_mesh_bodies(tetrahedra_obj& tetMesh)
{
    if(tetMesh.surface_mesh_bodies.empty())
        return;

    m_abd_system->m_surface_mesh_bodies.clear();
    int abd_point_offset = tetMesh.abd_fem_count_info.abd_point_offset;

    for(auto& smb : tetMesh.surface_mesh_bodies)
    {
        gipc::ABDSurfaceMeshBody body;
        body.vertices  = smb.vertices;
        body.triangles = smb.triangles;
        body.orient    = smb.orient;     // copy per-face orient labels (may be empty)
        body.body_id   = smb.body_id;

        // Find vertex range for this body in the ABD unique-point space
        int start = -1, count = 0;
        int abd_point_num = tetMesh.abd_fem_count_info.abd_point_num;
        for(int i = 0; i < abd_point_num; i++)
        {
            if(tetMesh.point_id_to_body_id[abd_point_offset + i] == smb.body_id)
            {
                if(start < 0) start = i;
                count++;
            }
        }
        body.point_start = start;
        body.point_count = count;

        std::cout << "[GIPC] Surface mesh body " << smb.body_id
                  << ": point_start=" << start << ", point_count=" << count << std::endl;

        m_abd_system->m_surface_mesh_bodies.push_back(std::move(body));
    }
}

void GIPC::init_abd_system()
{
    m_abd_sim_data->upload();
    m_abd_system->init_system(*m_abd_sim_data);
}

void GIPC::init_joint_constraints_from_mesh(tetrahedra_obj& tetMesh)
{
    if(tetMesh.joint_constraints.empty() && tetMesh.prismatic_constraints.empty())
    {
        std::cout << "[GIPC] No joint constraints to initialize." << std::endl;
        return;
    }

    if(!tetMesh.joint_constraints.empty())
        m_abd_system->init_joint_constraints(*m_abd_sim_data, tetMesh.joint_constraints);

    if(!tetMesh.joint_angle_controls.empty())
    {
        m_abd_system->init_revolute_driving(
            *m_abd_sim_data, tetMesh.joint_angle_controls, tetMesh.joint_constraints);
    }

    if(!tetMesh.prismatic_constraints.empty())
    {
        m_abd_system->init_prismatic_constraints(*m_abd_sim_data, tetMesh.prismatic_constraints);

        if(!tetMesh.prismatic_drive_controls.empty())
        {
            m_abd_system->init_prismatic_driving(
                *m_abd_sim_data, tetMesh.prismatic_drive_controls, tetMesh.prismatic_constraints);
        }
    }
}

void GIPC::update_joint_angle_targets_from_mesh(tetrahedra_obj& tetMesh, double substep_ratio)
{
    m_drive_substep_mesh = &tetMesh;  // [drive-substep] solve_subIP re-ramps from here

    if(!tetMesh.joint_angle_controls.empty())
        m_abd_system->update_revolute_driving_targets(
            *m_abd_sim_data, tetMesh.joint_angle_controls, substep_ratio);

    if(!tetMesh.prismatic_drive_controls.empty())
        m_abd_system->update_prismatic_driving_targets(
            *m_abd_sim_data, tetMesh.prismatic_drive_controls, substep_ratio);
}

void GIPC::create_LinearSystem(device_TetraData& tet)
{
    std::cout << "    - create ABD Linear Subsystem ..." << std::endl;
    auto& abd = m_global_linear_system->create<gipc::ABDLinearSubsystem>(
        *this, *m_abd_system, *m_abd_sim_data);
    std::cout << "    - create FEM Linear Subsystem ..." << std::endl;
    auto& fem = m_global_linear_system->create<gipc::FEMLinearSubsystem>(*this, tet);


    std::cout << "- create PCG Solver" << std::endl;
    gipc::PCGSolverConfig cfg;
    cfg.global_tol_rate = pcg_threshold;
    auto& pcg           = m_global_linear_system->create<gipc::PCGSolver>(cfg);

    std::cout << "- create Preconditioner" << std::endl;
    
    m_global_linear_system->create<gipc::ABDPreconditioner>(abd, *m_abd_system, *m_abd_sim_data);

    if(pcg_data.P_type == 1)
    {

        m_global_linear_system->create<gipc::MAS_Preconditioner>(
            fem, pcg_data.MP, tet.masses, h_cpNum.raw_alias());  // [3b] MAS keeps this
            // uint32_t* for its lifetime; reads happen post-build only (audit-invisible)
    }
    else
    {
        m_global_linear_system->create<gipc::DiagPreconditioner>();
    }
}
