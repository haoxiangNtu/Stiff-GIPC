//
// MASPreconditioner.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "device_fem_data.cuh"
#include "eigen_data.h"
#include <muda/ext/linear_system/bcoo_matrix_view.h>
#include "linear_system/linear_system/global_matrix.h"

class MASPreconditioner
{

    int totalNodes = 0;
    int totalMapNodes = 0;
    int levelnum = 0;
    // [per-env MAS] capacity of the cluster-space scratch arrays
    // (d_nextConnectMask / d_nextPrefix / d_nextPrefixSum / d_goingNext-per-level).
    // Per-env padding can push a level's cluster count ABOVE vertNum on small
    // scenes (each env padded to a BANKSIZE multiple), so vertNum-sized arrays
    // overflow (caught by compute-sanitizer). Sized as
    // max(vertNum, partMapSize) + (m_numEnvs+1)*BANKSIZE.
    int m_clusterCap = 0;
    // [audit lens-A fix] TRUE allocation size of the OUTPUT-layer buffers
    // (d_multiLevelR/Z, d_mRbin/mZbin, d_matbin, d_precondMatMas,
    // d_inverseMatMas/d_MatMas). These were sized ONCE at init from the
    // zero-contact cluster count (*1.05) and the size was never stored — the
    // per-frame totalNumberClusters (real contact connectivity) was written
    // into them with the ONLY assertion checking m_clusterCap*levelnum, the
    // capacity of a DIFFERENT (much larger, per-level-padded) scratch group.
    int m_outputClusterCap = 0;
    void ensureOutputClusterCapacity(int need);
  public:
    int m_numEnvs = 1;   // [per-env MAS] #body-groups (envs); set at setup from tetMesh.body_groups
  private:
    int* d_envBase  = nullptr;   // [per-env MAS] device scratch: per-env aligned base offsets
    int* d_envStart = nullptr;   //   per-env pre-mutation scan start (avoids RAW hazard in _apply)
    int* d_padTot   = nullptr;   //   per-env-padded cluster total (written to d_levelSize on device)
    int2* d_segwpe  = nullptr;   // [B3 s8] {segN, wpe} decided in-kernel per level
    int collision_node_Offset = 0;
    int totalNumberClusters = 0;
    // Allocation-backed launch bound. When device-count mode is active the
    // exact per-frame hierarchy extent remains in d_levelSize[levelnum].
    int m_allocClusterTotal = 0;
    //int bankSize;
    int2  h_clevelSize{};
    int4* _collisonPairs = nullptr;

    int2*               d_levelSize = nullptr;
    int*                d_coarseSpaceTables = nullptr;
    int*                d_prefixOriginal = nullptr;
    int*                d_prefixSumOriginal = nullptr;
    int*                d_goingNext = nullptr;
    int*                d_denseLevel = nullptr;
    __GEIGEN__::itable* d_coarseTable = nullptr;
    unsigned int*       d_fineConnectMask = nullptr;
    unsigned int*       d_nextConnectMask = nullptr;
    unsigned int*       d_nextPrefix = nullptr;
    unsigned int*       d_nextPrefixSum = nullptr;
    // CUB scans run inside the full-frame capture.  Their temporary storage
    // must therefore be sized and allocated once at the scene boundary.
    void*  d_scanTemp = nullptr;
    size_t m_scanTempBytes = 0;


    __GEIGEN__::MasMatrixT*    d_MatMas = nullptr;
    __GEIGEN__::MasMatrixSymT* d_inverseMatMas = nullptr;
    __GEIGEN__::MasMatrixSymf* d_precondMatMas = nullptr;
    Eigen::Vector3f*           d_multiLevelR = nullptr;
    Precision_T3*              d_multiLevelZ = nullptr;
    double* d_mRbin  = nullptr;  // [4.3] binned d_multiLevelR coarse accumulation
    double* d_mZbin  = nullptr;  // [4.3] binned d_multiLevelZ Schwarz accumulation
    double* d_matbin = nullptr;  // [4.3] binned d_inverseMatMas coarse aggregation

  public:
    int           neighborListSize = 0;
    unsigned int* d_neighborList = nullptr;
    unsigned int* d_neighborStart = nullptr;
    unsigned int* d_neighborStartTemp = nullptr;
    unsigned int* d_neighborNum = nullptr;
    unsigned int* d_neighborListInit = nullptr;
    unsigned int* d_neighborNumInit = nullptr;
    int*          d_partId_map_real = nullptr;
    int*          d_real_map_partId = nullptr;

  public:
    void initPreconditioner_Neighbor(int   vertNum,
                                     int   mCollision_node_offset,
                                     int   totalNeighborNum,
                                     int4* m_collisonPairs,
                                     int   partMapSize);
    void computeNumLevels(int vertNum);  // called in initPreconditioner_Neighbor

    void initPreconditioner_Matrix();


    int  ReorderRealtime(int cpNum);
    bool deviceExtentActive() const;
    int  exactClusterCountBlocking() const;
    void BuildConnectMaskL0();           // called in ReorderRealtime
    void PreparePrefixSumL0();           // called in ReorderRealtime
    void BuildLevel1();                  // called in ReorderRealtime
    void BuildConnectMaskLx(int level);  // called in ReorderRealtime
    void NextLevelCluster(int level);    // called in ReorderRealtime
    void PrefixSumLx(int level);         // called in ReorderRealtime
    void ComputeNextLevel(int level);    // called in ReorderRealtime
    void AggregationKernel();            // called in ReorderRealtime
    void BuildCollisionConnection(unsigned int* connectionMsk,
                                  int*          coarseTableSpace,
                                  int           level,
                                  int cpNum);  // called in ReorderRealtime

    void setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values,
                                int*             row_ids,
                                int*             col_ids,
                                uint32_t*        indices,
                                int              offset,
                                int              triplet_num,
                                const int*       d_triplet_num,
                                int              cpNum);
    void PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                             int*             row_ids,
                             int*             col_ids,
                             uint32_t*        indices,
                             int              offset,
                             int              triplet_number,
                             const int*       d_triplet_number);

    void preconditioning(const double3* R, double3* Z);
    void BuildMultiLevelR(const double3* R);  // called in preconditioning
    void SchwarzLocalXSym();                  // called in preconditioning
    void SchwarzLocalXSym_block3();                  // called in preconditioning
    void SchwarzLocalXSym_sym();           // called in preconditioning
    void CollectFinalZ(double3* Z);           // called in preconditioning

    void FreeMAS();
};
