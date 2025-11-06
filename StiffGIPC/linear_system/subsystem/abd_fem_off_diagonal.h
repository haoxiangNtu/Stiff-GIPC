#pragma once
#include <linear_system/linear_system/linear_subsystem.h>
#include <linear_system/subsystem/abd_linear_subsystem.h>
#include <linear_system/subsystem/fem_linear_subsystem.h>

class GIPC;

namespace gipc
{
class ABDSystem;
class ABDFEMOffDiagonal : public OffDiagonalSubsystem
{
    using Base = OffDiagonalSubsystem;
  public:
    ABDFEMOffDiagonal(GIPC&                gipc,
                      device_TetraData&    tetra_data,
                      ABDLinearSubsystem&  abd,
                      FEMLinearSubsystem&  fem);

  public:
    virtual void report_subsystem_info() override;
    virtual void assemble(TripletMatrixView upper, TripletMatrixView lower) override;
    muda::CBufferView<int> boundary_type() const;
  private:
    GIPC&                m_gipc;
    device_TetraData&    m_tetra_data;
    gipc::ABDSimData&    m_abd_sim_data;
};
}  // namespace gipc
