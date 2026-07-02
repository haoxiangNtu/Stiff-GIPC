#pragma once
#include <gipc/type_define.h>
#include <muda/buffer/device_buffer.h>
#include "linear_system/linear_system/global_matrix.h"
namespace gipc
{
class Converter
{
  public:
    // Triplet -> BCOO
    void convert(GIPCTripletMatrix& global_triplets,
                 const int&         start,
                 const int&         length,
                 const int&         out_start_id);


    void _radix_sort_indices_and_blocks(GIPCTripletMatrix& global_triplets,
                                        const int&         start,
                                        const int&         length,
                                        const int&         out_start_id);


    void _make_unique_indices(GIPCTripletMatrix& global_triplets,
                              const int&         start,
                              const int&         length,
                              const int&         out_start_id);


    void _make_unique_block_warp_reduction(GIPCTripletMatrix& global_triplets,
                                           const int&         start,
                                           const int&         length,
                                           const int&         out_start_id);


    void ge2sym(GIPCTripletMatrix& global_triplets);

    ~Converter();

  private:
    // [multi-env determinism 4.3 #4] binned accumulator for the duplicate-block merge
    // (9 doubles per unique block, BINNED_K bins each): makes the merge order-independent
    // ⇒ deterministic matrix values (FastSegmentalReduce summed in emission order). Grown lazily.
    double* m_mergebin     = nullptr;
    size_t  m_mergebin_cap = 0;
};
}  // namespace gipc
