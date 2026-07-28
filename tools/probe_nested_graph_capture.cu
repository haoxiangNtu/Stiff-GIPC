// Runtime probe for composing an already-instantiated CUDA Graph inside a
// stream capture.  Phase C cannot assume this is legal: if cudaGraphLaunch()
// is rejected while a stream is capturing, PCG/LS graphs must be represented
// as child/conditional bodies instead of launched from the captured body.
#include <cstdio>
#include <cuda_runtime.h>

#include "frame_fsm/conditional_graph.h"

__global__ void add_one(int* value)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
        ++*value;
}

__global__ void continue_until(int* value,
                               int limit,
                               cudaGraphConditionalHandle handle)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
        cudaGraphSetConditional(handle, *value < limit ? 1u : 0u);
}

__global__ void set_if_equal(const int* value,
                             int expected,
                             cudaGraphConditionalHandle handle)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
        cudaGraphSetConditional(
            handle, *value == expected ? 1u : 0u);
}

static bool check(cudaError_t error, const char* what)
{
    std::printf("[nested-probe] %s -> %s\n",
                what,
                cudaGetErrorString(error));
    return error == cudaSuccess;
}

int main()
{
    int* value = nullptr;
    if(!check(cudaMalloc(&value, sizeof(int)), "cudaMalloc"))
        return 1;
    if(!check(cudaMemset(value, 0, sizeof(int)), "cudaMemset"))
        return 1;

    cudaGraph_t child = nullptr;
    cudaGraphExec_t child_exec = nullptr;
    if(!check(cudaStreamBeginCapture(cudaStreamPerThread,
                                     cudaStreamCaptureModeThreadLocal),
              "child begin capture"))
        return 1;
    add_one<<<1, 1, 0, cudaStreamPerThread>>>(value);
    if(!check(cudaStreamEndCapture(cudaStreamPerThread, &child),
              "child end capture"))
        return 1;
    if(!check(cudaGraphInstantiate(&child_exec, child, nullptr, nullptr, 0),
              "child instantiate"))
        return 1;

    cudaGraph_t outer = nullptr;
    if(!check(cudaStreamBeginCapture(cudaStreamPerThread,
                                     cudaStreamCaptureModeThreadLocal),
              "outer begin capture"))
        return 1;
    const cudaError_t nested_launch =
        cudaGraphLaunch(child_exec, cudaStreamPerThread);
    check(nested_launch, "child launch while outer captures");
    if(nested_launch == cudaSuccess)
        add_one<<<1, 1, 0, cudaStreamPerThread>>>(value);
    const cudaError_t end =
        cudaStreamEndCapture(cudaStreamPerThread, &outer);
    check(end, "outer end capture");

    if(nested_launch == cudaSuccess && end == cudaSuccess && outer)
    {
        cudaGraphExec_t outer_exec = nullptr;
        if(!check(cudaGraphInstantiate(
                      &outer_exec, outer, nullptr, nullptr, 0),
                  "outer instantiate"))
            return 1;
        if(!check(cudaGraphLaunch(outer_exec, cudaStreamPerThread),
                  "outer launch")
           || !check(cudaStreamSynchronize(cudaStreamPerThread),
                     "outer synchronize"))
            return 1;
        int host_value = -1;
        if(!check(cudaMemcpy(&host_value,
                             value,
                             sizeof(int),
                             cudaMemcpyDeviceToHost),
                  "result D2H"))
            return 1;
        std::printf("[nested-probe] result=%d (expected 2)\n", host_value);
        cudaGraphExecDestroy(outer_exec);
    }

    if(outer)
        cudaGraphDestroy(outer);
    cudaGraphExecDestroy(child_exec);
    cudaGraphDestroy(child);

    // Prove the composition pattern Phase C needs:
    // capture prefix -> add WHILE -> capture its body -> resume root capture.
    cudaGetLastError();
    if(!check(cudaMemset(value, 0, sizeof(int)), "conditional reset"))
        return 1;
    cudaGraph_t conditional_root = nullptr;
    if(!check(cudaGraphCreate(&conditional_root, 0), "conditional root create"))
        return 1;
    if(!check(cudaStreamBeginCaptureToGraph(
                  cudaStreamPerThread,
                  conditional_root,
                  nullptr,
                  nullptr,
                  0,
                  cudaStreamCaptureModeThreadLocal),
              "prefix begin capture-to-graph"))
        return 1;
    add_one<<<1, 1, 0, cudaStreamPerThread>>>(value);
    cudaGraph_t prefix_result = nullptr;
    if(!check(cudaStreamEndCapture(cudaStreamPerThread, &prefix_result),
              "prefix end capture"))
        return 1;

    size_t prefix_count = 0;
    cudaGraphGetNodes(conditional_root, nullptr, &prefix_count);
    cudaGraphNode_t prefix_tail = nullptr;
    if(prefix_count == 1)
        cudaGraphGetNodes(conditional_root, &prefix_tail, &prefix_count);
    if(prefix_count != 1 || !prefix_tail)
    {
        std::printf("[nested-probe] unexpected prefix node count=%zu\n",
                    prefix_count);
        return 1;
    }

    cudaGraphConditionalHandle while_handle{};
    if(!check(cudaGraphConditionalHandleCreate(
                  &while_handle,
                  conditional_root,
                  1,
                  cudaGraphCondAssignDefault),
              "while handle create"))
        return 1;
    cudaGraphNodeParams while_params{};
    while_params.type               = cudaGraphNodeTypeConditional;
    while_params.conditional.handle = while_handle;
    while_params.conditional.type   = cudaGraphCondTypeWhile;
    while_params.conditional.size   = 1;
    cudaGraphNode_t while_node = nullptr;
    if(!check(cudaGraphAddNode(&while_node,
                               conditional_root,
                               &prefix_tail,
                               1,
                               &while_params),
              "while node add"))
        return 1;

    cudaGraph_t while_body = while_params.conditional.phGraph_out[0];
    if(!check(cudaStreamBeginCaptureToGraph(
                  cudaStreamPerThread,
                  while_body,
                  nullptr,
                  nullptr,
                  0,
                  cudaStreamCaptureModeThreadLocal),
              "while body begin capture"))
        return 1;
    add_one<<<1, 1, 0, cudaStreamPerThread>>>(value);
    continue_until<<<1, 1, 0, cudaStreamPerThread>>>(
        value, 4, while_handle);
    cudaGraph_t body_result = nullptr;
    if(!check(cudaStreamEndCapture(cudaStreamPerThread, &body_result),
              "while body end capture"))
        return 1;

    if(!check(cudaStreamBeginCaptureToGraph(
                  cudaStreamPerThread,
                  conditional_root,
                  &while_node,
                  nullptr,
                  1,
                  cudaStreamCaptureModeThreadLocal),
              "suffix begin capture-to-graph"))
        return 1;
    add_one<<<1, 1, 0, cudaStreamPerThread>>>(value);
    cudaGraph_t suffix_result = nullptr;
    if(!check(cudaStreamEndCapture(cudaStreamPerThread, &suffix_result),
              "suffix end capture"))
        return 1;

    cudaGraphExec_t conditional_exec = nullptr;
    if(!check(cudaGraphInstantiate(
                  &conditional_exec,
                  conditional_root,
                  nullptr,
                  nullptr,
                  0),
              "conditional root instantiate")
       || !check(cudaGraphLaunch(
                     conditional_exec, cudaStreamPerThread),
                 "conditional root launch")
       || !check(cudaStreamSynchronize(cudaStreamPerThread),
                 "conditional root synchronize"))
        return 1;
    int conditional_value = -1;
    if(!check(cudaMemcpy(&conditional_value,
                         value,
                         sizeof(int),
                         cudaMemcpyDeviceToHost),
              "conditional result D2H"))
        return 1;
    std::printf("[nested-probe] conditional result=%d (expected 5)\n",
                conditional_value);
    cudaGraphExecDestroy(conditional_exec);
    cudaGraphDestroy(conditional_root);

    // Exercise the production recorder, including a WHILE nested in another
    // WHILE body.  The inner handle is associated with the owner root, while
    // its node lives in CUDA's outer-body graph.
    int* outer_value = nullptr;
    int* inner_value = nullptr;
    if(!check(cudaMalloc(&outer_value, sizeof(int)), "outer value malloc")
       || !check(cudaMalloc(&inner_value, sizeof(int)), "inner value malloc")
       || !check(cudaMemset(outer_value, 0, sizeof(int)), "outer value reset")
       || !check(cudaMemset(inner_value, 0, sizeof(int)), "inner value reset"))
        return 1;

    cudaGraph_t recorder_root = nullptr;
    if(!check(cudaGraphCreate(&recorder_root, 0), "recorder root create")
       || !check(cudaStreamBeginCaptureToGraph(
                     cudaStreamPerThread,
                     recorder_root,
                     nullptr,
                     nullptr,
                     0,
                     cudaStreamCaptureModeThreadLocal),
                 "recorder prefix capture"))
        return 1;
    add_one<<<1, 1, 0, cudaStreamPerThread>>>(value);
    try
    {
        frame_fsm::ConditionalGraphRecorder recorder(
            recorder_root, cudaStreamPerThread);
        frame_fsm::ConditionalGraphRecorderScope scope(recorder);
        recorder.while_loop(
            1,
            cudaGraphCondAssignDefault,
            [&](cudaGraphConditionalHandle outer_handle)
            {
                add_one<<<1, 1, 0, cudaStreamPerThread>>>(outer_value);
                recorder.while_loop(
                    1,
                    cudaGraphCondAssignDefault,
                    [&](cudaGraphConditionalHandle inner_handle)
                    {
                        add_one<<<1, 1, 0, cudaStreamPerThread>>>(
                            inner_value);
                        continue_until<<<1, 1, 0, cudaStreamPerThread>>>(
                            inner_value, 2, inner_handle);
                    });
                continue_until<<<1, 1, 0, cudaStreamPerThread>>>(
                    outer_value, 4, outer_handle);
            });
        recorder.if_then(
            [&](cudaGraphConditionalHandle if_handle)
            {
                set_if_equal<<<1, 1, 0, cudaStreamPerThread>>>(
                    outer_value, 4, if_handle);
            },
            [&](cudaGraphConditionalHandle)
            {
                add_one<<<1, 1, 0, cudaStreamPerThread>>>(
                    inner_value);
            });
    }
    catch(const std::exception& error)
    {
        std::printf("[nested-probe] recorder exception: %s\n", error.what());
        return 1;
    }
    add_one<<<1, 1, 0, cudaStreamPerThread>>>(value);
    cudaGraph_t recorder_result = nullptr;
    if(!check(cudaStreamEndCapture(
                  cudaStreamPerThread, &recorder_result),
              "recorder suffix end capture"))
        return 1;

    cudaGraphExec_t recorder_exec = nullptr;
    if(!check(cudaGraphInstantiate(
                  &recorder_exec, recorder_root, nullptr, nullptr, 0),
              "recorder root instantiate")
       || !check(cudaGraphLaunch(recorder_exec, cudaStreamPerThread),
                 "recorder root launch")
       || !check(cudaStreamSynchronize(cudaStreamPerThread),
                 "recorder root synchronize"))
        return 1;
    int outer_host = -1;
    int inner_host = -1;
    if(!check(cudaMemcpy(&outer_host,
                         outer_value,
                         sizeof(int),
                         cudaMemcpyDeviceToHost),
              "outer result D2H")
       || !check(cudaMemcpy(&inner_host,
                            inner_value,
                            sizeof(int),
                            cudaMemcpyDeviceToHost),
                 "inner result D2H"))
        return 1;
    std::printf(
        "[nested-probe] recorder nested result outer=%d inner=%d "
        "(expected outer=4, inner=3)\n",
        outer_host,
        inner_host);
    cudaGraphExecDestroy(recorder_exec);
    cudaGraphDestroy(recorder_root);
    cudaFree(outer_value);
    cudaFree(inner_value);
    cudaFree(value);
    return nested_launch != cudaSuccess && end != cudaSuccess
                   && conditional_value == 5 && outer_host == 4
                   && inner_host == 3
               ? 0
               : 2;
}
