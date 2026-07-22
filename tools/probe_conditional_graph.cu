// Minimal runtime probe: does THIS driver actually support CUDA conditional
// graph nodes (IF) and device graph launch?  Version numbers are not proof —
// cross-minor features may return cudaErrorCallRequiresNewerDriver, so run
// this on the EXACT production image (A800 / R535) before choosing a backend.
//
// Build:  nvcc -arch=sm_80 -o probe_conditional_graph probe_conditional_graph.cu
// Run:    ./probe_conditional_graph
// Exit 0 = conditional supported; 2 = only device tail-launch; 3 = neither.
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call, what)                                                      \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if(_e != cudaSuccess)                                                  \
        {                                                                      \
            printf("[probe] %s -> %s (%d)\n", what, cudaGetErrorString(_e), _e); \
        }                                                                      \
        else                                                                   \
            printf("[probe] %s -> OK\n", what);                                \
    } while(0)

__global__ void noop() {}

static bool probe_device_tail_launch()
{
    cudaGraph_t g = nullptr;
    cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    noop<<<1, 1, 0, cudaStreamPerThread>>>();
    cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &g);
    if(e != cudaSuccess || !g) { cudaGetLastError(); return false; }
    cudaGraphExec_t x = nullptr;
    e = cudaGraphInstantiateWithFlags(&x, g, cudaGraphInstantiateFlagDeviceLaunch);
    printf("[probe] instantiate(DeviceLaunch) -> %s\n", cudaGetErrorString(e));
    bool ok = (e == cudaSuccess && x);
    if(ok)
    {
        e = cudaGraphUpload(x, cudaStreamPerThread);
        printf("[probe] graphUpload -> %s\n", cudaGetErrorString(e));
        ok = (e == cudaSuccess);
    }
    if(x) cudaGraphExecDestroy(x);
    cudaGraphDestroy(g);
    cudaGetLastError();
    return ok;
}

static bool probe_conditional_if()
{
#if CUDART_VERSION >= 12030
    cudaGraph_t g = nullptr;
    cudaError_t e = cudaGraphCreate(&g, 0);
    if(e != cudaSuccess) { cudaGetLastError(); return false; }

    cudaGraphConditionalHandle h;
    e = cudaGraphConditionalHandleCreate(&h, g, 1, cudaGraphCondAssignDefault);
    printf("[probe] conditionalHandleCreate -> %s\n", cudaGetErrorString(e));
    bool ok = (e == cudaSuccess);
    if(ok)
    {
        cudaGraphNodeParams p{};
        p.type                = cudaGraphNodeTypeConditional;
        p.conditional.handle  = h;
        p.conditional.type    = cudaGraphCondTypeIf;
        p.conditional.size    = 1;
        cudaGraphNode_t n = nullptr;
        e = cudaGraphAddNode(&n, g, nullptr, 0, &p);
        printf("[probe] addNode(conditional IF) -> %s\n", cudaGetErrorString(e));
        ok = (e == cudaSuccess);
        if(ok)
        {
            cudaGraphExec_t x = nullptr;
            e = cudaGraphInstantiate(&x, g, 0);
            printf("[probe] instantiate(conditional) -> %s\n", cudaGetErrorString(e));
            ok = (e == cudaSuccess && x);
            if(ok)
            {
                e = cudaGraphLaunch(x, cudaStreamPerThread);
                cudaStreamSynchronize(cudaStreamPerThread);
                printf("[probe] launch(conditional) -> %s\n", cudaGetErrorString(e));
                ok = (e == cudaSuccess);
            }
            if(x) cudaGraphExecDestroy(x);
        }
    }
    cudaGraphDestroy(g);
    cudaGetLastError();
    return ok;
#else
    printf("[probe] toolkit < 12.3: conditional API not compiled in\n");
    return false;
#endif
}

int main()
{
    int drv = 0, rt = 0, dev = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("[probe] device=%s cc=%d.%d driver=%d runtime=%d\n",
           prop.name, prop.major, prop.minor, drv, rt);

    const bool tail = probe_device_tail_launch();
    const bool cond = probe_conditional_if();
    printf("[probe] RESULT: device_tail_launch=%s conditional_if=%s\n",
           tail ? "YES" : "NO", cond ? "YES" : "NO");
    if(cond) return 0;
    if(tail) return 2;
    return 3;
}
