// Smoke and experimental sweep methods adapted from thomas-hiddenpeak/thor-probe
// at 481668514756b8decda421290c32dbbc130b897f. See LICENSE.thor-probe.
// Timings are whole launches on this GPU; no scheduler count or cache capacity is asserted.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CUDA(expr) do { cudaError_t e=(expr); if (e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#expr,cudaGetErrorString(e)); exit(1); } } while (0)

template<class T, class Acc>
__global__ void uniform_mma(Acc *out, T operand) {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16,Acc> c;
    wmma::fill_fragment(a,operand);
    wmma::fill_fragment(b,operand);
    wmma::fill_fragment(c,Acc(1));
    wmma::mma_sync(c,a,b,c);
    wmma::store_matrix_sync(out,c,16,wmma::mem_row_major);
}

__global__ void fp8_mma(float *out, uint32_t four_ones) {
    float a=1, b=1, c=1, d=1;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%4,%4,%4},{%4,%4},{%0,%1,%2,%3};"
        : "+f"(a),"+f"(b),"+f"(c),"+f"(d):"r"(four_ones));
    int i=4*threadIdx.x;
    out[i]=a; out[i+1]=b; out[i+2]=c; out[i+3]=d;
}

__global__ void warp_ops(unsigned *out) {
    auto lane=threadIdx.x;
    unsigned shuf=__shfl_sync(0xffffffffu,lane+1,31);
    unsigned ballot=__ballot_sync(0xffffffffu,lane<16);
    unsigned any=__any_sync(0xffffffffu,lane<16), all=__all_sync(0xffffffffu,lane<16);
    // Empty async group; exercise instructions only. No TMA descriptor or transfer.
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\nfence.proxy.async.shared::cta;" ::: "memory");
    __threadfence();
    if (!lane) {out[0]=shuf;out[1]=ballot;out[2]=any;out[3]=all;out[4]=1;}
}

__global__ void cluster_exchange(int *out) {
    auto cluster=cooperative_groups::this_cluster();
    __shared__ int shared_value;
    if (!threadIdx.x) shared_value=3+int(cluster.block_rank());
    cluster.sync();
    if (!threadIdx.x) out[cluster.block_rank()]=*cluster.map_shared_rank(&shared_value,1-cluster.block_rank());
    cluster.sync();
}

__global__ void scheduler_work(unsigned *out, unsigned iterations) {
    unsigned v=threadIdx.x+1;
    for (unsigned i=0;i<iterations;i++) {
        asm volatile("mad.lo.u32 %0,%0,1664525,%1;" : "+r"(v):"r"(i));
    }
    out[threadIdx.x]=v;
}

__global__ void bank_work(unsigned *out, unsigned stride, unsigned iterations) {
    __shared__ volatile unsigned data[4096];
    for (int i=threadIdx.x;i<4096;i+=blockDim.x) data[i]=i*7+13;
    __syncthreads();
    unsigned v=0;
    for (unsigned i=0;i<iterations;i++) v+=data[(threadIdx.x*stride+i)&4095];
    out[blockIdx.x*blockDim.x+threadIdx.x]=v;
}

__global__ void pointer_chase(const volatile unsigned *next,unsigned *out,unsigned iters) {
    unsigned i=0;
    for (unsigned n=0;n<iters;n++) i=next[i];
    *out=i;
}

template<class T>
void check_values(const char *name,T *ptr,int count,int expected) {
    std::vector<T> host(count);
    CUDA(cudaMemcpy(host.data(),ptr,count*sizeof(T),cudaMemcpyDeviceToHost));
    int mismatches=0;
    printf("{\"type\":\"numeric\",\"test\":\"%s\",\"expected\":%d,\"values\":[",name,expected);
    for (int i=0;i<count;i++) {
        if (host[i]!=expected) mismatches++;
        printf("%s%.9g",i?",":"",double(host[i]));
    }
    printf("],\"mismatches\":%d}\n",mismatches);
    if (mismatches) exit(2);
}

struct SweepSensors {
    const char *gpc, *temperature;

    static std::string read_integer(const char *path) {
        std::ifstream file(path);
        long long value;
        return file >> value ? std::to_string(value) : "null";
    }

    void sample(int sweep, const char *test, int setting, const char *edge) const {
        auto hz = read_integer(gpc);
        auto mc = read_integer(temperature);
        auto now = std::chrono::system_clock::now().time_since_epoch();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        printf("{\"type\":\"sweep_sensor\",\"sweep\":%d,\"test\":\"%s\",\"setting\":%d,"
               "\"edge\":\"%s\",\"unix_ns\":%lld,\"gpc_hz\":%s,\"gpu_mC\":%s}\n",
               sweep, test, setting, edge, static_cast<long long>(ns), hz.c_str(), mc.c_str());
    }
};

template<class Launch>
void measure(const SweepSensors &sensors, int sweep,const char *test,int setting,int iterations,int block,int grid,int repeats,Launch launch) {
    sensors.sample(sweep,test,setting,"before");
    cudaEvent_t begin,end;
    CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    launch(); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    for (int i=0;i<repeats;i++) {
        CUDA(cudaEventRecord(begin)); launch(); CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
        float ms; CUDA(cudaEventElapsedTime(&ms,begin,end));
        printf("{\"type\":\"timing\",\"sweep\":%d,\"test\":\"%s\",\"setting\":%d,\"repeat\":%d,"
               "\"iterations\":%d,\"block\":%d,\"grid\":%d,\"elapsed_ms\":%.9g}\n",
               sweep,test,setting,i,iterations,block,grid,ms);
    }
    CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
    sensors.sample(sweep,test,setting,"after");
}

int main(int argc,char **argv) {
    if(argc!=5) return 2;
    SweepSensors sensors{argv[3],argv[4]};
    int reps=atoi(argv[1]);
    int sweeps=atoi(argv[2]);
    if(reps<2 || sweeps<1) return 2;
    cudaDeviceProp p; CUDA(cudaGetDeviceProperties(&p,0));
    if (p.major!=11 || p.minor!=0) {fprintf(stderr,"requires CC 11.0\n");return 3;}
    unsigned *out;
    CUDA(cudaMalloc(&out,4096*sizeof(unsigned)));
    auto fp=reinterpret_cast<float*>(out); auto ip=reinterpret_cast<int*>(out);
    CUDA(cudaMemset(out,0xff,256*sizeof(unsigned)));
    uniform_mma<<<1,32>>>(fp,__float2half(1.f)); CUDA(cudaGetLastError()); check_values("fp16_mma",fp,256,17);
    CUDA(cudaMemset(out,0xff,256*sizeof(unsigned)));
    uniform_mma<<<1,32>>>(fp,__float2bfloat16(1.f)); CUDA(cudaGetLastError()); check_values("bf16_mma",fp,256,17);
    CUDA(cudaMemset(out,0xff,256*sizeof(unsigned)));
    uniform_mma<<<1,32>>>(ip,static_cast<signed char>(1)); CUDA(cudaGetLastError()); check_values("int8_mma",ip,256,17);
    CUDA(cudaMemset(out,0xff,128*sizeof(unsigned)));
    fp8_mma<<<1,32>>>(fp,0x38383838u); CUDA(cudaGetLastError()); check_values("fp8_e4m3_mma",fp,128,33);
    warp_ops<<<1,32>>>(out); CUDA(cudaGetLastError());
    unsigned v[5]; CUDA(cudaMemcpy(v,out,sizeof(v),cudaMemcpyDeviceToHost));
    printf("{\"type\":\"warp\",\"shuffle\":%u,\"ballot\":%u,\"any\":%u,\"all\":%u,\"fence_and_empty_group_completed\":%u}\n",v[0],v[1],v[2],v[3],v[4]);
    if(v[0]!=32 || v[1]!=65535 || v[2]!=1 || v[3]!=0 || v[4]!=1) return 2;
    cudaLaunchConfig_t launch{}; launch.gridDim=dim3(2); launch.blockDim=dim3(32);
    cudaLaunchAttribute attr{}; attr.id=cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x=2; attr.val.clusterDim.y=1; attr.val.clusterDim.z=1;
    launch.attrs=&attr; launch.numAttrs=1;
    int cluster_cap;
    CUDA(cudaDeviceGetAttribute(&cluster_cap,cudaDevAttrClusterLaunch,0));
    if (cluster_cap) {
        int max_size;
        CUDA(cudaOccupancyMaxPotentialClusterSize(&max_size,cluster_exchange,&launch));
        CUDA(cudaLaunchKernelEx(&launch,cluster_exchange,ip));
        CUDA(cudaMemcpy(v,out,2*sizeof(unsigned),cudaMemcpyDeviceToHost));
        printf("{\"type\":\"cluster\",\"supported\":true,\"grid\":2,\"cluster_dim\":[2,1,1],\"max_for_this_config\":%d,\"remote_values\":[%u,%u]}\n",max_size,v[0],v[1]);
        if(v[0]!=4 || v[1]!=3) return 2;
    } else puts("{\"type\":\"cluster\",\"supported\":false}");
    for(int sweep=0; sweep<sweeps; ++sweep) {
        for (int w : {1,2,4,8,12,16,20,24,32,40,48}) {
            if(32*w>p.maxThreadsPerBlock) break;
            measure(sensors,sweep,"warp_work",w,2000,32*w,1,reps,[&]{scheduler_work<<<1,32*w>>>(out,2000);});
        }
        for (int stride : {1,2,3,4,5,6,7,8,16,32,64,128})
            measure(sensors,sweep,"bank_word_stride",stride,4096,128,4,reps,[&]{bank_work<<<4,128>>>(out,stride,4096);});
        for (int kb : {4,8,16,32,48,64,96,128,192,256,384,512}) {
            unsigned size=kb*1024/sizeof(unsigned),steps=std::max(65536u,size*2);
            std::vector<unsigned> order(size), next(size);
            std::iota(order.begin(),order.end(),0); std::mt19937 rng(7429); std::shuffle(order.begin(),order.end(),rng);
            for(unsigned i=0;i<size;i++) next[order[i]]=order[(i+1)%size];
            unsigned *chain; CUDA(cudaMalloc(&chain,size*sizeof(unsigned)));
            CUDA(cudaMemcpy(chain,next.data(),size*sizeof(unsigned),cudaMemcpyHostToDevice));
            measure(sensors,sweep,"pointer_chase_KiB",kb,steps,1,1,reps,[&]{pointer_chase<<<1,1>>>(chain,out,steps);});
            unsigned end; CUDA(cudaMemcpy(&end,out,sizeof(end),cudaMemcpyDeviceToHost));
            unsigned expect=0; for(unsigned i=0;i<steps;i++) expect=next[expect];
            printf("{\"type\":\"chain_check\",\"sweep\":%d,\"KiB\":%d,\"expected\":%u,\"observed\":%u}\n",sweep,kb,expect,end);
            if (end!=expect) return 2;
            CUDA(cudaFree(chain));
        }
    }
    cudaFuncAttributes f; CUDA(cudaFuncGetAttributes(&f,scheduler_work));
    for(int threads : {32,64,128,256,512,1024}) for(int smem : {0,4096,8192,16384,32768,40960,49152}) {
        if(threads>f.maxThreadsPerBlock || smem>int(p.sharedMemPerBlock)) continue;
        int active; CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,scheduler_work,threads,smem));
        printf("{\"type\":\"occupancy_api\",\"threads\":%d,\"dynamic_smem_bytes\":%d,\"max_active_blocks_per_sm\":%d,\"kernel_regs_per_thread\":%d,\"kernel_static_shared_bytes\":%zu}\n",
               threads,smem,active,f.numRegs,f.sharedSizeBytes);
    }
    CUDA(cudaFree(out));
    return 0;
}
