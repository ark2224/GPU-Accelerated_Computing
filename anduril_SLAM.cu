// =============================================================
// Anduril SLAM CUDA Pack — illustrative .cu files for Steps 1–5
// -------------------------------------------------------------
// Notes:
// • These are self-contained CUDA/C++ examples meant to accompany the
// written outline. They focus on clarity and systems thinking
// (profiling, robustness, latency control, streaming, and integration).
// • External deps are avoided (no OpenCV/Eigen). This keeps them portable
// and easy to compile with: nvcc -O3 -std=c++17 file.cu -o app
// • Real-world integration would swap the toy math with your shipping
// geometry libs, use real camera/IMU models, and wire into ROS2/LCM.
// • CUDA version assumed: 11.x+.
// =============================================================

// =============================================================
// ----- FILE: step1_profiler.cu
// Purpose: Latency & utilization measurement with NVTX ranges, CUDA events,
// percentile (p50/p95) computation, and reproducible runs.
// =============================================================


#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>


#ifdef __has_include
# if __has_include(<nvToolsExt.h>)
# include <nvToolsExt.h>
# define HAS_NVTX 1
# else
# define HAS_NVTX 0
# endif
#else
# define HAS_NVTX 0
#endif


#define CUDA_CHECK(x) do { \
cudaError_t err_ = (x); \
if (err_ != cudaSuccess) { \
fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
std::abort(); \
} \
} while(0)


__global__ void noop_kernel(float *out, const float *in, int n) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) out[i] = in[i] * 1.000001f; // prevent optimization-away
}


struct GpuTimer {
cudaEvent_t start_, stop_;
GpuTimer() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
~GpuTimer(){ cudaEventDestroy(start_); cudaEventDestroy(stop_); }
void start(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(start_, s)); }
float stop(cudaStream_t s=0){ CUDA_CHECK(cudaEventRecord(stop_, s)); CUDA_CHECK(cudaEventSynchronize(stop_)); float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start_,stop_)); return ms; }
};


struct HostTimer {
std::chrono::high_resolution_clock::time_point t0;
void tic(){ t0 = std::chrono::high_resolution_clock::now(); }
double toc_ms() const {
auto t1 = std::chrono::high_resolution_clock::now();
return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
};


static double percentile(std::vector<double> v, double p){
if (v.empty()) return 0.0;
size_t k = (size_t)((p/100.0)*(v.size()-1));
std::nth_element(v.begin(), v.begin()+k, v.end());
return v[k];
}

extern "C" void step1_profile_example(int N=1<<20, int warmup=5, int iters=50){
const int BS=256; int GS=(N+BS-1)/BS;
float *d_in=nullptr,*d_out=nullptr; CUDA_CHECK(cudaMalloc(&d_in,N*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_out,N*sizeof(float)));
std::vector<float> h_in(N, 1.0f); CUDA_CHECK(cudaMemcpy(d_in,h_in.data(),N*sizeof(float),cudaMemcpyHostToDevice));
std::vector<double> frame_ms; frame_ms.reserve(iters);


// Reproducibility hint: fixed seeds (not shown) & fixed clocks (if possible)


// Warmup
for(int i=0;i<warmup;++i){
#if HAS_NVTX
nvtxRangePushA("noop_warmup");
#endif
noop_kernel<<<GS,BS>>>(d_out,d_in,N);
CUDA_CHECK(cudaDeviceSynchronize());
#if HAS_NVTX
nvtxRangePop();
#endif
}


// Measured runs
for(int i=0;i<iters;++i){
GpuTimer gt; gt.start();
#if HAS_NVTX
nvtxRangePushA("noop_iter");
#endif
noop_kernel<<<GS,BS>>>(d_out,d_in,N);
float gpu_ms = gt.stop();
#if HAS_NVTX
nvtxRangePop();
#endif
frame_ms.push_back(gpu_ms);
}


double p50 = percentile(frame_ms, 50.0);
double p95 = percentile(frame_ms, 95.0);
printf("[Step1] noop_kernel N=%d => p50=%.3f ms, p95=%.3f ms (iters=%d)\n", N, p50, p95, iters);


CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
}



// =============================================================
// ----- FILE: step2_frontend.cu
// Purpose: Robust CV front-end on GPU — FAST corners + BRIEF descriptors
// with semantic down-weighting & IMU-prior inlier filter.
// =============================================================


// Minimal FAST corner score (Bresenham circle of 16, threshold t)
__device__ __forceinline__ int fast_score(const uint8_t* img, int w, int h, int x, int y, int t){
if (x<3||y<3||x>=w-3||y>=h-3) return 0;
const int off = y*w + x; int c = img[off];
// circle offsets (12 o'clock start, clockwise)
const int dx[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
const int dy[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};
int brighter=0, darker=0;
// Short early exits: need >= 9 contiguous; here we just count total as score proxy
#pragma unroll
for(int i=0;i<16;++i){ int v = img[(y+dy[i])*w + (x+dx[i])]; brighter += (v >= c + t); darker += (v <= c - t); }
return max(brighter, darker); // proxy for FAST score
}

// Kernel: compute FAST scores with semantic mask down-weighting
// mask: 0=static, 1=dynamic (down-weight), 255=unknown (treat static)
__global__ void fast_score_kernel(const uint8_t* __restrict__ img, const uint8_t* __restrict__ mask,
int w, int h, int stride, int t, float dyn_w,
float* __restrict__ scores){
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (x>=w || y>=h) return;
int s = fast_score(img, w, h, x, y, t);
uint8_t m = mask ? mask[y*stride + x] : 0;
float wgt = (m==1) ? dyn_w : 1.0f;
scores[y*w + x] = wgt * (float)s;
}


// Simple nonmax suppression in 3x3 window (shared-memory tile)
__global__ void nms3x3_kernel(const float* __restrict__ scores, int w, int h, float thresh,
uint2* __restrict__ keypoints, int max_kp, int* __restrict__ out_count){
extern __shared__ float tile[]; // (BLOCK_Y+2)*(BLOCK_X+2)
constexpr int BX=16, BY=16;
int tx=threadIdx.x, ty=threadIdx.y;
int x = blockIdx.x*BX + tx; int y = blockIdx.y*BY + ty;


// Load with 1-pixel halo
int lx = tx+1, ly = ty+1; int W = BX+2; int H = BY+2;
if (x<w && y<h) tile[ly*W + lx] = scores[y*w + x];
if (tx==0 && x>0 && y<h) tile[ly*W + 0] = scores[y*w + (x-1)];
if (tx==BX-1 && x+1<w && y<h) tile[ly*W + (lx+1)] = scores[y*w + (x+1)];
if (ty==0 && y>0 && x<w) tile[0*W + lx] = scores[(y-1)*w + x];
if (ty==BY-1 && y+1<h && x<w) tile[(ly+1)*W + lx] = scores[(y+1)*w + x];
// corners of halo
if (tx==0 && ty==0 && x>0 && y>0) tile[0*W+0] = scores[(y-1)*w + (x-1)];
if (tx==BX-1 && ty==0 && x+1<w && y>0) tile[0*W+(lx+1)] = scores[(y-1)*w + (x+1)];
if (tx==BX-1 && ty==BY-1 && x+1<w && y+1<h) tile[(ly+1)*W+(lx+1)] = scores[(y+1)*w + (x+1)];
__syncthreads();


if (x>=w || y>=h) return;
float c = tile[ly*W + lx];
if (c < thresh) return;
bool is_max = true;
#pragma unroll
for(int dy=-1; dy<=1; ++dy){
#pragma unroll
for(int dx=-1; dx<=1; ++dx){
if (dx==0 && dy==0) continue;
is_max &= (c >= tile[(ly+dy)*W + (lx+dx)]);
}
}
if (is_max){
int idx = atomicAdd(out_count, 1);
if (idx < max_kp) keypoints[idx] = make_uint2(x,y);
}
}


// BRIEF descriptor (256-bit) with pre-defined pairs
__constant__ int2 c_brief_pairs[256];


__global__ void brief_kernel(const uint8_t* __restrict__ img, int w, int h, const uint2* __restrict__ kps,
int num_kp, uint32_t* __restrict__ desc){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= num_kp) return;
int x = kps[i].x, y = kps[i].y;
uint32_t d[8] = {0};
#pragma unroll
for(int b=0;b<256;++b){
int2 p = c_brief_pairs[b];
int x1 = min(max(x + (p.x>>8), 0), w-1); // high byte = dx1, low byte = dy1 (packing trick if desired)
int y1 = min(max(y + (p.x & 0xFF) - 128, 0), h-1);
int x2 = min(max(x + (p.y>>8), 0), w-1);
int y2 = min(max(y + (p.y & 0xFF) - 128, 0), h-1);
int bit = (img[y1*w+x1] < img[y2*w+x2]);
d[b>>5] |= (bit << (b & 31));
}
// write 8 words (256 bits)
#pragma unroll
for(int j=0;j<8;++j) desc[i*8 + j] = d[j];
}

// Hamming distance for 256-bit BRIEF
__device__ __forceinline__ int hamming256(const uint32_t* a, const uint32_t* b){
int d=0;
#pragma unroll
for(int i=0;i<8;++i) d += __popc(a[i]^b[i]);
return d;
}


// Blocked matcher: for each kpA[i], find best match in kpB within window
__global__ void match_brief_kernel(const uint32_t* __restrict__ descA, const uint2* __restrict__ kpA, int nA,
const uint32_t* __restrict__ descB, const uint2* __restrict__ kpB, int nB,
int max_dist, int* __restrict__ match_idx, int* __restrict__ match_dist){
int i = blockIdx.x * blockDim.x + threadIdx.x; if (i>=nA) return;
const uint32_t* dA = &descA[i*8];
int best_j=-1, best_d=1e9;
for(int j=0;j<nB;++j){
int d = hamming256(dA, &descB[j*8]);
if (d < best_d){ best_d=d; best_j=j; }
}
if (best_d <= max_dist){ match_idx[i]=best_j; match_dist[i]=best_d; }
else { match_idx[i]=-1; match_dist[i]=INT_MAX; }
}


// IMU-prior inlier filter: given a provisional Essential matrix E (from IMU delta-R,t dir),
// compute x2^T E x1 residuals and Huber weight.
__global__ void epipolar_inlier_kernel(const float* __restrict__ E9, // row-major 3x3
const float2* __restrict__ pts1, const float2* __restrict__ pts2,
int n, float huber, uint8_t* __restrict__ inlier){
float E00=E9[0],E01=E9[1],E02=E9[2];
float E10=E9[3],E11=E9[4],E12=E9[5];
float E20=E9[6],E21=E9[7],E22=E9[8];
int i = blockIdx.x * blockDim.x + threadIdx.x; if (i>=n) return;
float x1=pts1[i].x, y1=pts1[i].y, x2=pts2[i].x, y2=pts2[i].y;
float l0 = E00*x1 + E01*y1 + E02;
float l1 = E10*x1 + E11*y1 + E12;
float l2 = E20*x1 + E21*y1 + E22;
float r = x2*l0 + y2*l1 + l2; // epipolar residual
float a = fabsf(r);
inlier[i] = (a < huber) ? 1 : 0;
}

// Host wrapper for Step 2 (toy wiring)
extern "C" void step2_frontend_example(int W, int H, int max_kp){
const dim3 BS2(16,16), GS2((W+15)/16,(H+15)/16);
// Allocate dummy image + mask
std::vector<uint8_t> h_img(W*H, 127), h_msk(W*H, 0);
uint8_t *d_img=nullptr,*d_msk=nullptr; CUDA_CHECK(cudaMalloc(&d_img,W*H)); CUDA_CHECK(cudaMalloc(&d_msk,W*H));
CUDA_CHECK(cudaMemcpy(d_img,h_img.data(),W*H,cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(d_msk,h_msk.data(),W*H,cudaMemcpyHostToDevice));


float *d_scores=nullptr; CUDA_CHECK(cudaMalloc(&d_scores,W*H*sizeof(float)));
fast_score_kernel<<<GS2,BS2>>>(d_img,d_msk,W,H,W,20,0.3f,d_scores);


uint2 *d_kp=nullptr; int *d_cnt=nullptr; CUDA_CHECK(cudaMalloc(&d_kp,max_kp*sizeof(uint2))); CUDA_CHECK(cudaMalloc(&d_cnt,sizeof(int))); CUDA_CHECK(cudaMemset(d_cnt,0,sizeof(int)));
size_t shmem = (16+2)*(16+2)*sizeof(float);
nms3x3_kernel<<<GS2,BS2,shmem>>>(d_scores,W,H,5.0f,d_kp,max_kp,d_cnt);


int h_cnt=0; CUDA_CHECK(cudaMemcpy(&h_cnt,d_cnt,sizeof(int),cudaMemcpyDeviceToHost)); h_cnt = std::min(h_cnt, max_kp);
printf("[Step2] detected %d keypoints (capped at %d)\n", h_cnt, max_kp);


// Clean up (descriptors/matching omitted in this short example)
CUDA_CHECK(cudaFree(d_scores)); CUDA_CHECK(cudaFree(d_kp)); CUDA_CHECK(cudaFree(d_cnt));
CUDA_CHECK(cudaFree(d_img)); CUDA_CHECK(cudaFree(d_msk));
}



// =============================================================
// ----- FILE: step3_backend_pcg.cu
// Purpose: Resilient back-end — PCG solver on GPU for sparse normal eqns
// A*x = b with CSR storage. Demonstrates incremental optimization
// loop infrastructure.
// =============================================================


struct CsrMatrix {
int n; // dimension
int nnz; // non-zeros
int *rowPtr; // size n+1
int *colInd; // size nnz
float *vals; // size nnz
};


__global__ void spmv_csr_kernel(const int n, const int* __restrict__ rowPtr, const int* __restrict__ colInd,
const float* __restrict__ vals, const float* __restrict__ x, float* __restrict__ y){
int r = blockIdx.x * blockDim.x + threadIdx.x; if (r>=n) return;
float sum=0.f; int start=rowPtr[r], end=rowPtr[r+1];
for(int i=start;i<end;++i) sum += vals[i]*x[colInd[i]];
y[r]=sum;
}


__global__ void vec_axpy(int n, float a, const float* __restrict__ x, float* __restrict__ y){
int i=blockIdx.x*blockDim.x+threadIdx.x; if (i<n) y[i] += a*x[i];
}
__global__ void vec_scale(int n, float a, float* __restrict__ x){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=a; }
__global__ void vec_copy(int n, const float* __restrict__ x, float* __restrict__ y){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=x[i]; }


__global__ void vec_pointwise_div(int n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=a[i]/(b[i]+1e-12f);
}


__global__ void vec_mul(int n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){
int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=a[i]*b[i];
}


__global__ void vec_set(int n, float v, float* __restrict__ x){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]=v; }


// Dot product via block reduction
__global__ void dot_kernel(int n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out){
extern __shared__ float s[]; int i=blockIdx.x*blockDim.x+threadIdx.x; float v=0.f;
if (i<n) v = a[i]*b[i]; s[threadIdx.x]=v; __syncthreads();
// reduce
for(int sft=blockDim.x/2; sft>0; sft>>=1){ if(threadIdx.x<sft) s[threadIdx.x]+=s[threadIdx.x+sft]; __syncthreads(); }
if(threadIdx.x==0) out[blockIdx.x]=s[0];
}


static float device_dot(int n, const float* a, const float* b){
const int BS=256; int GS=(n+BS-1)/BS; size_t sh=BS*sizeof(float);
float *d_partial=nullptr; CUDA_CHECK(cudaMalloc(&d_partial, GS*sizeof(float)));
dot_kernel<<<GS,BS,sh>>>(n,a,b,d_partial);
std::vector<float> h_partial(GS); CUDA_CHECK(cudaMemcpy(h_partial.data(),d_partial,GS*sizeof(float),cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaFree(d_partial));
double sum=0.0; for(float v: h_partial) sum+=v; return (float)sum;
}


// Jacobi preconditioner (diag of A)
__global__ void csr_diag_kernel(int n, const int* __restrict__ rowPtr, const int* __restrict__ colInd, const float* __restrict__ vals, float* __restrict__ diag){
int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n) return; float d=0.f; for(int i=rowPtr[r]; i<rowPtr[r+1]; ++i) if(colInd[i]==r){ d=vals[i]; break; } diag[r]=d; }


extern "C" bool step3_pcg_solve(const CsrMatrix& A, const float* d_b, float* d_x, int iters=100, float tol=1e-4f){
const int n=A.n; const int BS=256; int GS=(n+BS-1)/BS;
float *d_r=nullptr,*d_p=nullptr,*d_Ap=nullptr,*d_z=nullptr,*d_M=nullptr;
CUDA_CHECK(cudaMalloc(&d_r,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_p,n*sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_Ap,n*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_z,n*sizeof(float)));
CUDA_CHECK(cudaMalloc(&d_M,n*sizeof(float)));


// M = diag(A)
csr_diag_kernel<<<GS,BS>>>(n,A.rowPtr,A.colInd,A.vals,d_M);


// r = b - A*x
spmv_csr_kernel<<<GS,BS>>>(n,A.rowPtr,A.colInd,A.vals,d_x,d_Ap);
CUDA_CHECK(cudaDeviceSynchronize());
// r = b - Ap
vec_copy<<<GS,BS>>>(n,d_b,d_r); vec_axpy<<<GS,BS>>>(n,-1.0f,d_Ap,d_r);


// z = M^{-1} r (Jacobi)
vec_pointwise_div<<<GS,BS>>>(n,d_r,d_M,d_z);
vec_copy<<<GS,BS>>>(n,d_z,d_p);


float rz_old = device_dot(n,d_r,d_z);
float b_norm = sqrtf(device_dot(n,d_b,d_b)+1e-12f);


bool ok=false;
for(int k=0;k<iters;++k){
spmv_csr_kernel<<<GS,BS>>>(n,A.rowPtr,A.colInd,A.vals,d_p,d_Ap);
float pAp = device_dot(n,d_p,d_Ap) + 1e-12f;
float alpha = rz_old / pAp;
vec_axpy<<<GS,BS>>>(n, alpha, d_p, d_x); // x += alpha p
vec_axpy<<<GS,BS>>>(n,-alpha, d_Ap, d_r); // r -= alpha A p


float r_norm = sqrtf(device_dot(n,d_r,d_r));
if (r_norm / b_norm < tol){ ok=true; break; }


vec_pointwise_div<<<GS,BS>>>(n,d_r,d_M,d_z); // z = M^{-1} r
float rz_new = device_dot(n,d_r,d_z);
float beta = rz_new / (rz_old + 1e-12f);


// p = z + beta p
// p *= beta; p += z;
vec_scale<<<GS,BS>>>(n,beta,d_p);
vec_axpy<<<GS,BS>>>(n,1.0f,d_z,d_p);


rz_old = rz_new;
}


CUDA_CHECK(cudaFree(d_r)); CUDA_CHECK(cudaFree(d_p)); CUDA_CHECK(cudaFree(d_Ap)); CUDA_CHECK(cudaFree(d_z)); CUDA_CHECK(cudaFree(d_M));
return ok;
}






// =============================================================
// ----- FILE: step4_streaming.cu
// Purpose: Real-time optimization patterns — pinned host buffers, pre-alloc
// device memory, multi-stream overlap, and event fencing.
// =============================================================


struct FrameBuffers {
// Device
uint8_t *d_img=nullptr, *d_msk=nullptr; float *d_scores=nullptr; uint2 *d_kp=nullptr; int *d_cnt=nullptr;
// Host pinned
uint8_t *h_img=nullptr, *h_msk=nullptr; int *h_cnt=nullptr;
int W=0,H=0,max_kp=0; cudaStream_t stream=0; cudaEvent_t done;
};


extern "C" void step4_allocate(FrameBuffers& fb, int W, int H, int max_kp){
fb.W=W; fb.H=H; fb.max_kp=max_kp;
CUDA_CHECK(cudaHostAlloc(&fb.h_img, W*H, cudaHostAllocDefault));
CUDA_CHECK(cudaHostAlloc(&fb.h_msk, W*H, cudaHostAllocDefault));
CUDA_CHECK(cudaHostAlloc(&fb.h_cnt, sizeof(int), cudaHostAllocDefault));
CUDA_CHECK(cudaMalloc(&fb.d_img, W*H));
CUDA_CHECK(cudaMalloc(&fb.d_msk, W*H));
CUDA_CHECK(cudaMalloc(&fb.d_scores, W*H*sizeof(float)));
CUDA_CHECK(cudaMalloc(&fb.d_kp, max_kp*sizeof(uint2)));
CUDA_CHECK(cudaMalloc(&fb.d_cnt, sizeof(int)));
CUDA_CHECK(cudaStreamCreateWithFlags(&fb.stream, cudaStreamNonBlocking));
CUDA_CHECK(cudaEventCreateWithFlags(&fb.done, cudaEventDisableTiming));
}


extern "C" void step4_process_async(FrameBuffers& fb){
// H->D async copies
CUDA_CHECK(cudaMemcpyAsync(fb.d_img, fb.h_img, fb.W*fb.H, cudaMemcpyHostToDevice, fb.stream));
CUDA_CHECK(cudaMemcpyAsync(fb.d_msk, fb.h_msk, fb.W*fb.H, cudaMemcpyHostToDevice, fb.stream));
CUDA_CHECK(cudaMemsetAsync(fb.d_cnt, 0, sizeof(int), fb.stream));


dim3 BS(16,16), GS((fb.W+15)/16,(fb.H+15)/16);
fast_score_kernel<<<GS,BS,0,fb.stream>>>(fb.d_img, fb.d_msk, fb.W, fb.H, fb.W, 20, 0.3f, fb.d_scores);


size_t shmem=(16+2)*(16+2)*sizeof(float);
nms3x3_kernel<<<GS,BS,shmem,fb.stream>>>(fb.d_scores, fb.W, fb.H, 5.0f, fb.d_kp, fb.max_kp, fb.d_cnt);


// D->H async copy of count only (results buffer can remain device-resident for downstream)
CUDA_CHECK(cudaMemcpyAsync(fb.h_cnt, fb.d_cnt, sizeof(int), cudaMemcpyDeviceToHost, fb.stream));
CUDA_CHECK(cudaEventRecord(fb.done, fb.stream));
}


extern "C" bool step4_poll_complete(FrameBuffers& fb){ return cudaEventQuery(fb.done) == cudaSuccess; }
extern "C" int step4_get_count(const FrameBuffers& fb){ return *fb.h_cnt; }
extern "C" void step4_free(FrameBuffers& fb){
if(fb.d_img) cudaFree(fb.d_img); if(fb.d_msk) cudaFree(fb.d_msk); if(fb.d_scores) cudaFree(fb.d_scores);
if(fb.d_kp) cudaFree(fb.d_kp); if(fb.d_cnt) cudaFree(fb.d_cnt);
if(fb.h_img) cudaFreeHost(fb.h_img); if(fb.h_msk) cudaFreeHost(fb.h_msk); if(fb.h_cnt) cudaFreeHost(fb.h_cnt);
if(fb.stream) cudaStreamDestroy(fb.stream); if(fb.done) cudaEventDestroy(fb.done);
fb = FrameBuffers{};
}



// =============================================================
// ----- FILE: step5_state_bus.cu
// Purpose: Systems-level integration — time-synced state bus (SPSC ring),
// tail-latency guardrails, and planner-friendly output struct.
// =============================================================


#include <atomic>
#include <cstring>
#include <cmath>


struct StateEstimate {
double t_sec; // monotonic time
float pos[3]; // x,y,z (world)
float quat[4]; // w,x,y,z
float vel[3]; // vx,vy,vz
float cov_pos[3]; // diag covariance for pos (toy)
uint32_t seq; // sequence number
};


// Single-producer single-consumer lock-free ring
template<int CAP>
struct StateRing {
StateEstimate buf[CAP];
std::atomic<uint32_t> head{0}; // write index
std::atomic<uint32_t> tail{0}; // read index


bool push(const StateEstimate& s){
uint32_t h=head.load(std::memory_order_relaxed);
uint32_t t=tail.load(std::memory_order_acquire);
if (((h+1)%CAP)==t) return false; // full
buf[h]=s; head.store((h+1)%CAP, std::memory_order_release); return true;
}
bool pop(StateEstimate& out){
uint32_t t=tail.load(std::memory_order_relaxed);
uint32_t h=head.load(std::memory_order_acquire);
if (t==h) return false; // empty
out=buf[t]; tail.store((t+1)%CAP, std::memory_order_release); return true;
}
};


// Tail-latency watchdog: monitor p95 over sliding window; trigger degrade mode
struct TailLatencyGuard {
std::vector<double> hist_ms; size_t cap; double p95_limit_ms; bool degrade=false;
TailLatencyGuard(size_t cap_, double p95_lim): cap(cap_), p95_limit_ms(p95_lim){}
void add(double ms){ hist_ms.push_back(ms); if(hist_ms.size()>cap) hist_ms.erase(hist_ms.begin()); }
double percentile95() const {
if (hist_ms.empty()) return 0.0; auto v=hist_ms; size_t k=(size_t)(0.95*(v.size()-1)); std::nth_element(v.begin(), v.begin()+k, v.end()); return v[k];
}
void tick(){ degrade = (percentile95() > p95_limit_ms); }
};


// Example planner-facing publisher (toy). In a real system this would be an LCM/ROS2 pub
// and would perform time alignment: t_cam + t_offset -> t_body.
extern "C" void step5_publish_example(StateRing<256>& ring, TailLatencyGuard& g, double t_now,
const float* T_wb_pos, const float* T_wb_quat, const float* vel,
const float* cov_pos_diag, uint32_t seq, double last_frame_ms){
StateEstimate s{}; s.t_sec=t_now; std::memcpy(s.pos,T_wb_pos,3*sizeof(float)); std::memcpy(s.quat,T_wb_quat,4*sizeof(float));
std::memcpy(s.vel,vel,3*sizeof(float)); std::memcpy(s.cov_pos,cov_pos_diag,3*sizeof(float)); s.seq=seq;
bool ok = ring.push(s);
if (!ok) {
// backpressure: drop oldest by popping once, then push
StateEstimate tmp; ring.pop(tmp); ring.push(s);
}
g.add(last_frame_ms); g.tick();
if (g.degrade){
// Signal upstream front-end to reduce load (e.g., higher FAST threshold or fewer features)
// (left as a shared atomic flag in a real system)
fprintf(stderr, "[Step5] Degrade mode ON (p95=%.2fms) -> raise FAST thresh / skip descriptors)\n", g.percentile95());
}
}


// =============================================================
// ----- FILE: demo_main.cu
// Purpose: Tiny driver to show how pieces could be stitched together in tests.
// =============================================================


extern "C" void step1_profile_example(int N,int warmup,int iters);
extern "C" void step2_frontend_example(int W,int H,int max_kp);
extern "C" bool step3_pcg_solve(const CsrMatrix& A, const float* d_b, float* d_x, int iters, float tol);
extern "C" void step4_allocate(FrameBuffers& fb, int W, int H, int max_kp);
extern "C" void step4_process_async(FrameBuffers& fb);
extern "C" bool step4_poll_complete(FrameBuffers& fb);
extern "C" int step4_get_count(const FrameBuffers& fb);
extern "C" void step4_free(FrameBuffers& fb);


int main(){
// Step 1: profile placeholder
step1_profile_example(1<<20, 3, 20);


// Step 2: front-end demo
step2_frontend_example(640,480,2000);


// Step 4: streaming example (async)
FrameBuffers fb{}; step4_allocate(fb, 640, 480, 2000);
// Fill host pinned buffers with dummy data
std::fill(fb.h_img, fb.h_img + fb.W*fb.H, 127);
std::fill(fb.h_msk, fb.h_msk + fb.W*fb.H, 0);
step4_process_async(fb);
while(!step4_poll_complete(fb)) { /* spin/yield */ }
int kp = step4_get_count(fb); printf("[Step4] async kp count = %d\n", kp);
step4_free(fb);


// Step 3: PCG (toy 3x3 SPD system: [[4,1,0],[1,3,0],[0,0,2]])
CsrMatrix A{}; A.n=3; A.nnz=7;
CUDA_CHECK(cudaMalloc(&A.rowPtr,(A.n+1)*sizeof(int)));
CUDA_CHECK(cudaMalloc(&A.colInd,A.nnz*sizeof(int)));
CUDA_CHECK(cudaMalloc(&A.vals,A.nnz*sizeof(float)));
int h_rp[4]={0,3,6,7}; int h_ci[7]={0,1,0,1,2,1,2}; float h_v[7]={4,1,1,1,3,1,2};
CUDA_CHECK(cudaMemcpy(A.rowPtr,h_rp,sizeof(h_rp),cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(A.colInd,h_ci,sizeof(h_ci),cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(A.vals,h_v,sizeof(h_v),cudaMemcpyHostToDevice));
float h_b[3]={1,2,3}; float *d_b=nullptr,*d_x=nullptr; CUDA_CHECK(cudaMalloc(&d_b,3*sizeof(float))); CUDA_CHECK(cudaMalloc(&d_x,3*sizeof(float)));
CUDA_CHECK(cudaMemcpy(d_b,h_b,3*sizeof(float),cudaMemcpyHostToDevice)); vec_set<<<1,32>>>(3,0.f,d_x);
bool ok = step3_pcg_solve(A,d_b,d_x,100,1e-6f);
float h_x[3]; CUDA_CHECK(cudaMemcpy(h_x,d_x,3*sizeof(float),cudaMemcpyDeviceToHost));
printf("[Step3] PCG ok=%d solution ~ [%.4f %.4f %.4f]\n", ok, h_x[0],h_x[1],h_x[2]);
cudaFree(A.rowPtr); cudaFree(A.colInd); cudaFree(A.vals); cudaFree(d_b); cudaFree(d_x);


// Step 5: publisher demo
StateRing<256> ring{}; TailLatencyGuard guard(64, /*p95 limit*/ 8.0);
float pos[3]={0,0,0}, quat[4]={1,0,0,0}, vel[3]={0,0,0}, cov[3]={0.1f,0.1f,0.1f};
for(int i=0;i<10;++i){ step5_publish_example(ring,guard, (double)i*0.02, pos, quat, vel, cov, i, /*last_frame_ms*/ 5.0 + (i%3)); }
printf("[Step5] Degrade=%s (p95=%.2fms)\n", guard.degrade?"true":"false", guard.percentile95());


CUDA_CHECK(cudaDeviceSynchronize());
return 0;
}